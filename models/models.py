'''
#  filename: model.py
#  Likun Qin, 2021
'''

import os
import torch
import torch.nn as nn
# from ME import MEModule
# from r2plus1d import r2plus1d_18_layer
from models.mce import MCEModule
from models.r2plus1d import r2plus1d_18_layer


class ConvLSTM_Cell(nn.Module):
    def __init__(self, input_shape, input_dim, hidden_dim, kernel_size, bias=1):
        """
        input_shape: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """
        super(ConvLSTM_Cell, self).__init__()

        self.height, self.width = input_shape
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        self.n_segment = 6

        self.conv = nn.Conv2d(in_channels=2 * self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding, bias=self.bias)

        self.me = MCEModule(channel=input_dim, reduction=4)

        self.longterm = r2plus1d_18_layer(planes=input_dim)

        self.conv_agg = nn.Conv2d(in_channels=self.input_dim * self.n_segment,
                                  out_channels=self.input_dim, kernel_size=(1, 1),
                                  bias=False)
        self.bn_agg = nn.BatchNorm2d(num_features=self.input_dim)

        # build edge_index
        # indices = []
        # for i in range(self.height * self.width):
        #     for j in range(self.height * self.width):
        #         if i != j:
        #             indices.append([i, j])
        # edge_indices = torch.tensor(indices, dtype=torch.long)
        # edge_indices = edge_indices.t().contiguous()
        # self.edge_indices = nn.Parameter(edge_indices, requires_grad=False)
        #
        # self.gcn = GCNConv(in_channels=4 * self.hidden_dim,
        #                     out_channels=4 * self.hidden_dim)

    # we implement LSTM that process only one timestep
    def forward(self, x, ex, hidden, timestep):  # x [batch, hidden_dim, width, height]
        '''

        :param x: input, [batch_size, self.input_dim, 16, 16]
        :param ex: preivous input features, [batch_size, self.input_dim, 5, 16, 16]
        :param hidden: [h, c], h,c : [batch_size, hidden_dim, 16, 16]
        :param timestep: the position in a sequence, int
        :return:
        '''
        h_cur, c_cur = hidden
        nt, c, h, w = x.size()

        # build ex_input for next forward pass
        all = torch.cat([torch.unsqueeze(x, dim=2), ex], dim=2)    # [batch_size, self.input_dim, 6, 16, 16]
        ex_input = all[:, :, :-1, :, :]   # [batch_size, self.input_dim, 5, 16, 16]

        # enhance short-term channels
        weighted_x = self.me(x, ex[:, :, 0, :, :], timestep)  # [batch_size, self.input_dim, 16, 16]

        # long-term
        if timestep < self.n_segment - 1:
            aggregation = x
        else:
            aggregation = self.longterm(all)  # [batch_size, self.input_dim, 6, 16, 16]
            aggregation = aggregation.contiguous().view(nt, c * self.n_segment, h, w)
            aggregation = self.conv_agg(aggregation)  # [batch_size, self.input_dim, 16, 16]
            aggregation = self.bn_agg(aggregation)  # [batch_size, self.input_dim, 16, 16]

        combined = torch.cat([weighted_x, aggregation, h_cur], dim=1)  # concatenate along channel axis, N C H W
        # combined = self.me(combined, ex)    # N input_dim+hidden_dim H W
        combined_conv = self.conv(combined)  # [batch_size, 4 * hidden_dim, 16, 16]

        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next, ex_input

    def init_hidden(self, batch_size, image_size):
        '''
        initialize hidden states
        :param batch_size: int
        :param image_size: tuple of ints, (height, width)
        :return: zero tensors
        '''
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))

    def init_ex_input(self, batch_size, image_size):
        '''
        initialize previous input for the first input in a sequence
        :param batch_size: int
        :param image_size: tuple of ints, (height, width)
        :return: zero tensor
        '''
        height, width = image_size
        return torch.zeros(batch_size, self.input_dim, 5, height, width, device=self.conv.weight.device)


class ConvLSTM(nn.Module):
    def __init__(self, input_shape, input_dim, hidden_dims, n_layers, kernel_size):
        super(ConvLSTM, self).__init__()
        self.input_shape = input_shape
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.n_layers = n_layers
        self.kernel_size = kernel_size

        # H, C = [],[]
        #
        # for i in range(self.n_layers):
        #     H.append(nn.Parameter(torch.zeros(batch,self.hidden_dims[i], self.input_shape[0],
        #                                        self.input_shape[1]), requires_grad=False
        #                           )
        #             )
        #     C.append(nn.Parameter(torch.zeros(batch,self.hidden_dims[i], self.input_shape[0],
        #                                       self.input_shape[1]), requires_grad=False
        #                           )
        #              )
        # self.H = nn.ParameterList(H)
        # self.C = nn.ParameterList(C)

        cell_list = []
        for i in range(0, self.n_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dims[i - 1]
            print('layer ', i, 'input dim ', cur_input_dim, ' hidden dim ', self.hidden_dims[i])
            cell_list.append(ConvLSTM_Cell(input_shape=self.input_shape,
                                           input_dim=cur_input_dim,
                                           hidden_dim=self.hidden_dims[i],
                                           kernel_size=self.kernel_size))
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_, ex_input, hidden, timestep):
        '''

        :param input_: input tensor, [batch_size, 64, 16, 16]
        :param ex_input: previous input tensor, [batch, 64, 16, 16]
        :param hidden: a list of hidden states, [H, C]
                       H, C:  a list of ndarray, [batch_size, hidden_dim, 16, 16]
        :param timestep: the number indicating current position in a sequence, int
        :return:
        '''
        batch_size = input_.data.size()[0]
        cell_inputs = []

        # set hidden to zeros if it is the start of the sequence
        if timestep == 0 or not hidden:
            self._init_hidden(batch_size)  # init Hidden at each forward start
        else:
            self.setHidden(hidden)

        # set ex_input to zeros if it is the start of the sequence
        if timestep == 0 or not ex_input:
            ex_input = self._init_ex_input(batch_size)

        for j, cell in enumerate(self.cell_list):
            if j == 0:  # bottom layer
                self.H[j], self.C[j], ex = \
                    cell(x=input_, ex=ex_input[j], hidden=(self.H[j], self.C[j]), timestep=timestep)
                cell_inputs.append(ex)
            else:
                self.H[j], self.C[j], ex = \
                    cell(x=self.H[j - 1], ex=ex_input[j], hidden=(self.H[j], self.C[j]), timestep=timestep)
                cell_inputs.append(ex)
        return (self.H, self.C), self.H, cell_inputs  # (hidden, output, cell_inputs)

    def _init_hidden(self, batch_size):
        self.C = []
        self.H = []
        for i in range(self.n_layers):
            c, h = self.cell_list[i].init_hidden(batch_size, self.input_shape)
            self.C.append(c)
            self.H.append(h)

    def setHidden(self, hidden):
        H, C = hidden
        self.H, self.C = H, C

    def _init_ex_input(self, batch_size):
        ex = []
        for i in range(self.n_layers):
            one = self.cell_list[i].init_ex_input(batch_size, self.input_shape)
            ex.append(one)
        return ex

    # def set_ex_input(self, ex_input):
    #     self.ex = ex_input


class dcgan_conv(nn.Module):
    def __init__(self, nin, nout, stride):
        super(dcgan_conv, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels=nin, out_channels=nout, kernel_size=(3, 3), stride=stride, padding=1),
            nn.GroupNorm(16, nout),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, input):
        return self.main(input)


class dcgan_upconv(nn.Module):
    def __init__(self, nin, nout, stride):
        super(dcgan_upconv, self).__init__()
        if (stride == 2):
            output_padding = 1
        else:
            output_padding = 0
        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_channels=nin, out_channels=nout, kernel_size=(3, 3), stride=stride, padding=1,
                               output_padding=output_padding),
            nn.GroupNorm(16, nout),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, input):
        return self.main(input)


class encoder_E(nn.Module):
    def __init__(self, nc=1, nf=32):
        super(encoder_E, self).__init__()
        # input is (1) x 64 x 64
        self.c1 = dcgan_conv(nc, nf, stride=2)  # (32) x 32 x 32
        self.c2 = dcgan_conv(nf, nf, stride=1)  # (32) x 32 x 32
        self.c3 = dcgan_conv(nf, 2 * nf, stride=2)  # (64) x 16 x 16

    def forward(self, input):
        h1 = self.c1(input)
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        return h3


class decoder_D(nn.Module):
    def __init__(self, nc=1, nf=32):
        super(decoder_D, self).__init__()
        self.upc1 = dcgan_upconv(2 * nf, nf, stride=2)  # (32) x 32 x 32
        self.upc2 = dcgan_upconv(nf, nf, stride=1)  # (32) x 32 x 32
        self.upc3 = nn.ConvTranspose2d(in_channels=nf, out_channels=nc, kernel_size=(3, 3), stride=2, padding=1,
                                       output_padding=1)  # (nc) x 64 x 64

    def forward(self, input):
        d1 = self.upc1(input)
        d2 = self.upc2(d1)
        d3 = self.upc3(d2)
        return d3


class encoder_specific(nn.Module):
    def __init__(self, nc=64, nf=64):
        super(encoder_specific, self).__init__()
        self.c1 = dcgan_conv(nc, nf, stride=1)  # (64) x 16 x 16
        self.c2 = dcgan_conv(nf, nf, stride=1)  # (64) x 16 x 16

    def forward(self, input):
        h1 = self.c1(input)
        h2 = self.c2(h1)
        return h2


class decoder_specific(nn.Module):
    def __init__(self, nc=64, nf=64):
        super(decoder_specific, self).__init__()
        self.upc1 = dcgan_upconv(nf, nf, stride=1)  # (64) x 16 x 16
        self.upc2 = dcgan_upconv(nf, nc, stride=1)  # (32) x 32 x 32

    def forward(self, input):
        d1 = self.upc1(input)
        d2 = self.upc2(d1)
        return d2


class EncoderRNN(torch.nn.Module):
    def __init__(self, input_dim=1, input_shape=(64, 64)):
        super(EncoderRNN, self).__init__()
        self.encoder_E = encoder_E(nc=input_dim)  # general encoder 64x64x1 -> 32x32x32
        # self.encoder_Ep = encoder_specific()  # specific image encoder 32x32x32 -> 16x16x64
        self.encoder_Er = encoder_specific()
        # self.decoder_Dp = decoder_specific()  # specific image decoder 16x16x64 -> 32x32x32
        self.decoder_Dr = decoder_specific()
        self.decoder_D = decoder_D(nc=input_dim)  # general decoder 32x32x32 -> 64x64x1

        # self.phycell = PhyCell(input_shape=(16, 16), input_dim=64, F_hidden_dims=[49],
        #                        n_layers=1, kernel_size=(7, 7))
        self.convcell = ConvLSTM(input_shape=(input_shape[0] // 4, input_shape[1] // 4),
                                 input_dim=64, hidden_dims=[128, 128, 64],
                                 n_layers=3, kernel_size=(3, 3))

        # self.convcell.edge_indices = self.convcell.edge_indices.to(device)
        # for i in range(self.convcell.n_layers):
        #     self.convcell.cell_list[i].edge_indices = self.convcell.cell_list[i].edge_indices.to(device)

    def forward(self, input, ex_input_conv=None, hidden1=None, hidden2=None,
                timestep=0, decoding=False):
        '''

        :param input:
        :param ex_input_conv:
        :param hidden1:
        :param hidden2:
        :param timestep:
        :param decoding:
        :return:
        '''

        input = self.encoder_E(input)  # general encoder 1x64x64 -> 64x16x16

        # if decoding:  # input=None in decoding phase
        #     input_phys = None
        # else:
        #     input_phys = self.encoder_Ep(input)  # 64x16x16
        input_conv = self.encoder_Er(input)  # 64x16x16

        # hidden1, output1 = self.phycell(input_phys, hidden1, first_timestep)
        hidden2, output2, cell_inputs2 = self.convcell(input_=input_conv, ex_input=ex_input_conv,
                                                       hidden=hidden2, timestep=timestep)

        # decoded_Dp = self.decoder_Dp(output1[-1])
        decoded_Dr = self.decoder_Dr(output2[-1])

        # out_phys = torch.sigmoid(self.decoder_D(decoded_Dp))  # partial reconstructions for vizualization
        out_conv = torch.sigmoid(self.decoder_D(decoded_Dr))

        # concat = decoded_Dp + decoded_Dr
        # concat = decoded_Dr
        # output_image = torch.sigmoid(self.decoder_D(concat))

        return out_conv, cell_inputs2, out_conv, hidden1, hidden2

    def get_optim_policies(self):

        ops_pool = (torch.nn.Conv2d, torch.nn.Conv1d, torch.nn.Conv3d,
                    torch.nn.Linear,
                    torch.nn.modules.batchnorm._BatchNorm,
                    torch.nn.ConvTranspose1d, torch.nn.ConvTranspose2d, torch.nn.ConvTranspose3d
                    )

        modules = list(self.modules())

        # use pretrained weights, set lr to 1/10
        encoder_decoder = []

        # use pretrained weights, set lr to 3/10
        r2plus1d = []

        # train from scratch
        convlstm = []

        for i, m in enumerate(self.modules()):

            # if not isinstance(m, torch.nn.Conv2d) and not isinstance(m, torch.nn.Conv1d) \
            #         and not isinstance(m, torch.nn.Conv3d) and not isinstance(m, torch.nn.Linear) \
            #         and not isinstance(m, torch.nn.BatchNorm1d) and not isinstance(m, torch.nn.BatchNorm2d) \
            #         and not isinstance(m, torch.nn.BatchNorm3d) and not isinstance(m, torch.nn.InstanceNorm1d) \
            #         and not isinstance(m, torch.nn.InstanceNorm2d) and not isinstance(m, torch.nn.InstanceNorm3d):
            #     continue
            if not isinstance(m, ops_pool):
                continue

            ps = list(m.parameters())
            if i < 51:
                # weights for encoder or decoder
                encoder_decoder.extend(ps)
            else:
                if (i >= 66 and i <= 72) or (i >=87 and i <= 102) or (i >= 117 and i <= 123) or\
                    (i >= 139 and i <= 145) or (i >= 160 and i <= 175) or (i >= 190 and i <= 196) or \
                        (i >= 212 and i <= 218) or (i >= 233 and i <= 248) or (i >= 263 and i<= 269):
                    r2plus1d.extend(ps)
                # weights for convlstm
                else:
                    convlstm.extend(ps)

        return [
            {
                'params': encoder_decoder,
                'lr_mult': 0.1,
                'name': "encoder_decoder",
            },
            {
                'params': r2plus1d,
                'lr_mult': 0.3,
                'name': "r2plus1d",
            },
            {
                'params': convlstm,
                'lr_mult': 1,
                'name': "convlstm",
            }
        ]


def load_partial_dict(net_ckpt, model, loc, layer_ckpt='./save/r2plus1d_layer.pth'):
    '''

    :param net_ckpt: the path of model file, str
    :param model: an instance of the network
    :param loc: location of the model
    :return:
    '''

    model_dict = model.state_dict()
    if net_ckpt and os.path.exists(net_ckpt):
        print('using part of existing model')
        print('===> loading existing model: ', net_ckpt)
        # weights = torch.load(net_ckpt, map_location=loc)
        weights = torch.load(net_ckpt, map_location=loc)['phydnet']
        backbone_weights = {k: v for k, v in weights.items()
                            if k in model_dict.keys() and ('convcell' not in k or 'me' in k)}
                            # if k in model_dict.keys() and 'convcell' not in k }
        model_dict.update(backbone_weights)
        # regressor_dict = regressor.state_dict()
        # regressor_weight = {k[j:]: v for k, v in weights.items() if k[j:] in regressor_dict.keys()}

    if layer_ckpt and os.path.exists(layer_ckpt):
        print('==> loading r(2+1)d pretrained model: ', layer_ckpt)
        layer_weights = torch.load(layer_ckpt, map_location=loc)
        channels = [model.convcell.input_dim]
        channels.extend(model.convcell.hidden_dims[:-1])
        for i, input_dim in enumerate(channels):
            prefix = 'convcell.cell_list.{}.longterm.'.format(i)
            if input_dim == 64:
                layer_w = {prefix + k: v for k, v in layer_weights['layer1'].items()}
            elif input_dim == 128:
                layer_w = {prefix + k: v for k, v in layer_weights['layer2'].items()}
            else:
                layer_w = {prefix + k: v for k, v in layer_weights['layer3'].items()}
            model_dict.update(layer_w)

    model.load_state_dict(model_dict)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    # convlstm = ConvLSTM(input_shape=(16, 16), input_dim=64, hidden_dims=[128, 128, 64],
    #                              n_layers=3, kernel_size=(3, 3))
    # print(count_parameters(convlstm.cell_list[2]))
    encoder = EncoderRNN()
    policies = encoder.get_optim_policies()
    print(count_parameters(encoder))
    print('done!')
