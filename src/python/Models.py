
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import pytorch_tps

class PA(nn.Module):
    ''' 
    '''
    def __init__(self,channel):
        super(PA, self).__init__()
        self.conv1 = nn.Conv2d(channel, channel, 1, padding=0, bias=True)
        self.sig = nn.Sigmoid()
        def _initialize_weights(m):
            if type(m) == nn.Conv2d:
                nn.init.kaiming_normal_(m.weight)

        self.apply(_initialize_weights)

    def forward(self, x):
        c1_ = self.conv1(x)
        c1 = self.sig(c1_)
        
        return x * c1

class PANet(nn.Module):
    def __init__(self):
        super(PANet, self).__init__()
        self.name = self.__class__.__name__
        self.relu = nn.ReLU()
        self.scale = 4

        self.simplified = False
        self.unshuffle = nn.PixelUnshuffle(self.scale)
        self.pa1 = PA(48)
        self.pa2 = PA(48)

        # siamese encoder
        self.conv1 = nn.Conv2d(3*self.scale*self.scale, 48, 3, 2, 1)
        self.conv2 = nn.Conv2d(48, 64, 3, 2, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv4 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv5 = nn.Conv2d(256, 128, 3, 1, 1)

        # transposed conv
        self.transConv1 = nn.ConvTranspose2d(128, 64, 2, 2, 0)
        self.transConv2 = nn.ConvTranspose2d(64, 48, 2, 2, 0)

        # output layer
        self.conv6 = nn.Conv2d(48, 3*self.scale*self.scale, 3, 1, 1)
        self.shuffle = nn.PixelShuffle(self.scale)

        # skip layers (see s3 in forward)
        # s1
        self.skipConv11 = nn.Conv2d(48, 48, 1, 1, 0)
        self.skipConv12 = nn.Conv2d(48, 48, 3, 1, 1)
        self.skipConv13 = nn.Conv2d(48, 48, 3, 1, 1)

        # s2
        self.skipConv21 = nn.Conv2d(48, 64, 1, 1, 0)
        self.skipConv22 = nn.Conv2d(64, 64, 3, 1, 1)

   

        # stores biases of surface feature branch (net simplification)
        self.register_buffer('res1_s_pre', None)
        self.register_buffer('res2_s_pre', None)
        self.register_buffer('res3_s_pre', None)

        # initialization function, first checks the module type,
        def _initialize_weights(m):
            if type(m) == nn.Conv2d:
                nn.init.kaiming_normal_(m.weight)

        self.apply(_initialize_weights)

    # simplify trained model by trimming surface branch to biases
 

    def simplify(self, s):
        s = (self.unshuffle(s))
        s = self.pa1(s)


        res1_s = self.relu(self.skipConv11(s))
        res1_s = self.relu(self.skipConv12(res1_s))
        res1_s = self.skipConv13(res1_s)
        self.res1_s_pre = res1_s

        s = (self.conv1(s))
        s = self.pa2(s)
        

        res2_s = self.skipConv21(s)
        res2_s = self.relu(res2_s)
        res2_s = self.skipConv22(res2_s)
        self.res2_s_pre = res2_s

        s = self.relu(self.conv2(s))
        s = self.relu(self.conv3(s))
        self.res3_s_pre = s

        self.res1_s_pre = self.res1_s_pre.squeeze()
        self.res2_s_pre = self.res2_s_pre.squeeze()
        self.res3_s_pre = self.res3_s_pre.squeeze()

        self.simplified = True

    # x is the input uncompensated image, s is a surface image
    def forward(self, x, s):
        # surface feature extraction

        # alternate between surface and image branch
        if self.simplified:
            res1_s = self.res1_s_pre
            res2_s = self.res2_s_pre
            res3_s = self.res3_s_pre
        else:
            s = self.unshuffle(s)
            s = self.pa1(s)

            res1_s = self.relu(self.skipConv11(s))
            res1_s = self.relu(self.skipConv12(res1_s))
            res1_s = self.skipConv13(res1_s)

            s = (self.conv1(s))
            s = self.pa2(s)


            res2_s = self.skipConv21(s)
            res2_s = self.relu(res2_s)
            res2_s = self.skipConv22(res2_s)

            s = self.relu(self.conv2(s))
            res3_s = self.relu(self.conv3(s))


        x = self.unshuffle(x)
        x = self.pa1(x)


        res1 = self.relu(self.skipConv11(x))
        res1 = self.relu(self.skipConv12(res1))
        res1 = self.skipConv13(res1)
        res1 = res1 -res1_s

        x = (self.conv1(x))
        x = self.pa2(x)

        res2 = self.skipConv21(x)
        res2 = self.relu(res2)
        res2 = self.skipConv22(res2)
        res2 = res2- res2_s

        x = self.relu(self.conv2(x))

        x = self.relu(self.conv3(x))
        x = x-res3_s # s3

        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))


        x = self.relu(self.transConv1(x) +res2)
    

        x = self.relu(self.transConv2(x)+res1)
        x = (self.conv6(x))
        x = self.shuffle(x)
        x = torch.clamp(x, max=1)
        x = torch.clamp(x, min=0)

        return x

class GridRefine(nn.Module):
    def __init__(self):
        super(GridRefine, self).__init__()
        self.name = self.__class__.__name__        
        # grid refinement net
        self.conv1 = nn.Conv2d(2, 32, 7, 4, 3)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(64, 32, 3, 1, 1)
        self.conv2f = nn.Conv2d(32, 32, 1, 1, 0)
        self.conv31 = nn.Conv2d(64, 64, 7, 4, 3)
        self.conv32 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv33 = nn.Conv2d(128, 64, 3, 1, 1)
        self.conv3f = nn.Conv2d(64, 64, 1, 1, 0)


        self.trans1 = nn.ConvTranspose2d(32, 2, 4, 4, 0)
        self.trans2 = nn.ConvTranspose2d(64, 64, 4, 4, 0)
        self.relu =  nn.ReLU()
        self.sig =  nn.Sigmoid()
        self.lrelu =  nn.LeakyReLU()

        # initialization function, first checks the module type,
        def _initialize_weights(m):
            if type(m) == nn.Conv2d:
                nn.init.normal_(m.weight, 0, 1e-4)
        self.apply(_initialize_weights)

    def forward(self, x):
        # surface feature extraction
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x31 = self.relu(self.conv31(x2))
        x32 = self.relu(self.conv32(x31))
        x3_f = self.sig(self.conv3f(x31))
        x33 = self.relu(self.conv33(x32))
        x3_out = self.lrelu(self.trans2(x33*x3_f))
        x3 = self.relu(self.conv3(x3_out))
        x2_f = self.sig(self.conv2f(x1))
        out = self.lrelu(self.trans1(x3*x2_f))
        return x+out


    def __init__(self):
        super(GridRefine, self).__init__()
        self.name = self.__class__.__name__        
        f = 32
        self.conv1 = nn.Conv2d(2, f, kernel_size=3, stride=2, padding=1)
        self.conv_f = nn.Conv2d(f, f, kernel_size=1)
        self.conv2 = nn.Conv2d(f, f, kernel_size=3, stride=2, padding=1)
       
        self.transConv1 = nn.ConvTranspose2d(f, f, 3, 2, 1, 1)
        self.transConv2 = nn.ConvTranspose2d(f, 2, 2, 2, 0)

        self.sigmoid = nn.Sigmoid()
        self.lrelu = nn.LeakyReLU(0.1)
        self.relu = nn.ReLU(inplace=True)
        def _initialize_weights(m):
            if type(m) == nn.Conv2d:
                nn.init.normal_(m.weight, 0, 1e-4)

        self.apply(_initialize_weights)

    def forward(self, x):
        c1_ = self.relu(self.conv1(x))
        c2 = self.relu(self.conv2(c1_))
        
        c3 = self.relu(self.transConv1(c2))
        
        cf = self.sigmoid(self.conv_f(c1_))
        c4 = self.transConv2(c3*cf)
        m = self.lrelu(c4)
        return m+x

class GANet(nn.Module):
    def __init__(self, grid_shape=(5, 5), out_size=(1024, 1024), with_refine=True):
        super(GANet, self).__init__()
        self.grid_shape = grid_shape
        self.out_size =  out_size
        self.with_refine = with_refine  # becomes WarpingNet w/o refine if set to false
        self.name = 'GANet' if not with_refine else 'GANet_without_refine'

        # relu
        self.relu = nn.ReLU()
        self.leakyRelu = nn.LeakyReLU(0.1)

        # final refined grid
        self.register_buffer('fine_grid', None)

        # affine params
        self.affine_mat = nn.Parameter(torch.Tensor([1, 0, 0, 0, 1, 0]).view(-1, 2, 3))

        # tps params
        self.nctrl = self.grid_shape[0] * self.grid_shape[1]
        self.nparam = (self.nctrl + 2)
        ctrl_pts = pytorch_tps.uniform_grid(grid_shape)
        self.register_buffer('ctrl_pts', ctrl_pts.view(-1, 2))
        self.theta = nn.Parameter(torch.ones((1, self.nparam * 2), dtype=torch.float32).view(-1, self.nparam, 2) * 1e-3)


        # grid refinement net
        if self.with_refine:
            self.grid_refine_net = GridRefine()
        else:
            self.grid_refine_net = None  # WarpingNet w/o refine

    # initialize WarpingNet's affine matrix to the input affine_vec
    def set_affine(self, affine_vec):
        self.affine_mat.data = torch.Tensor(affine_vec).view(-1, 2, 3)

    # simplify trained model to a single sampling grid for faster testing
    def simplify(self, x):
        # generate coarse affine and TPS grids
        coarse_affine_grid = F.affine_grid(self.affine_mat, torch.Size([1, x.shape[1], x.shape[2], x.shape[3]])).permute((0, 3, 1, 2))
        coarse_tps_grid = pytorch_tps.tps_grid(self.theta, self.ctrl_pts, (1, x.size()[1]) + self.out_size)

        # use TPS grid to sample affine grid
        tps_grid = F.grid_sample(coarse_affine_grid, coarse_tps_grid)

        # refine TPS grid using grid refinement net and save it to self.fine_grid
        if self.with_refine:
            self.fine_grid = torch.clamp(self.grid_refine_net(tps_grid), min=-1, max=1).permute((0, 2, 3, 1))
        else:
            self.fine_grid = torch.clamp(tps_grid, min=-1, max=1).permute((0, 2, 3, 1))

    def forward(self, x):

        if self.fine_grid is None:
            # not simplified (training/validation)
            # generate coarse affine and TPS grids
            coarse_affine_grid = F.affine_grid(self.affine_mat, torch.Size([1, x.shape[1], x.shape[2], x.shape[3]])).permute((0, 3, 1, 2))

            coarse_tps_grid = pytorch_tps.tps_grid(self.theta, self.ctrl_pts, (1, x.size()[1]) + self.out_size)

            # use TPS grid to sample affine grid
            tps_grid = F.grid_sample(coarse_affine_grid, coarse_tps_grid).repeat(x.shape[0], 1, 1, 1)

            # refine TPS grid using grid refinement net and save it to self.fine_grid
            if self.with_refine:
                fine_grid = torch.clamp(self.grid_refine_net(tps_grid), min=-1, max=1).permute((0, 2, 3, 1))
                fine_grid = fine_grid
  
            else:
                fine_grid = torch.clamp(tps_grid, min=-1, max=1).permute((0, 2, 3, 1))
        else:
            fine_grid = self.fine_grid.repeat(x.shape[0], 1, 1, 1)

        # warp
        x = F.grid_sample(x, fine_grid)
        return x

class CompenHD(nn.Module):
    def __init__(self, ga_net=None, pa_net=None):
        super(CompenHD, self).__init__()
        self.name = self.__class__.__name__

        # initialize from existing models or create new models
        self.ga_net = copy.deepcopy(ga_net.module) if ga_net is not None else GANet()
        self.pa_net = copy.deepcopy(pa_net.module) if pa_net is not None else PANet()

    # simplify trained model to a single sampling grid for faster testing
    def simplify(self, s):
        self.ga_net.simplify(s)
        self.pa_net.simplify(self.ga_net(s))

    # s is Bx3x256x256 surface image
    def forward(self, x, s):
        # geometric correction using WarpingNet (both x and s)
        x = self.ga_net(x)
        s = self.ga_net(s)

        # photometric compensation using CompenNet
        x = self.pa_net(x, s)

        return x

               