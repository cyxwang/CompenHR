'''
CompenHD training functions
'''

from utils import *
import ImgProc
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import time
import visdom
from differential_color_function import rgb2lab_diff, ciede2000_diff
# for visualization
vis = visdom.Visdom()  # default port is 8097
assert vis.check_connection(), 'Visdom: No connection, start visdom first!'

# loss functions
l1_fun = nn.L1Loss()
l2_fun = nn.MSELoss()
ssim_fun = pytorch_ssim.SSIM().cuda()

# %% load training and validation data for CompenNe(S)t++
def loadData(dataset_root, data_name, input_size, data_type='raw', CompenNeSt_only=False):
    if CompenNeSt_only:
        data_type = 'warpSL'

    # data paths
    data_root = fullfile(dataset_root, data_name)
    cam_ref_path = fullfile(data_root, 'cam/{}/ref'.format(data_type))
    cam_train_path = fullfile(data_root, 'cam/{}/train'.format(data_type))
    prj_train_path = fullfile(dataset_root, 'train')
    cam_valid_path = fullfile(data_root, 'cam/{}/test'.format(data_type))
    prj_valid_path = fullfile(dataset_root, 'test')
    print("Loading data from '{}'".format(data_root))

    # training data
    cam_surf = readImgsMT(cam_ref_path, index=[125])  # ref/img_0126.png is cam-captured surface image i.e., s when img_gray.png i.e., x0 projected
    cam_train = readImgsMT(cam_train_path)

    prj_train = readImgsMT(prj_train_path,size=input_size)

    # validation data
    cam_valid = readImgsMT(cam_valid_path)
    prj_valid = readImgsMT(prj_valid_path,size=input_size)

    mask_corners = None
    if not CompenNeSt_only:
        # find projector FOV mask
        im_diff = readImgsMT(cam_ref_path, index=[124]) - readImgsMT(cam_ref_path, index=[0])
        im_diff = im_diff.numpy().transpose((2, 3, 1, 0))
        prj_fov_mask = torch.zeros(cam_surf.shape)

        # threshold im_diff with Otsu's method
        mask_corners = [None] * im_diff.shape[-1]
        for i in range(im_diff.shape[-1]):
            im_mask, mask_corners[i] = ImgProc.thresh(im_diff[:, :, :, i])
            prj_fov_mask[i, :, :, :] = repeat_np(torch.Tensor(np.uint8(im_mask)).unsqueeze(0), 3, 0)

        prj_fov_mask = prj_fov_mask.bool()

        # mask out background areas that are out of projector's FOV
        cam_surf[~prj_fov_mask] = 0

        cam_train = torch.where(prj_fov_mask, cam_train, torch.tensor([0.]))
        cam_valid = torch.where(prj_fov_mask, cam_valid, torch.tensor([0.]))

    return cam_surf, cam_train, cam_valid, prj_train, prj_valid, mask_corners

def loadTestData(dataset_root, data_name, input_size, data_type='raw', CompenNeSt_only=False):
    if CompenNeSt_only:
        data_type = 'warpSL'

    # data paths
    data_root = fullfile(dataset_root, data_name)
    cam_ref_path = fullfile(data_root, 'cam/{}/ref'.format(data_type))
    cam_valid_path = fullfile(data_root, 'cam/{}/test'.format(data_type))
    prj_valid_path = fullfile(dataset_root, 'test')
    print("Loading data from '{}'".format(data_root))

    # training data
    cam_surf = readImgsMT(cam_ref_path, index=[125])  # ref/img_0126.png is cam-captured surface image i.e., s when img_gray.png i.e., x0 projected
    

    # validation data
    cam_valid = readImgsMT(cam_valid_path)
    prj_valid = readImgsMT(prj_valid_path,size=input_size)

    mask_corners = None
    if not CompenNeSt_only:
        # find projector FOV mask
        im_diff = readImgsMT(cam_ref_path, index=[124]) - readImgsMT(cam_ref_path, index=[0])
        im_diff = im_diff.numpy().transpose((2, 3, 1, 0))
        prj_fov_mask = torch.zeros(cam_surf.shape)

        # threshold im_diff with Otsu's method
        mask_corners = [None] * im_diff.shape[-1]
        for i in range(im_diff.shape[-1]):
            im_mask, mask_corners[i] = ImgProc.thresh(im_diff[:, :, :, i])
            prj_fov_mask[i, :, :, :] = repeat_np(torch.Tensor(np.uint8(im_mask)).unsqueeze(0), 3, 0)

        prj_fov_mask = prj_fov_mask.bool()

        # mask out background areas that are out of projector's FOV
        cam_surf[~prj_fov_mask] = 0

        cam_valid = torch.where(prj_fov_mask, cam_valid, torch.tensor([0.]))

    return cam_surf, cam_valid, prj_valid, mask_corners


def trainModel(net, train_data, valid_data, train_option):
    device = train_option['device']

    if device.type == 'cuda': torch.cuda.empty_cache()

    # training data
    cam_surf_train = train_data['cam_surf']
    cam_train = train_data['cam_train']
    prj_train = train_data['prj_train']
 


    # list of parameters to be optimized
    params_ga = net.module.ga_net.parameters()
    params_pa = net.module.pa_net.parameters()
    params = [
        {'params': params_ga, 'lr': train_option['lr_cmp']},
        {'params': params_pa, 'lr': train_option['lr_warp']},
    ]
    optimizer = optim.Adam(params, weight_decay=train_option['l2_reg'])

    # learning rate drop scheduler
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=train_option['lr_drop_rate'], gamma=train_option['lr_drop_ratio'])

    # %% start train
    start_time = time.time()
    time_lapse = time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))

    # get model name
    if not 'model_name' in train_option: train_option['model_name'] = net.name if hasattr(net, 'name') else net.module.name

    # initialize visdom data visualization figure
    if 'plot_on' not in train_option: train_option['plot_on'] = True

    # title string of current training option
    title = optionToString(train_option)

    if train_option['plot_on']:
        # intialize visdom figures
        vis_train_fig = None
        vis_valid_fig = None
        vis_train_fig2 = None
        vis_valid_fig2 = None
        vis_curve_fig = vis.line(X=np.array([0]), Y=np.array([0]), name='origin',
                                 opts=dict(width=1300, height=500, markers=True, markersize=3,
                                           layoutopts=dict(
                                               plotly=dict(title={'text': title, 'font': {'size': 24}},
                                                           font={'family': 'Arial', 'size': 20},
                                                           hoverlabel={'font': {'size': 20}},
                                                           xaxis={'title': 'Iteration'},
                                                           yaxis={'title': 'Metrics', 'hoverformat': '.4f'}))))
    # main loop
    iters = 0

    while iters < train_option['max_iters']:
        # randomly sample training batch and send to GPU
        idx = random.sample(range(train_option['num_train']), train_option['batch_size'])
        cam_surf_train_batch = cam_surf_train[idx, :, :, :].to(device) if cam_surf_train.device.type != 'cuda' else cam_surf_train[idx, :, :, :]
        cam_train_batch = cam_train[idx, :, :, :].to(device) if cam_train.device.type != 'cuda' else cam_train[idx, :, :, :]
        prj_train_batch = prj_train[idx, :, :, :].to(device) if prj_train.device.type != 'cuda' else prj_train[idx, :, :, :]

        # predict and compute loss
        net.train()  
        prj_train_pred = predict(net, dict(cam=cam_train_batch, cam_surf=cam_surf_train_batch))

        if train_option['pre-trained']:
            if iters <=200:
                train_loss_batch, train_l2_loss_batch = computeLoss(prj_train_pred, prj_train_batch, 'l1')
            else:
                train_loss_batch, train_l2_loss_batch = computeLoss(prj_train_pred, prj_train_batch, train_option['loss'])  #  
        else:
            train_loss_batch, train_l2_loss_batch = computeLoss(prj_train_pred, prj_train_batch, train_option['loss'])

        train_rmse_batch = math.sqrt(train_l2_loss_batch.item() * 3)  # 3 channel, rgb

        # backpropagation and update params
        optimizer.zero_grad()
        train_loss_batch.backward()
        optimizer.step()

        # record running time
        time_lapse = time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))

        # plot train
        if train_option['plot_on']:
            if iters % train_option['train_plot_rate'] == 0 or iters == train_option['max_iters'] - 1:
                vis_train_fig = plotMontage(cam_train_batch, prj_train_pred, prj_train_batch, win=vis_train_fig, title='[Train]' + title)
                appendDataPoint(iters, train_loss_batch.item(), vis_curve_fig, 'train_loss')
                appendDataPoint(iters, train_rmse_batch, vis_curve_fig, 'train_rmse')

        # validation
        valid_psnr, valid_rmse, valid_ssim,valid_diff ,valid_lpips= 0., 0., 0.,0.,0.
        if valid_data is not None and ((iters+1) % (train_option['valid_rate']) == 0 or iters == train_option['max_iters'] - 1):
            valid_psnr, valid_rmse, valid_ssim,valid_diff, valid_lpips,prj_valid_pred,valid_speed = evaluate(net, valid_data)

            # plot validation
            if train_option['plot_on']:
                idx = np.array([9, 10, 11, 14, 70]) - 1  # fix validatio visulization
                vis_valid_fig = plotMontage(valid_data['cam_valid'][idx], prj_valid_pred[idx], valid_data['prj_valid'][idx],
                                            win=vis_valid_fig, title='[Valid]' + title)
                appendDataPoint(iters, valid_rmse, vis_curve_fig, 'valid_rmse')
                appendDataPoint(iters, valid_ssim, vis_curve_fig, 'valid_ssim')

        # print to console
        print('Iter:{:5d} | Time: {} | Train Loss: {:.4f} | Train RMSE: {:.4f} | Valid PSNR: {:7s}  | Valid RMSE: {:6s} | Valid SSIM: {:6s}  '
              '| Valid DIFF: {:6s}  | Learn Rate: {:.5f} |'.format(iters, time_lapse, train_loss_batch.item(), train_rmse_batch,
                                                                   '{:>2.4f}'.format(valid_psnr) if valid_psnr else '',
                                                                   '{:.4f}'.format(valid_rmse) if valid_rmse else '',
                                                                   '{:.4f}'.format(valid_ssim) if valid_ssim else '',
                                                                   '{:.4f}'.format(valid_diff) if valid_diff else '',
                                                                   optimizer.param_groups[0]['lr']))

        lr_scheduler.step()  # update learning rate according to schedule
        
        iters += 1
        if iters % (10*train_option['train_plot_rate']) == 0 :
            checkpoint_tmp_dir = '../../checkpoint_tmp'
            if not os.path.exists(checkpoint_tmp_dir): os.makedirs(checkpoint_tmp_dir)
            checkpoint_tmp_file_name = fullfile(checkpoint_tmp_dir, title + str(iters) + '.pth')
            torch.save(net.state_dict(), checkpoint_tmp_file_name)


    # Done training and save the last epoch model
    checkpoint_dir = '../../checkpoint'
    if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)
    checkpoint_file_name = fullfile(checkpoint_dir, title + '.pth')
    torch.save(net.state_dict(), checkpoint_file_name)
    print('Checkpoint saved to {}\n'.format(checkpoint_file_name))

    return net, valid_psnr, valid_rmse, valid_ssim, valid_diff,valid_speed,time_lapse


# %% local functions

# compute loss between prediction and ground truth
def computeLoss(prj_pred, prj_train, loss_option):
    train_loss = 0

    # l1
    if 'l1' in loss_option:
        l1_loss = l1_fun(prj_pred, prj_train)
        train_loss = train_loss+ l1_loss

    # l2
    l2_loss = l2_fun(prj_pred, prj_train)
    if 'l2' in loss_option:
        train_loss = train_loss+ l2_loss

    # ssim

    if 'ssim' in loss_option:
        ssim_loss = 1 * (1 - ssim_fun(prj_pred, prj_train))
        train_loss = train_loss+ ssim_loss
        


    return train_loss, l2_loss


# append a data point to the curve in win
def appendDataPoint(x, y, win, name, env=None):
    vis.line(
        X=np.array([x]),
        Y=np.array([y]),
        env=env,
        win=win,
        update='append',
        name=name,
        opts=dict(markers=True, markersize=3)
    )


# plot sample predicted images using visdom, more than three rows
def plotMontage(*argv, index=None, win=None, title=None, env=None):
    with torch.no_grad():  # just in case
        # compute montage grid size
        if argv[0].shape[0] > 5:
            grid_w = 5
            idx = random.sample(range(argv[0].shape[0]), grid_w) if index is None else index
        else:
            grid_w = argv[0].shape[0]
            idx = range(grid_w)

        tile_size = (256, 256)
        im_resize = torch.empty((len(argv) * grid_w, argv[0].shape[1]) + tile_size)
        i = 0
        for im in argv:
            if im.shape[2] != tile_size[0] or im.shape[3] != tile_size[1]:
                im_resize[i:i + grid_w] = F.interpolate(im[idx, :, :, :], tile_size)
            else:
                im_resize[i:i + grid_w] = im[idx, :, :, :]
            i += grid_w

        # title
        plot_opts = dict(title=title, caption=title, font=dict(size=18), width=1300, store_history=False)

        im_montage = torchvision.utils.make_grid(im_resize, nrow=grid_w, padding=10, pad_value=1)
        win = vis.image(im_montage, win=win, opts=plot_opts, env=env)

    return win


# plot sample predicted images using visdom, more than three rows
def plotMontage_patch(*argv, index=None, win=None, title=None, env=None):
    with torch.no_grad():  # just in case
        if argv[0].shape[0] > 5:
            grid_w = 5
            idx = random.sample(range(argv[0].shape[0]), grid_w) if index is None else index
        else:
            grid_w = argv[0].shape[0]
            idx = range(grid_w)

        tile_size = (256, 256)
        im_resize = torch.empty((len(argv) * grid_w, argv[0].shape[1]) + tile_size)
        i = 0
        for im in argv:
            if im.shape[2] > tile_size[0] or im.shape[3] > tile_size[1]:
                x = int(0.5* (tile_size[0] - im.shape[2]))
                y = int(0.5* (tile_size[1] - im.shape[3]))
                im_resize[i:i + grid_w] = (im[idx, :, x:(tile_size[0]+x), y:(tile_size[1]+y)])
            else:
                im_resize[i:i + grid_w] = im[idx, :, :, :]
            i += grid_w

        # title
        plot_opts = dict(title=title, caption=title, font=dict(size=18), width=1300, store_history=False)

        im_montage = torchvision.utils.make_grid(im_resize, nrow=grid_w, padding=10, pad_value=1)
        win = vis.image(im_montage, win=win, opts=plot_opts, env=env)

    return win


def predict(net, data):
    if 'cam_surf' in data and data['cam_surf'] is not None:
        prj_pred = net(data['cam'], data['cam_surf'])
    else:
        prj_pred = net(data['cam'])

    if type(prj_pred) == tuple and len(prj_pred) > 1: prj_pred = prj_pred[0]
    return prj_pred


# evaluate model on validation dataset
def evaluate(net, valid_data):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cam_surf = valid_data['cam_surf']
    cam_valid = valid_data['cam_valid']
    prj_valid = valid_data['prj_valid']
    valid_speed = 0.0
    

    with torch.no_grad():
        net.eval()  # explicitly set to eval mode

        # if you have limited GPU memory, we need to predict in batch mode
        if cam_surf.device.type != device.type:
            last_loc = 0
            valid_mse, valid_ssim, valid_diff ,valid_lpips = 0., 0., 0.,0.

            prj_valid_pred = torch.zeros(prj_valid.shape)
            num_valid = cam_valid.shape[0]
            batch_size = 1 if num_valid > 16 else num_valid  

            for i in range(0, num_valid // batch_size):
                idx = range(last_loc, last_loc + batch_size)
                cam_surf_batch = cam_surf[idx, :, :, :].to(device) if cam_surf.device.type != 'cuda' else cam_surf[idx, :, :, :]
                cam_valid_batch = cam_valid[idx, :, :, :].to(device) if cam_valid.device.type != 'cuda' else cam_valid[idx, :, :, :]
                prj_valid_batch = prj_valid[idx, :, :, :].to(device) if prj_valid.device.type != 'cuda' else prj_valid[idx, :, :, :]
                # predict batch
                start_time = time.time()
                prj_valid_pred_batch = predict(net, dict(cam=cam_valid_batch, cam_surf=cam_surf_batch)).detach()
                valid_speed += time.time() - start_time
                if type(prj_valid_pred_batch) == tuple and len(prj_valid_pred_batch) > 1: prj_valid_pred_batch = prj_valid_pred_batch[0]
                prj_valid_pred[last_loc:last_loc + batch_size, :, :, :] = prj_valid_pred_batch.cpu()

                # compute loss
                valid_mse += l2_fun(prj_valid_pred_batch, prj_valid_batch).item() * batch_size
                valid_ssim += ssim(prj_valid_pred_batch, prj_valid_batch) * batch_size

                # differential color
                valid_pred_lab_batch = rgb2lab_diff(prj_valid_pred_batch,device)            
                valid_gt_lab_batch = rgb2lab_diff(prj_valid_batch,device)
                diff_map = ciede2000_diff(valid_pred_lab_batch,valid_gt_lab_batch,device).unsqueeze(1)
                color_loss=diff_map.mean()
                valid_diff += color_loss*prj_valid_pred_batch.shape[0]
                last_loc += batch_size

            # average
            valid_lpips /= num_valid
            valid_mse /= num_valid
            valid_ssim /= num_valid
            valid_diff /= num_valid

            valid_rmse = math.sqrt(valid_mse * 3)  # 3 channel image
            valid_psnr = 10 * math.log10(1 / valid_mse)
        else:
            # if all data can be loaded to GPU memory
            valid_speed = 0.0
            start_time = time.time()
            prj_valid_pred = predict(net, dict(cam=cam_valid, cam_surf=cam_surf)).detach()
            valid_speed += time.time() - start_time
            valid_mse = l2_fun(prj_valid_pred, prj_valid).item()
            valid_rmse = math.sqrt(valid_mse * 3)  # 3 channel image
            valid_psnr = 10 * math.log10(1 / valid_mse)
            valid_ssim = ssim_fun(prj_valid_pred, prj_valid).item()
            valid_diff = 0.0
            valid_lpips = 0.0

    return valid_psnr, valid_rmse, valid_ssim, valid_diff, valid_lpips, prj_valid_pred,valid_speed
