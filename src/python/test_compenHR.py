

# %% Set environment
import os

# set device
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device_ids = [0]

from trainNetwork import *
import Models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() >= 1:
    print('Train with', len(device_ids), 'GPUs!')
else:
    print('Train with CPU!')

# %% K=20 setups
dataset_root = fullfile(os.getcwd(), '../../data/1024')

data_list = [
    'lemon/2',
]


# Training configurations of CompenNet++ reported in the paper
num_train_list = [500]
loss_list = ['l1+l2+ssim']

# You can also compare different configurations, such as different number of training images and loss functions as shown below

model_list = ['CompenHR' ] #

# default training options
train_option_default = {'max_iters': 1,
                        'batch_size': 4,
                        'lr': 1e-3,  # learning rate
                        'lr_drop_ratio': 0.2,
                        'lr_drop_rate': 1500,  # adjust this according to max_iters (lr_drop_rate < max_iters)
                        'loss': '',  # loss will be set to one of the loss functions in loss_list later
                        'l2_reg': 1e-4,  # l2 regularization
                        'device': device,
                        'pre-trained': False,
                        'pretrain_csr': '',
                        'pretrain_cmp': '', 
                        'plot_on': False,  # plot training progress using visdom (disable for faster training)
                        'train_plot_rate': 100,  # training and visdom plot rate (increase for faster training)
                        'valid_rate': 100}  # validation and visdom plot rate (increase for faster training)

# a flag that decides whether to compute and save the compensated images to the drive
save_compensation = True

# log file
from time import localtime, strftime

log_dir = '../../log'
if not os.path.exists(log_dir): os.makedirs(log_dir)
log_file_name = strftime('%Y-%m-%d_%H_%M_%S', localtime()) + '.txt'
log_file = open(fullfile(log_dir, log_file_name), 'w')
title_str = '{:30s}{:<30}{:<20}{:<15}{:<15}{:<15}{:<15}{:<15}{:<15}{:<15}{:<15}{:<15}{:<15}{:<15}{:<15}\n'
log_file.write(title_str.format('data_name', 'model_name', 'loss_function',
                                'num_train', 'batch_size', 'max_iters',
                                'uncmp_psnr', 'uncmp_rmse', 'uncmp_ssim','uncmp_diff','uncmp_lpips',
                                'valid_psnr', 'valid_rmse', 'valid_ssim', 'valid_diff','valid_lpips','valid_speed'))
log_file.close()


# resize the input images if input_size is not None
input_size = (1024, 1024) 
input_lr_size = (1024,1024)
upscale_factor = 4
resetRNGseed(0)


# stats for different setups
for data_name in data_list:
    torch.cuda.empty_cache()

    # load training and validation data
    data_root = fullfile(dataset_root, data_name)
    cam_surf,cam_valid, prj_valid, mask_corners = loadTestData(dataset_root, data_name, input_size, CompenNeSt_only=False)

    # surface image for training and validation
    cam_surf_valid = cam_surf.expand_as(cam_valid)


    # validation data, 200 image pairs
    valid_data = dict(cam_surf=cam_surf_valid, cam_valid=cam_valid, prj_valid=prj_valid)

    # stats for different #Train
    for num_train in num_train_list:
        train_option = train_option_default.copy()
        train_option['num_train'] = num_train
        
        # stats for different models
        for model_name in model_list:

            train_option['model_name'] = model_name.replace('/', '_')
            
            # stats for different loss functions
            for loss in loss_list:
                log_file = open(fullfile(log_dir, log_file_name), 'a')
                
                # set seed of rng for repeatability
                resetRNGseed(0)

                # create a GANet
                ga_net = Models.GANet(out_size=input_lr_size)

                if torch.cuda.device_count() >= 1: ga_net = nn.DataParallel(ga_net, device_ids=device_ids).to(device)


                if model_name == 'CompenHR': 
                    pa_net = Models.PANet()
                    if torch.cuda.device_count() >= 1: pa_net = nn.DataParallel(pa_net, device_ids=device_ids).to(device)
                    compen_hd = Models.CompenNeStPlusplusDUShuffleSA(ga_net, pa_net)
                    if torch.cuda.device_count() >= 1: compen_hd = nn.DataParallel(compen_hd, device_ids=device_ids).to(device)
                    if train_option['pretrain_csr'] !='':
                        compen_hd.load_state_dict(torch.load(train_option['pretrain_csr'])) 
                


                # train option for current configuration, i.e., data name and loss function
                train_option['data_name'] = data_name.replace('/', '_')
                train_option['loss'] = loss

                print('-------------------------------------- Training Options -----------------------------------')
                print('\n'.join('{}: {}'.format(k, v) for k, v in train_option.items()))
                print('------------------------------------ Start training {:s} ---------------------------'.format(model_name))

                
                # train model
                valid_psnr, valid_rmse, valid_ssim ,valid_diff,valid_lpips, prj_valid_pred,valid_speed = evaluate(compen_hd, valid_data)
                print(valid_lpips)
                print('| cmp Valid PSNR: {:7s}  | Valid RMSE: {:6s} | Valid SSIM: {:6s} | Valid DIFF: {:6s}  | Valid LPIPS: {:6s}  | Valid Speed: {:6s}  |'.format('{:>2.4f}'.format(valid_psnr) if valid_psnr else '',
                                                                   '{:.4f}'.format(valid_rmse) if valid_rmse else '',
                                                                   '{:.4f}'.format(valid_ssim) if valid_ssim else '',
                                                                   '{:.4f}'.format(valid_diff) if valid_diff else '',
                                                                   '{:.4f}'.format(valid_lpips) if valid_lpips else '',
                                                                   '{:.4f}'.format(valid_speed) if valid_speed else ''))
                # uncompensated metrics
                uncmp_psnr = 0.0
                uncmp_rmse = 0.0
                uncmp_ssim = 0.0
                uncmp_diff = 0.0
                uncmp_lpips = 0.0
                
                print('| unc Valid PSNR: {:7s}  | Valid RMSE: {:6s} |Valid SSIM: {:6s} '
              '| Valid DIFF: {:6s}  | Valid LPIPS: {:6s} |'.format('{:>2.4f}'.format(uncmp_psnr) if uncmp_psnr else '',
                                                                   '{:.4f}'.format(uncmp_rmse) if uncmp_rmse else '',
                                                                   '{:.4f}'.format(uncmp_ssim) if uncmp_ssim else '',
                                                                   '{:.4f}'.format(uncmp_diff) if uncmp_diff else '',
                                                                   '{:.4f}'.format(uncmp_lpips) if uncmp_lpips else ''))
                # uncompensated metrics

                
                valid_speed = 0.0

                # [testing phase] create compensated testing images
                if save_compensation:
                    print('------------------------------------ Start testing {:s} ---------------------------'.format(model_name))
                    torch.cuda.empty_cache()
                    
                    # desired test images are created such that they can fill the optimal displayable area (see paper for detail)
                    desire_test_path = fullfile(data_root, 'cam/desire/test')
                    assert os.path.isdir(desire_test_path), 'images and folder {:s} does not exist!'.format(desire_test_path)
                    desire_test = readImgsMT(fullfile(data_root, 'cam/desire/test'))
                    uncmp_psnr, uncmp_rmse, uncmp_ssim,uncmp_diff,uncmp_lpips = computeMetrics(cam_valid, desire_test)

                    # compensate and save images
                    cam_surf_test = cam_surf.expand_as(desire_test)
                    num_valid = cam_surf_test.shape[0]
                    batch_size = 1
                    last_cal = 0
                    prj_cmp_test = torch.zeros(prj_valid.shape)
                    
                    
                    # create image save path
                    cmp_folder_name = '{}_{}_{}_{}_{}'.format(train_option['model_name'], loss, num_train, train_option['batch_size'],
                                                              train_option['max_iters'])
                                                              
                    for i in range(0,num_valid//batch_size):
                        
                        cam_surf_batch = cam_surf_test[last_cal:last_cal+batch_size, :, :, :].to(device)
                        desire_test_batch = desire_test[last_cal:last_cal+batch_size, :, :, :].to(device)
                        
                        with torch.no_grad():

                            compen_hd.eval()
                            prj_cmp_test_batch = compen_hd(desire_test_batch, cam_surf_batch).detach()  # compensated prj input image x^{*}
                            prj_cmp_test[last_cal:last_cal+batch_size, :, :, :] = prj_cmp_test_batch.cpu()
                            
                            last_cal = last_cal + batch_size
                            
                        del desire_test_batch, cam_surf_batch
                    del desire_test, cam_surf_test

                    
                    # save images
                    
                    prj_cmp_path = fullfile(data_root, 'prj/cmp', cmp_folder_name)
                    if not os.path.exists(prj_cmp_path): os.makedirs(prj_cmp_path)

                    saveImgs(prj_cmp_test, prj_cmp_path) 
                    
                    
                    print('Compensation images saved to ' + prj_cmp_path)
    

                # save results to log file
                ret_str = '{:30s}{:<30}{:<20}{:<15}{:<15}{:<15}{:<15.4f}{:<15.4f}{:<15.4f}{:<15.4f}{:<15.4f}{:<15.4f}{:<15.4f}{:<15.4f}{:<15.4f}{:<15.4f}{:<15.4f}\n'
                log_file.write(ret_str.format(data_name, model_name, loss, num_train, train_option['batch_size'], train_option['max_iters'],uncmp_psnr, uncmp_rmse, uncmp_ssim,uncmp_diff,uncmp_lpips,valid_psnr, valid_rmse ,valid_ssim,valid_diff,valid_lpips, valid_speed))
                log_file.close() 


                del compen_hd, ga_net
                torch.cuda.empty_cache()
                print('-------------------------------------- Done! ---------------------------\n')

    del cam_surf, cam_valid, prj_valid, mask_corners,valid_data,cam_surf_valid


print('All dataset done!')
