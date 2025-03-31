import argparse


vqvae_bodypart_cfg = {
    'default': dict(
        parts_code_nb={  # number of codes
            'Root': 512,
            'R_Leg': 512,
            'L_Leg': 512,
            'Backbone': 512,
            'R_Arm': 512,
            'L_Arm': 512,
        },
        parts_code_dim={  # Remember code_dim should be same to output_dim
            'Root': 64,
            'R_Leg': 128,
            'L_Leg': 128,
            'Backbone': 128,
            'R_Arm': 128,
            'L_Arm': 128,
        },
        parts_output_dim={  # dimension of encoder's output
            'Root': 64,
            'R_Leg': 128,
            'L_Leg': 128,
            'Backbone': 128,
            'R_Arm': 128,
            'L_Arm': 128,
        },
        parts_hidden_dim={  # hidden dimension of conv1d in encoder/decoder
            'Root': 64,
            'R_Leg': 128,
            'L_Leg': 128,
            'Backbone': 128,
            'R_Arm': 128,
            'L_Arm': 128,
        }

    ),

}

vqvae_bodypart_cfg_plus = {
    'default': dict(
        parts_code_nb={  # number of codes
            'Root': 1024,
            'R_Leg': 1024,
            'L_Leg': 1024,
            'Backbone': 1024,
            'R_Arm': 1024,
            'L_Arm': 1024,
        },
        parts_code_dim={  # Remember code_dim should be same to output_dim
            'Root': 64,
            'R_Leg': 256,
            'L_Leg': 256,
            'Backbone': 256,
            'R_Arm': 256,
            'L_Arm': 256,
        },
        parts_output_dim={  # dimension of encoder's output
            'Root': 64,
            'R_Leg': 256,
            'L_Leg': 256,
            'Backbone': 256,
            'R_Arm': 256,
            'L_Arm': 256,
        },
        parts_hidden_dim={  # hidden dimension of conv1d in encoder/decoder
            'Root': 64,
            'R_Leg': 256,
            'L_Leg': 256,
            'Backbone': 256,
            'R_Arm': 256,
            'L_Arm': 256,
        }

    ),

}

def get_args_parser(args=None):
    parser = argparse.ArgumentParser(description='Optimal Transport AutoEncoder training for AIST',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ## dataloader  
    parser.add_argument('--dataname', type=str, default='kit', help='dataset directory')
    parser.add_argument('--batch-size', default=128, type=int, help='batch size')
    parser.add_argument('--window-size', type=int, default=64, help='training motion length')
    parser.add_argument('--ddp', action='store_true', help='whether use ddp')
    ## optimization
    parser.add_argument('--total-iter', default=200000, type=int, help='number of total iterations to run')
    parser.add_argument('--warm-up-iter', default=1000, type=int, help='number of total iterations for warmup')
    parser.add_argument('--lr', default=2e-4, type=float, help='max learning rate')
    parser.add_argument('--lr-scheduler', default=[50000, 400000], nargs="+", type=int, help="learning rate schedule (iterations)")
    parser.add_argument('--sem-iter', default=100000, type=int, help='number of iterations for semantic loss')
    parser.add_argument('--gamma', default=0.05, type=float, help="learning rate decay")
    parser.add_argument('--clip_grad', default=0.99, type=float, help='clip grad norm')

    parser.add_argument('--weight-decay', default=0.0, type=float, help='weight decay')
    parser.add_argument("--commit", type=float, default=0.02, help="hyper-parameter for the commitment loss")
    parser.add_argument('--loss-vel', type=float, default=0.1, help='hyper-parameter for the velocity loss')
    parser.add_argument('--recons-loss', type=str, default='l2', help='reconstruction loss')
    parser.add_argument("--contrastive", type=float, default=0.5, help="hyper-parameter for the commitment loss")
    parser.add_argument("--Disentangle", type=float, default=0.5, help="hyper-parameter for the commitment loss")

    parser.add_argument("--vqvae-cfg", type=str, help="Base config for vqvae")

    parser.add_argument("--mu", type=float, default=0.99, help="exponential moving average to update the codebook")

    # It is the number of downsampling block in the net, not the downsampling rate referred in the paper.
    parser.add_argument("--down-t", type=int, default=2, help="downsampling rate")
    parser.add_argument("--stride-t", type=int, default=2, help="stride size")
    parser.add_argument("--depth", type=int, default=3, help="depth of the network")
    parser.add_argument("--dilation-growth-rate", type=int, default=3, help="dilation growth rate")
    parser.add_argument('--vq-act', type=str, default='relu', choices = ['relu', 'silu', 'gelu'], help='dataset directory')
    parser.add_argument('--vq-norm', type=str, default=None, help='dataset directory')
    parser.add_argument('--vqdec-norm', type=str, default='GN', help='dataset directory')
    parser.add_argument('--num-heads', type=int, default=4)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--numdec-layers', type=int, default=2)
    parser.add_argument('--bodyconfig', type=bool, default=False)
    parser.add_argument('--causal', type=bool, default=False, help='causal squence')
    parser.add_argument('--position', type=int, default=0, help='0:without pos 1:learnable 2:cos sin')
    parser.add_argument('--d_model', type=int, default=256, help='d_model')
    parser.add_argument('--with_attn', type=bool, default=False, help='with_attn')
    parser.add_argument('--with_global', type=int, default=1, help='with_global')
    parser.add_argument('--text_dim', type=int, default=512, help='text_dim')
    ## quantizer
    parser.add_argument("--quantizer", type=str, default='ema_reset', choices = ['ema', 'orig', 'ema_reset', 'reset'], help="eps for optimal transport")
    parser.add_argument('--beta', type=float, default=1.0, help='commitment loss in standard VQ')
    parser.add_argument('--num_quantizers_global', type=int, default=6, help='num_quantizers')
    parser.add_argument('--num_quantizers', type=int, default=3, help='num_quantizers')
    parser.add_argument('--shared_codebook', action="store_true")
    parser.add_argument('--quantize_dropout_prob', type=float, default=0.2, help='quantize_dropout_prob')
    parser.add_argument('--decoder_vision', type=int, default=1, help='decoder vision') # 1:cnn 2:transformer
    parser.add_argument('--vision', type=int, default=4, help='arch vision')
    parser.add_argument('--lgvq',  type=int, default=0, help='lgvq version 0: no lgvq 1: lgvq global 2: lgvq global+mask')
    parser.add_argument('--lglayers', type=int, default=2, help='num_layers')

    ## resume
    parser.add_argument("--resume-pth", type=str, default=None, help='resume pth for VQ')
    
    ## output directory 
    parser.add_argument('--out-dir', type=str, default='output/', help='output directory')
    parser.add_argument('--results-dir', type=str, default='visual_results/', help='output directory')
    parser.add_argument('--visual-name', type=str, default='baseline', help='output directory')
    parser.add_argument('--exp-name', type=str, default='exp_debug', help='name of the experiment, will create a file inside out-dir')
    ## other
    parser.add_argument('--print-iter', default=200, type=int, help='print frequency')
    parser.add_argument('--eval-iter', default=1000, type=int, help='evaluation frequency')
    parser.add_argument('--seed', default=123, type=int, help='seed for initializing training.')
    
    parser.add_argument('--vis-gt', action='store_true', help='whether visualize GT motions')
    parser.add_argument('--nb-vis', default=20, type=int, help='nb of visualizations')
    parser.add_argument('--vqvae_sem_nb', default=512, type=int, help='nb of visualizations')
    
    if args is None:
        return parser.parse_args()

    else:
        return parser.parse_args(args=args)


def get_vavae_test_args_parser():
    parser = argparse.ArgumentParser(description='Evaluate the body VQVAE_bodypart',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--vqvae-train-dir', type=str, help='VQVAE training directory')
    parser.add_argument('--select-vqvae-ckpt', type=str, help='Select which ckpt for use: [last, fid, div, top1, matching]')


    return parser.parse_args()



