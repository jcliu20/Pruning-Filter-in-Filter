#CUDA_VISIBLE_DEVICES=1 python main.py --arch VGG --data_path ../../datasets/data.cifar10 --sr 0.00001 --threshold 0.01 --save checkpoint/exp02_vgg_sr1e5_thr0.01 --batch-size 640 --epochs 160 

#CUDA_VISIBLE_DEVICES=0 python main.py --arch ResNet56 --data_path ../../datasets/data.cifar10 --sr 0.00001 --threshold 0.01 --save checkpoint/exp01_res56_sr1e5_thr0.01 --batch-size 640 --epochs 160 

#CUDA_VISIBLE_DEVICES=0 python main.py --arch ResNet56 --data_path ../../datasets/data.cifar10 --sr 0.0001 --threshold 0.1 --save checkpoint/exp04_res56_sr1e4_thr0.1 --batch-size 64 --epochs 160 

#CUDA_VISIBLE_DEVICES=1 python main.py --arch ResNet56 --data_path ../../datasets/data.cifar10 --sr 0.0001 --threshold 0.05 --save checkpoint/exp03_res56_sr1e4_thr0.05 --batch-size 64 --epochs 160 

#CUDA_VISIBLE_DEVICES=1 python main.py --arch ResNet56 --data_path ../../datasets/data.cifar10 --sr 0.0001 --threshold 0.05 --save checkpoint/exp00_debug

CUDA_VISIBLE_DEVICES=0 python main.py --arch ResNet56 --data_path ../../datasets/data.cifar10 --sr 0.001 --threshold 0.1 --save checkpoint/exp05_res56_sr1e3_thr0.1 --batch-size 64 --epochs 120 


