#CUDA_VISIBLE_DEVICES=0 python sparsity_stats.py --arch ResNet56 --data_path ../../datasets/data.cifar10 --sr 0.0001  --save checkpoint/exp04_res56_sr1e4_thr0.1 --threshold 0.1

#CUDA_VISIBLE_DEVICES=1 python sparsity_stats.py --arch ResNet56 --data_path ../../datasets/data.cifar10 --sr 0.0001 --threshold 0.05 --save checkpoint/exp03_res56_sr1e4_thr0.05 --batch-size 64 --epochs 160 

