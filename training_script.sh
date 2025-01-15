
nvidia-smi

python main.py --exp_name='E1' \
--data_path='./data/Retinal_OCT-C8/' \
--node_num=12 \
--iid=0 \
--dirichlet_alpha=0.05 \
--local_model='ResNet18' \
--dataset='Kvasir' \
--T=200 \
--E=2 \
--select_ratio=0.5 \
--method='FedAvg' \
--batchsize=8 \
--lr=0.01 \
--num_classes=8 \


python main.py --exp_name='E1' \
--data_path='./data/Retinal_OCT-C8/' \
--node_num=12 \
--iid=0 \
--dirichlet_alpha=0.1 \
--local_model='ResNet18' \
--dataset='Kvasir' \
--T=200 \
--E=2 \
--select_ratio=0.5 \
--method='FedAvg' \
--batchsize=8 \
--lr=0.01 \
--num_classes=8 \



python main.py --exp_name='E1' \
--data_path='./data/kvasir-dataset-v2-processed/' \
--node_num=10 \
--iid=0 \
--dirichlet_alpha=0.05 \
--local_model='ResNet18' \
--dataset='Kvasir' \
--T=200 \
--E=2 \
--select_ratio=0.5 \
--method='FedAvg' \
--batchsize=8 \
--lr=0.01 \
--num_classes=8 \

python main.py --exp_name='E1' \
--data_path='./data/kvasir-dataset-v2-processed/' \
--node_num=10 \
--iid=0 \
--dirichlet_alpha=0.1 \
--local_model='ResNet18' \
--dataset='Kvasir' \
--T=200 \
--E=2 \
--select_ratio=0.5 \
--method='FedAvg' \
--batchsize=8 \
--lr=0.01 \
--num_classes=8 \









