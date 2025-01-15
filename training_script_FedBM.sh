
nvidia-smi

python main.py --exp_name='E1' \
--data_path='/home/meiluzhu2/data/Retinal_OCT-C8/' \
--node_num=12 \
--iid=0 \
--dirichlet_alpha=0.05 \
--local_model='ResNet18' \
--dataset='OCT' \
--T=200 \
--E=2 \
--select_ratio=0.5 \
--temperature=17.5 \
--method='FedBM' \
--batchsize=8 \
--lr=0.01 \
--num_classes=8 \
--g_iter=100 \
--lr_g=3e-4 \
--freq_g_driver=5 \
--synthesis_batch_size=64 \
--g_batchsize=16 \
--lambda_div=1 \
--lambda_dis=0.1 \


python main.py --exp_name='E1' \
--data_path='./data/Retinal_OCT-C8/' \
--node_num=12 \
--iid=0 \
--dirichlet_alpha=0.1 \
--local_model='ResNet18' \
--dataset='OCT' \
--T=200 \
--E=2 \
--select_ratio=0.5 \
--temperature=12.5 \
--method='FedBM' \
--batchsize=8 \
--lr=0.01 \
--num_classes=8 \
--g_iter=100 \
--lr_g=3e-4 \
--freq_g_driver=5 \
--synthesis_batch_size=64 \
--g_batchsize=16 \
--lambda_div=1 \
--lambda_dis=0.1 \



python main.py --exp_name='E1' \
--data_path='./data/kvasir-dataset-v2-processed' \
--node_num=10 \
--iid=0 \
--dirichlet_alpha=0.05 \
--local_model='ResNet18' \
--dataset='Kvasir' \
--T=200 \
--E=2 \
--select_ratio=0.5 \
--temperature=12.5 \
--method='FedBM' \
--batchsize=8 \
--lr=0.01 \
--num_classes=8 \
--g_iter=100 \
--lr_g=3e-4 \
--freq_g_driver=5 \
--synthesis_batch_size=64 \
--g_batchsize=32 \
--lambda_div=1 \
--lambda_dis=1 \

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
--temperature=7.5 \
--method='FedBM' \
--batchsize=8 \
--lr=0.01 \
--num_classes=8 \
--g_iter=100 \
--lr_g=3e-4 \
--freq_g_driver=5 \
--synthesis_batch_size=64 \
--g_batchsize=32 \
--lambda_div=1 \
--lambda_dis=0.1 \














