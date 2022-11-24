In the following, the first row means the python file in 'examples', other rows are the running scripts

A. Ensemble model in main experiments

1.Ensemble of CDANs and JAN 

cdan_jan_com.py

CUDA_VISIBLE_DEVICES=6 python cdan_jan_com.py data/office31 -d Office31 -s W -t A -a resnet50 --epochs 20 --seed 2 --log logs/office4/W2A 
CUDA_VISIBLE_DEVICES=6 python cdan_jan_com.py data/office31 -d Office31 -s A -t W -a resnet50 --epochs 20 --seed 2 --log logs/office2/A2W 
CUDA_VISIBLE_DEVICES=6 python cdan_jan_com.py data/office31 -d Office31 -s A -t D -a resnet50 --epochs 20 --seed 2 --log logs/office2/A2D 
CUDA_VISIBLE_DEVICES=6 python cdan_jan_com.py data/office31 -d Office31 -s D -t A -a resnet50 --epochs 20 --seed 2 --log logs/office2/D2A 
CUDA_VISIBLE_DEVICES=6 python cdan_jan_com.py data/office31 -d Office31 -s D -t W -a resnet50 --epochs 20 --seed 2 --log logs/office2/D2W 
CUDA_VISIBLE_DEVICES=6 python cdan_jan_com.py data/office31 -d Office31 -s W -t D -a resnet50 --epochs 20 --seed 2 --log logs/office2/W2D 

2. Ensemble of two CDANs

cdan_two_com.py

CUDA_VISIBLE_DEVICES=6 python cdan_two_com.py data/office31 -d Office31 -s W -t A -a resnet50 --epochs 20 --distill_epochs 5 --seed 2 --log logs/office1/W2A 
CUDA_VISIBLE_DEVICES=1 python cdan_two_com.py data/office31 -d Office31 -s A -t W -a resnet50 --epochs 20 --distill_epochs 5 --seed 2 --log logs/office1/A2W 
CUDA_VISIBLE_DEVICES=1 python cdan_two_com.py data/office31 -d Office31 -s A -t D -a resnet50 --epochs 20 --distill_epochs 5 --seed 2 --log logs/office1/A2D 
CUDA_VISIBLE_DEVICES=3 python cdan_two_com.py data/office31 -d Office31 -s D -t A -a resnet50 --epochs 20 --distill_epochs 5 --seed 2 --log logs/office1/D2A 
CUDA_VISIBLE_DEVICES=1 python cdan_two_com.py data/office31 -d Office31 -s D -t W -a resnet50 --epochs 20 --distill_epochs 5 --seed 2 --log logs/office1/D2W 
CUDA_VISIBLE_DEVICES=4 python cdan_two_com.py data/office31 -d Office31 -s W -t D -a resnet50 --epochs 20 --distill_epochs 5 --seed 2 --log logs/office1/W2D 

3. Ensemble of three CDANs

cdan_three_com.py

CUDA_VISIBLE_DEVICES=1 python cdan_three_com.py data/office31 -d Office31 -s W -t A -a resnet50 --epochs 20 --seed 2 --log logs/office_cdan4/W2A 
CUDA_VISIBLE_DEVICES=2 python cdan_three_com.py data/office31 -d Office31 -s A -t W -a resnet50 --epochs 20 --seed 2 --log logs/office_cdan4/A2W 
CUDA_VISIBLE_DEVICES=3 python cdan_three_com.py data/office31 -d Office31 -s A -t D -a resnet50 --epochs 20 --seed 2 --log logs/office_cdan4/A2D 
CUDA_VISIBLE_DEVICES=6 python cdan_three_com.py data/office31 -d Office31 -s D -t A -a resnet50 --epochs 20 --seed 2 --log logs/office_cdan4/D2A 
CUDA_VISIBLE_DEVICES=1 python cdan_three_com.py data/office31 -d Office31 -s D -t W -a resnet50 --epochs 20 --seed 2 --log logs/office_cdan4/D2W 
CUDA_VISIBLE_DEVICES=2 python cdan_three_com.py data/office31 -d Office31 -s W -t D -a resnet50 --epochs 20 --seed 2 --log logs/office_cdan4/W2D 

4. Ensemble of two CDAN+MCC+SDAT

cdan_mcc_sdat_com.py

python cdan_mcc_sdat_com.py data/office31 -d Office31 -s W -t A -a vit_base_patch16_224 --epochs 20 --seed 0 --seed2 1 -b 24 --no-pool --log logs/cdan_mcc_sdat_vit_5/Office31_W2A --log_name W2A_cdan_mcc_sdat_vit --gpu 2 --rho 0.02 --lr 0.002 
python cdan_mcc_sdat_com.py data/office31 -d Office31 -s A -t W -a vit_base_patch16_224 --epochs 20 --seed 0 --seed2 1 -b 24 --no-pool --log logs/cdan_mcc_sdat_vit_5/Office31_A2W --log_name A2W_cdan_mcc_sdat_vit --gpu 0 --rho 0.02 --lr 0.002 
python cdan_mcc_sdat_com.py data/office31 -d Office31 -s A -t D -a vit_base_patch16_224 --epochs 20 --seed 0 --seed2 1 -b 24 --no-pool --log logs/cdan_mcc_sdat_vit_5/Office31_A2D --log_name A2D_cdan_mcc_sdat_vit --gpu 3 --rho 0.02 --lr 0.002 
python cdan_mcc_sdat_com.py data/office31 -d Office31 -s D -t A -a vit_base_patch16_224 --epochs 20 --seed 0 --seed2 1 -b 24 --no-pool --log logs/cdan_mcc_sdat_vit_5/Office31_D2A --log_name D2A_cdan_mcc_sdat_vit --gpu 4 --rho 0.02 --lr 0.002 
python cdan_mcc_sdat_com.py data/office31 -d Office31 -s D -t W -a vit_base_patch16_224 --epochs 20 --seed 0 --seed2 1 -b 24 --no-pool --log logs/cdan_mcc_sdat_vit_5/Office31_D2W --log_name D2W_cdan_mcc_sdat_vit --gpu 6 --rho 0.02 --lr 0.002 
python cdan_mcc_sdat_com.py data/office31 -d Office31 -s W -t D -a vit_base_patch16_224 --epochs 20 --seed 0 --seed2 1 -b 24 --no-pool --log logs/cdan_mcc_sdat_vit_5/Office31_W2D --log_name W2D_cdan_mcc_sdat_vit --gpu 0 --rho 0.02 --lr 0.002 

B. Ablation emperiments

1. Non-instance aware

cdan_jan_com_fusion_learn.py

CUDA_VISIBLE_DEVICES=2 python cdan_jan_com_fusion_learn.py data/office31 -d Office31 -s W -t A -a resnet50 --epochs 20 --seed 2 --log logs/office10/W2A 
CUDA_VISIBLE_DEVICES=6 python cdan_jan_com_fusion_learn.py data/office31 -d Office31 -s A -t W -a resnet50 --epochs 20 --seed 2 --log logs/office10/A2W 
CUDA_VISIBLE_DEVICES=0 python cdan_jan_com_fusion_learn.py data/office31 -d Office31 -s A -t D -a resnet50 --epochs 20 --seed 2 --log logs/office10/A2D 
CUDA_VISIBLE_DEVICES=1 python cdan_jan_com_fusion_learn.py data/office31 -d Office31 -s D -t A -a resnet50 --epochs 20 --seed 2 --log logs/office10/D2A 
CUDA_VISIBLE_DEVICES=4 python cdan_jan_com_fusion_learn.py data/office31 -d Office31 -s D -t W -a resnet50 --epochs 10 --seed 2 --log logs/office10/D2W 
CUDA_VISIBLE_DEVICES=4 python cdan_jan_com_fusion_learn.py data/office31 -d Office31 -s W -t D -a resnet50 --epochs 20 --seed 2 --log logs/office10/W2D

2. Averaging ensemble model

cdan_jan_average.py

CUDA_VISIBLE_DEVICES=3 python cdan_jan_average.py data/office31 -d Office31 -s W -t A --arch resnet50 --epochs 10 --seed 2 --log logs/office6/W2A
CUDA_VISIBLE_DEVICES=3 python cdan_jan_average.py data/office31 -d Office31 -s A -t W --arch resnet50 --epochs 10 --seed 2 --log logs/office6/A2W
CUDA_VISIBLE_DEVICES=0 python cdan_jan_average.py data/office31 -d Office31 -s A -t D --arch resnet50 --epochs 10 --seed 2 --log logs/office6/A2D
CUDA_VISIBLE_DEVICES=1 python cdan_jan_average.py data/office31 -d Office31 -s D -t A --arch resnet50 --epochs 10 --seed 2 --log logs/office6/D2A
CUDA_VISIBLE_DEVICES=4 python cdan_jan_average.py data/office31 -d Office31 -s W -t D --arch resnet50 --epochs 10 --seed 2 --log logs/office6/W2D
CUDA_VISIBLE_DEVICES=6 python cdan_jan_average.py data/office31 -d Office31 -s D -t W --arch resnet50 --epochs 10 --seed 2 --log logs/office6/D2W

3. Use fully connected linear layer other that shuffle linear layer

cdan_jan_linear_layer.py

CUDA_VISIBLE_DEVICES=0 python cdan_jan_linear_layer.py data/office31 -d Office31 -s W -t A -a resnet50 --epochs 15 --seed 2 --log logs/office3/W2A 
CUDA_VISIBLE_DEVICES=1 python cdan_jan_linear_layer.py data/office31 -d Office31 -s A -t W -a resnet50 --epochs 15 --seed 2 --log logs/office3/A2W 
CUDA_VISIBLE_DEVICES=3 python cdan_jan_linear_layer.py data/office31 -d Office31 -s A -t D -a resnet50 --epochs 15 --seed 2 --log logs/office3/A2D 
CUDA_VISIBLE_DEVICES=4 python cdan_jan_linear_layer.py data/office31 -d Office31 -s D -t A -a resnet50 --epochs 15 --seed 2 --log logs/office3/D2A 
CUDA_VISIBLE_DEVICES=5 python cdan_jan_linear_layer.py data/office31 -d Office31 -s D -t W -a resnet50 --epochs 15 --seed 2 --log logs/office3/D2W  
CUDA_VISIBLE_DEVICES=6 python cdan_jan_linear_layer.py data/office31 -d Office31 -s W -t D -a resnet50 --epochs 15 --seed 2 --log logs/office3/W2D 

4. Group_num = 128 in shuffle linear layer

cdan_jan_com.py

CUDA_VISIBLE_DEVICES=6 python cdan_jan_com.py data/office31 -d Office31 -s W -t A -a resnet50 --epochs 20 --seed 2 --log logs/office4/W2A 
CUDA_VISIBLE_DEVICES=6 python cdan_jan_com.py data/office31 -d Office31 -s A -t W -a resnet50 --epochs 20 --seed 2 --log logs/office2/A2W 
CUDA_VISIBLE_DEVICES=6 python cdan_jan_com.py data/office31 -d Office31 -s A -t D -a resnet50 --epochs 20 --seed 2 --log logs/office2/A2D 
CUDA_VISIBLE_DEVICES=6 python cdan_jan_com.py data/office31 -d Office31 -s D -t A -a resnet50 --epochs 20 --seed 2 --log logs/office2/D2A 
CUDA_VISIBLE_DEVICES=6 python cdan_jan_com.py data/office31 -d Office31 -s D -t W -a resnet50 --epochs 20 --seed 2 --log logs/office2/D2W 
CUDA_VISIBLE_DEVICES=6 python cdan_jan_com.py data/office31 -d Office31 -s W -t D -a resnet50 --epochs 20 --seed 2 --log logs/office2/W2D 

5. Group_num = 32 in shuffle linear layer

cdan_jan_com_share_group.py

CUDA_VISIBLE_DEVICES=7 python cdan_jan_com_share_group.py data/office31 -d Office31 -s W -t A -a resnet50 --epochs 20 --seed 2 --log logs/office6/W2A --group_num 32
CUDA_VISIBLE_DEVICES=0 python cdan_jan_com_share_group.py data/office31 -d Office31 -s A -t W -a resnet50 --epochs 20 --seed 2 --log logs/office6/A2W --group_num 32
CUDA_VISIBLE_DEVICES=1 python cdan_jan_com_share_group.py data/office31 -d Office31 -s A -t D -a resnet50 --epochs 20 --seed 2 --log logs/office6/A2D --group_num 32 
CUDA_VISIBLE_DEVICES=4 python cdan_jan_com_share_group.py data/office31 -d Office31 -s D -t A -a resnet50 --epochs 20 --seed 2 --log logs/office6/D2A --group_num 32
CUDA_VISIBLE_DEVICES=6 python cdan_jan_com_share_group.py data/office31 -d Office31 -s D -t W -a resnet50 --epochs 20 --seed 2 --log logs/office6/D2W --group_num 32 
CUDA_VISIBLE_DEVICES=7 python cdan_jan_com_share_group.py data/office31 -d Office31 -s W -t D -a resnet50 --epochs 20 --seed 2 --log logs/office6/W2D --group_num 32

6. Group_num = 16 in shuffle linear layer

cdan_jan_com_share_group.py

CUDA_VISIBLE_DEVICES=1 python cdan_jan_com_share_group.py data/office31 -d Office31 -s W -t A -a resnet50 --epochs 20 --seed 2 --log logs/office5/W2A --group_num 16 
CUDA_VISIBLE_DEVICES=7 python cdan_jan_com_share_group.py data/office31 -d Office31 -s A -t W -a resnet50 --epochs 20 --seed 2 --log logs/office5/A2W --group_num 16 
CUDA_VISIBLE_DEVICES=4 python cdan_jan_com_share_group.py data/office31 -d Office31 -s A -t D -a resnet50 --epochs 20 --seed 2 --log logs/office5/A2D --group_num 16 
CUDA_VISIBLE_DEVICES=6 python cdan_jan_com_share_group.py data/office31 -d Office31 -s D -t A -a resnet50 --epochs 20 --seed 2 --log logs/office5/D2A --group_num 16 
CUDA_VISIBLE_DEVICES=0 python cdan_jan_com_share_group.py data/office31 -d Office31 -s D -t W -a resnet50 --epochs 20 --seed 2 --log logs/office5/D2W --group_num 16 
CUDA_VISIBLE_DEVICES=1 python cdan_jan_com_share_group.py data/office31 -d Office31 -s W -t D -a resnet50 --epochs 20 --seed 2 --log logs/office5/W2D --group_num 16 

7. There is one layer in fusion sub-network

cdan_jan_fusion_1_layer.py

CUDA_VISIBLE_DEVICES=0 python cdan_jan_fusion_1_layer.py data/office31 -d Office31 -s W -t A -a resnet50 --epochs 20 --seed 2 --log logs/office8/W2A 
CUDA_VISIBLE_DEVICES=1 python cdan_jan_fusion_1_layer.py data/office31 -d Office31 -s A -t W -a resnet50 --epochs 20 --seed 2 --log logs/office8/A2W 
CUDA_VISIBLE_DEVICES=2 python cdan_jan_fusion_1_layer.py data/office31 -d Office31 -s A -t D -a resnet50 --epochs 20 --seed 2 --log logs/office8/A2D 
CUDA_VISIBLE_DEVICES=3 python cdan_jan_fusion_1_layer.py data/office31 -d Office31 -s D -t A -a resnet50 --epochs 20 --seed 2 --log logs/office8/D2A
CUDA_VISIBLE_DEVICES=4 python cdan_jan_fusion_1_layer.py data/office31 -d Office31 -s D -t W -a resnet50 --epochs 20 --seed 2 --log logs/office8/D2W  
CUDA_VISIBLE_DEVICES=1 python cdan_jan_fusion_1_layer.py data/office31 -d Office31 -s W -t D -a resnet50 --epochs 20 --seed 2 --log logs/office8/W2D

8.There are two layers in fusion sub-network

cdan_jan_com.py

CUDA_VISIBLE_DEVICES=6 python cdan_jan_com.py data/office31 -d Office31 -s W -t A -a resnet50 --epochs 20 --seed 2 --log logs/office4/W2A 
CUDA_VISIBLE_DEVICES=6 python cdan_jan_com.py data/office31 -d Office31 -s A -t W -a resnet50 --epochs 20 --seed 2 --log logs/office2/A2W 
CUDA_VISIBLE_DEVICES=6 python cdan_jan_com.py data/office31 -d Office31 -s A -t D -a resnet50 --epochs 20 --seed 2 --log logs/office2/A2D 
CUDA_VISIBLE_DEVICES=6 python cdan_jan_com.py data/office31 -d Office31 -s D -t A -a resnet50 --epochs 20 --seed 2 --log logs/office2/D2A 
CUDA_VISIBLE_DEVICES=6 python cdan_jan_com.py data/office31 -d Office31 -s D -t W -a resnet50 --epochs 20 --seed 2 --log logs/office2/D2W 
CUDA_VISIBLE_DEVICES=6 python cdan_jan_com.py data/office31 -d Office31 -s W -t D -a resnet50 --epochs 20 --seed 2 --log logs/office2/W2D 

9. There are three layers in fusion sub-network

cdan_jan_fusion_3_layer.py

CUDA_VISIBLE_DEVICES=6 python cdan_jan_fusion_3_layer.py data/office31 -d Office31 -s W -t A -a resnet50 --epochs 20 --seed 2 --log logs/office9/W2A 
CUDA_VISIBLE_DEVICES=0 python cdan_jan_fusion_3_layer.py data/office31 -d Office31 -s A -t W -a resnet50 --epochs 20 --seed 2 --log logs/office9/A2W 
CUDA_VISIBLE_DEVICES=1 python cdan_jan_fusion_3_layer.py data/office31 -d Office31 -s A -t D -a resnet50 --epochs 20 --seed 2 --log logs/office9/A2D
CUDA_VISIBLE_DEVICES=3 python cdan_jan_fusion_3_layer.py data/office31 -d Office31 -s D -t A -a resnet50 --epochs 20 --seed 2 --log logs/office9/D2A 
CUDA_VISIBLE_DEVICES=4 python cdan_jan_fusion_3_layer.py data/office31 -d Office31 -s D -t W -a resnet50 --epochs 20 --seed 2 --log logs/office9/D2W 
CUDA_VISIBLE_DEVICES=1 python cdan_jan_fusion_3_layer.py data/office31 -d Office31 -s W -t D -a resnet50 --epochs 20 --seed 2 --log logs/office9/W2D 

10. Adjustment factors in loss items, (beta_1, beta_2, beta_3, beta_4) = (1, 1, 1, 1)

cdan_jan_com.py

CUDA_VISIBLE_DEVICES=6 python cdan_jan_com.py data/office31 -d Office31 -s W -t A -a resnet50 --epochs 20 --seed 2 --log logs/office4/W2A 
CUDA_VISIBLE_DEVICES=6 python cdan_jan_com.py data/office31 -d Office31 -s A -t W -a resnet50 --epochs 20 --seed 2 --log logs/office2/A2W 
CUDA_VISIBLE_DEVICES=6 python cdan_jan_com.py data/office31 -d Office31 -s A -t D -a resnet50 --epochs 20 --seed 2 --log logs/office2/A2D 
CUDA_VISIBLE_DEVICES=6 python cdan_jan_com.py data/office31 -d Office31 -s D -t A -a resnet50 --epochs 20 --seed 2 --log logs/office2/D2A 
CUDA_VISIBLE_DEVICES=6 python cdan_jan_com.py data/office31 -d Office31 -s D -t W -a resnet50 --epochs 20 --seed 2 --log logs/office2/D2W 
CUDA_VISIBLE_DEVICES=6 python cdan_jan_com.py data/office31 -d Office31 -s W -t D -a resnet50 --epochs 20 --seed 2 --log logs/office2/W2D 

11. Adjustment factors in loss items, (beta_1, beta_2, beta_3, beta_4) = (0.5, 1, 1, 1)

cdan_jan_com.py

CUDA_VISIBLE_DEVICES=2 python cdan_jan_com.py data/office31 -d Office31 -s W -t A -a resnet50 --trade-off1 0.5 --trade-off2 1 --trade-off3 1 --trade-off4 1 --epochs 15 --seed 2 --log logs/office17/W2A   
CUDA_VISIBLE_DEVICES=3 python cdan_jan_com.py data/office31 -d Office31 -s A -t W -a resnet50 --trade-off1 0.5 --trade-off2 1 --trade-off3 1 --trade-off4 1 --epochs 15 --seed 2 --log logs/office17/A2W   
CUDA_VISIBLE_DEVICES=5 python cdan_jan_com.py data/office31 -d Office31 -s A -t D -a resnet50 --trade-off1 0.5 --trade-off2 1 --trade-off3 1 --trade-off4 1 --epochs 15 --seed 2 --log logs/office17/A2D    
CUDA_VISIBLE_DEVICES=0 python cdan_jan_com.py data/office31 -d Office31 -s D -t A -a resnet50 --trade-off1 0.5 --trade-off2 1 --trade-off3 1 --trade-off4 1 --epochs 15 --seed 2 --log logs/office17/D2A 
CUDA_VISIBLE_DEVICES=1 python cdan_jan_com.py data/office31 -d Office31 -s D -t W -a resnet50 --trade-off1 0.5 --trade-off2 1 --trade-off3 1 --trade-off4 1 --epochs 15 --seed 2 --log logs/office17/D2W 
CUDA_VISIBLE_DEVICES=6 python cdan_jan_com.py data/office31 -d Office31 -s W -t D -a resnet50 --trade-off1 0.5 --trade-off2 1 --trade-off3 1 --trade-off4 1 --epochs 15 --seed 2 --log logs/office17/W2D 

12. Adjustment factors in loss items, (beta_1, beta_2, beta_3, beta_4) = (1, 0.5, 1, 1)

cdan_jan_com.py

CUDA_VISIBLE_DEVICES=6 python cdan_jan_com.py data/office31 -d Office31 -s W -t A -a resnet50 --trade-off1 1 --trade-off2 0.5 --trade-off3 1 --trade-off4 1 --epochs 15 --seed 2 --log logs/office18/W2A 
CUDA_VISIBLE_DEVICES=4 python cdan_jan_com.py data/office31 -d Office31 -s A -t W -a resnet50 --trade-off1 1 --trade-off2 0.5 --trade-off3 1 --trade-off4 1 --epochs 15 --seed 2 --log logs/office18/A2W 
CUDA_VISIBLE_DEVICES=0 python cdan_jan_com.py data/office31 -d Office31 -s A -t D -a resnet50 --trade-off1 1 --trade-off2 0.5 --trade-off3 1 --trade-off4 1 --epochs 15 --seed 2 --log logs/office18/A2D 
CUDA_VISIBLE_DEVICES=1 python cdan_jan_com.py data/office31 -d Office31 -s D -t A -a resnet50 --trade-off1 1 --trade-off2 0.5 --trade-off3 1 --trade-off4 1 --epochs 15 --seed 2 --log logs/office18/D2A 
CUDA_VISIBLE_DEVICES=2 python cdan_jan_com.py data/office31 -d Office31 -s D -t W -a resnet50 --trade-off1 1 --trade-off2 0.5 --trade-off3 1 --trade-off4 1 --epochs 15 --seed 2 --log logs/office18/D2W 
CUDA_VISIBLE_DEVICES=6 python cdan_jan_com.py data/office31 -d Office31 -s W -t D -a resnet50 --trade-off1 1 --trade-off2 0.5 --trade-off3 1 --trade-off4 1 --epochs 15 --seed 2 --log logs/office18/W2D 

13. Adjustment factors in loss items, (beta_1, beta_2, beta_3, beta_4) = (1, 1, 0.5, 1)

cdan_jan_com.py

CUDA_VISIBLE_DEVICES=3 python cdan_jan_com.py data/office31 -d Office31 -s W -t A -a resnet50 --trade-off1 1 --trade-off2 1 --trade-off3 0.5 --trade-off4 1 --epochs 15 --seed 2 --log logs/office19/W2A 
CUDA_VISIBLE_DEVICES=5 python cdan_jan_com.py data/office31 -d Office31 -s A -t W -a resnet50 --trade-off1 1 --trade-off2 1 --trade-off3 0.5 --trade-off4 1 --epochs 15 --seed 2 --log logs/office19/A2W 
CUDA_VISIBLE_DEVICES=6 python cdan_jan_com.py data/office31 -d Office31 -s A -t D -a resnet50 --trade-off1 1 --trade-off2 1 --trade-off3 0.5 --trade-off4 1 --epochs 15 --seed 2 --log logs/office19/A2D 
CUDA_VISIBLE_DEVICES=7 python cdan_jan_com.py data/office31 -d Office31 -s D -t A -a resnet50 --trade-off1 1 --trade-off2 1 --trade-off3 0.5 --trade-off4 1 --epochs 15 --seed 2 --log logs/office19/D2A 
CUDA_VISIBLE_DEVICES=4 python cdan_jan_com.py data/office31 -d Office31 -s D -t W -a resnet50 --trade-off1 1 --trade-off2 1 --trade-off3 0.5 --trade-off4 1 --epochs 15 --seed 2 --log logs/office19/D2W 
CUDA_VISIBLE_DEVICES=6 python cdan_jan_com.py data/office31 -d Office31 -s W -t D -a resnet50 --trade-off1 1 --trade-off2 1 --trade-off3 0.5 --trade-off4 1 --epochs 15 --seed 2 --log logs/office19/W2D 

14. Adjustment factors in loss items, (beta_1, beta_2, beta_3, beta_4) = (1, 1, 1, 0.5)

cdan_jan_com.py

CUDA_VISIBLE_DEVICES=3 python cdan_jan_com.py data/office31 -d Office31 -s W -t A -a resnet50 --trade-off1 1 --trade-off2 1 --trade-off3 1 --trade-off4 0.5 --epochs 15 --seed 2 --log logs/office20/W2A 
CUDA_VISIBLE_DEVICES=5 python cdan_jan_com.py data/office31 -d Office31 -s A -t W -a resnet50 --trade-off1 1 --trade-off2 1 --trade-off3 1 --trade-off4 0.5 --epochs 15 --seed 2 --log logs/office20/A2W 
CUDA_VISIBLE_DEVICES=0 python cdan_jan_com.py data/office31 -d Office31 -s A -t D -a resnet50 --trade-off1 1 --trade-off2 1 --trade-off3 1 --trade-off4 0.5 --epochs 15 --seed 2 --log logs/office20/A2D 
CUDA_VISIBLE_DEVICES=1 python cdan_jan_com.py data/office31 -d Office31 -s D -t A -a resnet50 --trade-off1 1 --trade-off2 1 --trade-off3 1 --trade-off4 0.5 --epochs 15 --seed 2 --log logs/office20/D2A 
CUDA_VISIBLE_DEVICES=2 python cdan_jan_com.py data/office31 -d Office31 -s D -t W -a resnet50 --trade-off1 1 --trade-off2 1 --trade-off3 1 --trade-off4 0.5 --epochs 15 --seed 2 --log logs/office20/D2W 
CUDA_VISIBLE_DEVICES=6 python cdan_jan_com.py data/office31 -d Office31 -s W -t D -a resnet50 --trade-off1 1 --trade-off2 1 --trade-off3 1 --trade-off4 0.5 --epochs 15 --seed 2 --log logs/office20/W2D 

15. Adjustment factors in loss items, (beta_1, beta_2, beta_3, beta_4) = (3, 1, 1, 1)

cdan_jan_com.py

CUDA_VISIBLE_DEVICES=2 python cdan_jan_com.py data/office31 -d Office31 -s W -t A -a resnet50 --trade-off1 3 --trade-off2 1 --trade-off3 1 --trade-off4 1 --epochs 15 --seed 2 --log logs/office27/W2A   
CUDA_VISIBLE_DEVICES=3 python cdan_jan_com.py data/office31 -d Office31 -s A -t W -a resnet50 --trade-off1 3 --trade-off2 1 --trade-off3 1 --trade-off4 1 --epochs 15 --seed 2 --log logs/office27/A2W   
CUDA_VISIBLE_DEVICES=5 python cdan_jan_com.py data/office31 -d Office31 -s A -t D -a resnet50 --trade-off1 3 --trade-off2 1 --trade-off3 1 --trade-off4 1 --epochs 15 --seed 2 --log logs/office27/A2D    
CUDA_VISIBLE_DEVICES=0 python cdan_jan_com.py data/office31 -d Office31 -s D -t A -a resnet50 --trade-off1 3 --trade-off2 1 --trade-off3 1 --trade-off4 1 --epochs 15 --seed 2 --log logs/office27/D2A 
CUDA_VISIBLE_DEVICES=1 python cdan_jan_com.py data/office31 -d Office31 -s D -t W -a resnet50 --trade-off1 3 --trade-off2 1 --trade-off3 1 --trade-off4 1 --epochs 15 --seed 2 --log logs/office27/D2W 
CUDA_VISIBLE_DEVICES=6 python cdan_jan_com.py data/office31 -d Office31 -s W -t D -a resnet50 --trade-off1 3 --trade-off2 1 --trade-off3 1 --trade-off4 1 --epochs 15 --seed 2 --log logs/office27/W2D 

16. Adjustment factors in loss items, (beta_1, beta_2, beta_3, beta_4) = (1, 3, 1, 1)

cdan_jan_com.py

CUDA_VISIBLE_DEVICES=4 python cdan_jan_com.py data/office31 -d Office31 -s W -t A -a resnet50 --trade-off1 1 --trade-off2 3 --trade-off3 1 --trade-off4 1 --epochs 15 --seed 2 --log logs/office28/W2A 
CUDA_VISIBLE_DEVICES=7 python cdan_jan_com.py data/office31 -d Office31 -s A -t W -a resnet50 --trade-off1 1 --trade-off2 3 --trade-off3 1 --trade-off4 1 --epochs 15 --seed 2 --log logs/office28/A2W 
CUDA_VISIBLE_DEVICES=6 python cdan_jan_com.py data/office31 -d Office31 -s A -t D -a resnet50 --trade-off1 1 --trade-off2 3 --trade-off3 1 --trade-off4 1 --epochs 15 --seed 2 --log logs/office28/A2D 
CUDA_VISIBLE_DEVICES=0 python cdan_jan_com.py data/office31 -d Office31 -s D -t A -a resnet50 --trade-off1 1 --trade-off2 3 --trade-off3 1 --trade-off4 1 --epochs 15 --seed 2 --log logs/office28/D2A 
CUDA_VISIBLE_DEVICES=1 python cdan_jan_com.py data/office31 -d Office31 -s D -t W -a resnet50 --trade-off1 1 --trade-off2 3 --trade-off3 1 --trade-off4 1 --epochs 15 --seed 2 --log logs/office28/D2W 
CUDA_VISIBLE_DEVICES=6 python cdan_jan_com.py data/office31 -d Office31 -s W -t D -a resnet50 --trade-off1 1 --trade-off2 3 --trade-off3 1 --trade-off4 1 --epochs 15 --seed 2 --log logs/office28/W2D 

17. Adjustment factors in loss items, (beta_1, beta_2, beta_3, beta_4) = (1, 1, 3, 1)

cdan_jan_com.py

CUDA_VISIBLE_DEVICES=2 python cdan_jan_com.py data/office31 -d Office31 -s W -t A -a resnet50 --trade-off1 1 --trade-off2 1 --trade-off3 0.5 --trade-off4 1 --epochs 15 --seed 2 --log logs/office29/W2A 
CUDA_VISIBLE_DEVICES=3 python cdan_jan_com.py data/office31 -d Office31 -s A -t W -a resnet50 --trade-off1 1 --trade-off2 1 --trade-off3 0.5 --trade-off4 1 --epochs 15 --seed 2 --log logs/office29/A2W 
CUDA_VISIBLE_DEVICES=4 python cdan_jan_com.py data/office31 -d Office31 -s A -t D -a resnet50 --trade-off1 1 --trade-off2 1 --trade-off3 0.5 --trade-off4 1 --epochs 15 --seed 2 --log logs/office29/A2D 
CUDA_VISIBLE_DEVICES=5 python cdan_jan_com.py data/office31 -d Office31 -s D -t A -a resnet50 --trade-off1 1 --trade-off2 1 --trade-off3 0.5 --trade-off4 1 --epochs 15 --seed 2 --log logs/office29/D2A 
CUDA_VISIBLE_DEVICES=6 python cdan_jan_com.py data/office31 -d Office31 -s D -t W -a resnet50 --trade-off1 1 --trade-off2 1 --trade-off3 0.5 --trade-off4 1 --epochs 15 --seed 2 --log logs/office29/D2W 
CUDA_VISIBLE_DEVICES=6 python cdan_jan_com.py data/office31 -d Office31 -s W -t D -a resnet50 --trade-off1 1 --trade-off2 1 --trade-off3 0.5 --trade-off4 1 --epochs 15 --seed 2 --log logs/office29/W2D 

18. Adjustment factors in loss items, (beta_1, beta_2, beta_3, beta_4) = (1, 1, 1, 3)

cdan_jan_com.py

CUDA_VISIBLE_DEVICES=7 python cdan_jan_com.py data/office31 -d Office31 -s W -t A -a resnet50 --trade-off1 1 --trade-off2 1 --trade-off3 1 --trade-off4 0.5 --epochs 15 --seed 2 --log logs/office30/W2A 
CUDA_VISIBLE_DEVICES=5 python cdan_jan_com.py data/office31 -d Office31 -s A -t W -a resnet50 --trade-off1 1 --trade-off2 1 --trade-off3 1 --trade-off4 0.5 --epochs 15 --seed 2 --log logs/office30/A2W 
CUDA_VISIBLE_DEVICES=0 python cdan_jan_com.py data/office31 -d Office31 -s A -t D -a resnet50 --trade-off1 1 --trade-off2 1 --trade-off3 1 --trade-off4 0.5 --epochs 15 --seed 2 --log logs/office30/A2D 
CUDA_VISIBLE_DEVICES=1 python cdan_jan_com.py data/office31 -d Office31 -s D -t A -a resnet50 --trade-off1 1 --trade-off2 1 --trade-off3 1 --trade-off4 0.5 --epochs 15 --seed 2 --log logs/office30/D2A 
CUDA_VISIBLE_DEVICES=2 python cdan_jan_com.py data/office31 -d Office31 -s D -t W -a resnet50 --trade-off1 1 --trade-off2 1 --trade-off3 1 --trade-off4 0.5 --epochs 15 --seed 2 --log logs/office30/D2W 
CUDA_VISIBLE_DEVICES=6 python cdan_jan_com.py data/office31 -d Office31 -s W -t D -a resnet50 --trade-off1 1 --trade-off2 1 --trade-off3 1 --trade-off4 0.5 --epochs 15 --seed 2 --log logs/office30/W2D 

19. Distillation temperature factor, alpha = 1

cdan_jan_com.py

CUDA_VISIBLE_DEVICES=6 python cdan_jan_com.py data/office31 -d Office31 -s W -t A -a resnet50 --epochs 20 --seed 2 --log logs/office4/W2A 
CUDA_VISIBLE_DEVICES=6 python cdan_jan_com.py data/office31 -d Office31 -s A -t W -a resnet50 --epochs 20 --seed 2 --log logs/office2/A2W 
CUDA_VISIBLE_DEVICES=6 python cdan_jan_com.py data/office31 -d Office31 -s A -t D -a resnet50 --epochs 20 --seed 2 --log logs/office2/A2D 
CUDA_VISIBLE_DEVICES=6 python cdan_jan_com.py data/office31 -d Office31 -s D -t A -a resnet50 --epochs 20 --seed 2 --log logs/office2/D2A 
CUDA_VISIBLE_DEVICES=6 python cdan_jan_com.py data/office31 -d Office31 -s D -t W -a resnet50 --epochs 20 --seed 2 --log logs/office2/D2W 
CUDA_VISIBLE_DEVICES=6 python cdan_jan_com.py data/office31 -d Office31 -s W -t D -a resnet50 --epochs 20 --seed 2 --log logs/office2/W2D

20. Distillation temperature factor, alpha = 2

cdan_jan_com.py

CUDA_VISIBLE_DEVICES=1 python cdan_jan_com.py data/office31 -d Office31 -s W -t A -a resnet50 --tem 2 --epochs 20 --seed 2 --log logs/office4/W2A 
CUDA_VISIBLE_DEVICES=5 python cdan_jan_com.py data/office31 -d Office31 -s A -t W -a resnet50 --tem 2 --epochs 20 --seed 2 --log logs/office2/A2W 
CUDA_VISIBLE_DEVICES=6 python cdan_jan_com.py data/office31 -d Office31 -s A -t D -a resnet50 --tem 2 --epochs 20 --seed 2 --log logs/office2/A2D 
CUDA_VISIBLE_DEVICES=0 python cdan_jan_com.py data/office31 -d Office31 -s D -t A -a resnet50 --tem 2 --epochs 20 --seed 2 --log logs/office2/D2A 
CUDA_VISIBLE_DEVICES=2 python cdan_jan_com.py data/office31 -d Office31 -s D -t W -a resnet50 --tem 2 --epochs 20 --seed 2 --log logs/office2/D2W 
CUDA_VISIBLE_DEVICES=4 python cdan_jan_com.py data/office31 -d Office31 -s W -t D -a resnet50 --tem 2 --epochs 20 --seed 2 --log logs/office2/W2D 

21. Distillation temperature factor, alpha = 3

cdan_jan_com.py

CUDA_VISIBLE_DEVICES=7 python cdan_jan_com.py data/office31 -d Office31 -s W -t A -a resnet50 --tem 3 --epochs 20 --seed 2 --log logs/office4/W2A 
CUDA_VISIBLE_DEVICES=1 python cdan_jan_com.py data/office31 -d Office31 -s A -t W -a resnet50 --tem 3 --epochs 20 --seed 2 --log logs/office2/A2W 
CUDA_VISIBLE_DEVICES=5 python cdan_jan_com.py data/office31 -d Office31 -s A -t D -a resnet50 --tem 3 --epochs 20 --seed 2 --log logs/office2/A2D 
CUDA_VISIBLE_DEVICES=4 python cdan_jan_com.py data/office31 -d Office31 -s D -t A -a resnet50 --tem 3 --epochs 20 --seed 2 --log logs/office2/D2A 
CUDA_VISIBLE_DEVICES=6 python cdan_jan_com.py data/office31 -d Office31 -s D -t W -a resnet50 --tem 3 --epochs 20 --seed 2 --log logs/office2/D2W 
CUDA_VISIBLE_DEVICES=6 python cdan_jan_com.py data/office31 -d Office31 -s W -t D -a resnet50 --tem 3 --epochs 20 --seed 2 --log logs/office2/W2D 



