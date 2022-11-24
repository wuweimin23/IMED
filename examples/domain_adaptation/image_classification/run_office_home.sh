In the following, the first row means the python file in 'examples', other rows are the running scripts

A. Ensemble model in main experiments

1.Ensemble of CDANs and JAN 

cdan_jan_com.py

CUDA_VISIBLE_DEVICES=2 python cdan_jan_com.py data/office-home -d OfficeHome -s Ar -t Cl -a resnet50 --epochs 20 --distill_epochs 5 --seed 0 --log logs/cdan2/OfficeHome_Ar2Cl
CUDA_VISIBLE_DEVICES=0 python cdan_jan_com.py data/office-home -d OfficeHome -s Ar -t Pr -a resnet50 --epochs 20 --distill_epochs 5 --seed 0 --log logs/cdan2/OfficeHome_Ar2Pr 
CUDA_VISIBLE_DEVICES=4 python cdan_jan_com.py data/office-home -d OfficeHome -s Ar -t Rw -a resnet50 --epochs 20 --distill_epochs 5 --seed 0 --log logs/cdan2/OfficeHome_Ar2Rw 
CUDA_VISIBLE_DEVICES=1 python cdan_jan_com.py data/office-home -d OfficeHome -s Cl -t Ar -a resnet50 --epochs 20 --distill_epochs 5 --seed 0 --log logs/cdan2/OfficeHome_Cl2Ar
CUDA_VISIBLE_DEVICES=7 python cdan_jan_com.py data/office-home -d OfficeHome -s Cl -t Pr -a resnet50 --epochs 20 --distill_epochs 5 --seed 0 --log logs/cdan2/OfficeHome_Cl2Pr 
CUDA_VISIBLE_DEVICES=2 python cdan_jan_com.py data/office-home -d OfficeHome -s Cl -t Rw -a resnet50 --epochs 20 --distill_epochs 5 --seed 0 --log logs/cdan2/OfficeHome_Cl2Rw 
CUDA_VISIBLE_DEVICES=3 python cdan_jan_com.py data/office-home -d OfficeHome -s Pr -t Ar -a resnet50 --epochs 20 --distill_epochs 5 --seed 0 --log logs/cdan2/OfficeHome_Pr2Ar 
CUDA_VISIBLE_DEVICES=2 python cdan_jan_com.py data/office-home -d OfficeHome -s Pr -t Cl -a resnet50 --epochs 20 --distill_epochs 5 --seed 0 --log logs/cdan2/OfficeHome_Pr2Cl 
CUDA_VISIBLE_DEVICES=6 python cdan_jan_com.py data/office-home -d OfficeHome -s Pr -t Rw -a resnet50 --epochs 20 --distill_epochs 5 --seed 0 --log logs/cdan2/OfficeHome_Pr2Rw 
CUDA_VISIBLE_DEVICES=2 python cdan_jan_com.py data/office-home -d OfficeHome -s Rw -t Ar -a resnet50 --epochs 20 --distill_epochs 5 --seed 0 --log logs/cdan2/OfficeHome_Rw2Ar 
CUDA_VISIBLE_DEVICES=0 python cdan_jan_com.py data/office-home -d OfficeHome -s Rw -t Cl -a resnet50 --epochs 20 --distill_epochs 5 --seed 0 --log logs/cdan2/OfficeHome_Rw2Cl 
CUDA_VISIBLE_DEVICES=4 python cdan_jan_com.py data/office-home -d OfficeHome -s Rw -t Pr -a resnet50 --epochs 20 --distill_epochs 5 --seed 0 --log logs/cdan2/OfficeHome_Rw2Pr 

2. Ensemble of two CDANs

cdan_two_com.py

CUDA_VISIBLE_DEVICES=1 python cdan_two_com.py data/office-home -d OfficeHome -s Ar -t Cl -a resnet50 --epochs 20 --distill_epochs 5 --seed 0 --log logs/cdan1/OfficeHome_Ar2Cl 
CUDA_VISIBLE_DEVICES=1 python cdan_two_com.py data/office-home -d OfficeHome -s Ar -t Pr -a resnet50 --epochs 20 --distill_epochs 5 --seed 0 --log logs/cdan1/OfficeHome_Ar2Pr
CUDA_VISIBLE_DEVICES=1 python cdan_two_com.py data/office-home -d OfficeHome -s Ar -t Rw -a resnet50 --epochs 20 --distill_epochs 5 --seed 0 --log logs/cdan1/OfficeHome_Ar2Rw 
CUDA_VISIBLE_DEVICES=1 python cdan_two_com.py data/office-home -d OfficeHome -s Cl -t Ar -a resnet50 --epochs 20 --distill_epochs 5 --seed 0 --log logs/cdan1/OfficeHome_Cl2Ar 
CUDA_VISIBLE_DEVICES=7 python cdan_two_com.py data/office-home -d OfficeHome -s Cl -t Pr -a resnet50 --epochs 20 --distill_epochs 5 --seed 0 --log logs/cdan1/OfficeHome_Cl2Pr 
CUDA_VISIBLE_DEVICES=1 python cdan_two_com.py data/office-home -d OfficeHome -s Cl -t Rw -a resnet50 --epochs 20 --distill_epochs 5 --seed 0 --log logs/cdan1/OfficeHome_Cl2Rw 
CUDA_VISIBLE_DEVICES=7 python cdan_two_com.py data/office-home -d OfficeHome -s Pr -t Ar -a resnet50 --epochs 20 --distill_epochs 5 --seed 0 --log logs/cdan1/OfficeHome_Pr2Ar 
CUDA_VISIBLE_DEVICES=0 python cdan_two_com.py data/office-home -d OfficeHome -s Pr -t Cl -a resnet50 --epochs 20 --distill_epochs 5 --seed 0 --log logs/cdan1/OfficeHome_Pr2Cl 
CUDA_VISIBLE_DEVICES=0 python cdan_two_com.py data/office-home -d OfficeHome -s Pr -t Rw -a resnet50 --epochs 20 --distill_epochs 5 --seed 0 --log logs/cdan1/OfficeHome_Pr2Rw 
CUDA_VISIBLE_DEVICES=3 python cdan_two_com.py data/office-home -d OfficeHome -s Rw -t Ar -a resnet50 --epochs 20 --distill_epochs 5 --seed 0 --log logs/cdan1/OfficeHome_Rw2Ar
CUDA_VISIBLE_DEVICES=3 python cdan_two_com.py data/office-home -d OfficeHome -s Rw -t Cl -a resnet50 --epochs 20 --distill_epochs 5 --seed 0 --log logs/cdan1/OfficeHome_Rw2Cl 
CUDA_VISIBLE_DEVICES=1 python cdan_two_com.py data/office-home -d OfficeHome -s Rw -t Pr -a resnet50 --epochs 20 --distill_epochs 5 --seed 0 --log logs/cdan1/OfficeHome_Rw2Pr

3. Ensemble of three CDANs

cdan_three_com.py

CUDA_VISIBLE_DEVICES=3 python cdan_three_com.py data/office-home -d OfficeHome -s Ar -t Cl -a resnet50 --epochs 30 --distill_epochs 10 --seed 2 --log logs/cdan_three/OfficeHome_Ar2Cl 
CUDA_VISIBLE_DEVICES=1 python cdan_three_com.py data/office-home -d OfficeHome -s Ar -t Pr -a resnet50 --epochs 30 --distill_epochs 10 --seed 0 --log logs/cdan_three/OfficeHome_Ar2Pr
CUDA_VISIBLE_DEVICES=2 python cdan_three_com.py data/office-home -d OfficeHome -s Ar -t Rw -a resnet50 --epochs 30 --distill_epochs 10 --seed 0 --log logs/cdan_three/OfficeHome_Ar2Rw
CUDA_VISIBLE_DEVICES=3 python cdan_three_com.py data/office-home -d OfficeHome -s Cl -t Ar -a resnet50 --epochs 30 --distill_epochs 10 --seed 0 --log logs/cdan_three/OfficeHome_Cl2Ar 
CUDA_VISIBLE_DEVICES=3 python cdan_three_com.py data/office-home -d OfficeHome -s Cl -t Pr -a resnet50 --epochs 30 --distill_epochs 10 --seed 0 --log logs/cdan_three/OfficeHome_Cl2Pr 
CUDA_VISIBLE_DEVICES=0 python cdan_three_com.py data/office-home -d OfficeHome -s Cl -t Rw -a resnet50 --epochs 30 --distill_epochs 10 --seed 0 --log logs/cdan_three/OfficeHome_Cl2Rw
CUDA_VISIBLE_DEVICES=1 python cdan_three_com.py data/office-home -d OfficeHome -s Pr -t Ar -a resnet50 --epochs 30 --distill_epochs 10 --seed 0 --log logs/cdan_three/OfficeHome_Pr2Ar
CUDA_VISIBLE_DEVICES=6 python cdan_three_com.py data/office-home -d OfficeHome -s Pr -t Cl -a resnet50 --epochs 30 --distill_epochs 10 --seed 0 --log logs/cdan_three/OfficeHome_Pr2Cl
CUDA_VISIBLE_DEVICES=2 python cdan_three_com.py data/office-home -d OfficeHome -s Pr -t Rw -a resnet50 --epochs 30 --distill_epochs 10 --seed 0 --log logs/cdan_three/OfficeHome_Pr2Rw 
CUDA_VISIBLE_DEVICES=4 python cdan_three_com.py data/office-home -d OfficeHome -s Rw -t Ar -a resnet50 --epochs 30 --distill_epochs 10 --seed 0 --log logs/cdan_three/OfficeHome_Rw2Ar
CUDA_VISIBLE_DEVICES=7 python cdan_three_com.py data/office-home -d OfficeHome -s Rw -t Cl -a resnet50 --epochs 30 --distill_epochs 10 --seed 0 --log logs/cdan_three/OfficeHome_Rw2Cl
CUDA_VISIBLE_DEVICES=0 python cdan_three_com.py data/office-home -d OfficeHome -s Rw -t Pr -a resnet50 --epochs 30 --distill_epochs 10 --seed 0 --log logs/cdan_three/OfficeHome_Rw2Pr

4. Ensemble of two CDAN+MCC+SDAT

cdan_mcc_sdat_com.py

python cdan_mcc_sdat_com.py data/office-home -d OfficeHome -s Ar -t Cl -a vit_base_patch16_224 --epochs 15 --seed 0 --seed2 1 -b 24 --no-pool --log logs/cdan_mcc_sdat_vit_5/OfficeHome_Ar2Cl --log_name Ar2Cl_cdan_mcc_sdat_vit --gpu 2 --rho 0.02 --lr 0.002 
python cdan_mcc_sdat_com.py data/office-home -d OfficeHome -s Ar -t Pr -a vit_base_patch16_224 --epochs 15 --seed 0 --seed2 1 -b 24 --no-pool --log logs/cdan_mcc_sdat_vit_5/OfficeHome_Ar2Pr --log_name Ar2Pr_cdan_mcc_sdat_vit --gpu 0 --rho 0.02 --lr 0.002 
python cdan_mcc_sdat_com.py data/office-home -d OfficeHome -s Ar -t Rw -a vit_base_patch16_224 --epochs 15 --seed 0 --seed2 1 -b 24 --no-pool --log logs/cdan_mcc_sdat_vit_5/OfficeHome_Ar2Rw --log_name Ar2Rw_cdan_mcc_sdat_vit --gpu 1 --rho 0.02 --lr 0.002 
python cdan_mcc_sdat_com.py data/office-home -d OfficeHome -s Cl -t Ar -a vit_base_patch16_224 --epochs 15 --seed 0 --seed2 1 -b 24 --no-pool --log logs/cdan_mcc_sdat_vit_5/OfficeHome_Cl2Ar --log_name Cl2Ar_cdan_mcc_sdat_vit --gpu 2 --rho 0.02 --lr 0.002 
python cdan_mcc_sdat_com.py data/office-home -d OfficeHome -s Cl -t Pr -a vit_base_patch16_224 --epochs 15 --seed 0 --seed2 1 -b 24 --no-pool --log logs/cdan_mcc_sdat_vit_5/OfficeHome_Cl2Pr --log_name Cl2Pr_cdan_mcc_sdat_vit --gpu 4 --rho 0.02 --lr 0.002 
python cdan_mcc_sdat_com.py data/office-home -d OfficeHome -s Cl -t Rw -a vit_base_patch16_224 --epochs 15 --seed 0 --seed2 1 -b 24 --no-pool --log logs/cdan_mcc_sdat_vit_5/OfficeHome_Cl2Rw --log_name Cl2Rw_cdan_mcc_sdat_vit --gpu 6 --rho 0.02 --lr 0.002
python cdan_mcc_sdat_com.py data/office-home -d OfficeHome -s Pr -t Ar -a vit_base_patch16_224 --epochs 15 --seed 0 --seed2 1 -b 24 --no-pool --log logs/cdan_mcc_sdat_vit_5/OfficeHome_Pr2Ar --log_name Pr2Ar_cdan_mcc_sdat_vit --gpu 6 --rho 0.02 --lr 0.002 
python cdan_mcc_sdat_com.py data/office-home -d OfficeHome -s Pr -t Cl -a vit_base_patch16_224 --epochs 15 --seed 0 --seed2 1 -b 24 --no-pool --log logs/cdan_mcc_sdat_vit_5/OfficeHome_Pr2Cl --log_name Pr2Cl_cdan_mcc_sdat_vit --gpu 7 --rho 0.02 --lr 0.002 
python cdan_mcc_sdat_com.py data/office-home -d OfficeHome -s Pr -t Rw -a vit_base_patch16_224 --epochs 15 --seed 0 --seed2 1 -b 24 --no-pool --log logs/cdan_mcc_sdat_vit_5/OfficeHome_Pr2Rw --log_name Pr2Rw_cdan_mcc_sdat_vit --gpu 1 --rho 0.02 --lr 0.002 
python cdan_mcc_sdat_com.py data/office-home -d OfficeHome -s Rw -t Ar -a vit_base_patch16_224 --epochs 15 --seed 0 --seed2 1 -b 24 --no-pool --log logs/cdan_mcc_sdat_vit_5/OfficeHome_Rw2Ar --log_name Rw2Ar_cdan_mcc_sdat_vit --gpu 1 --rho 0.02 --lr 0.002 
python cdan_mcc_sdat_com.py data/office-home -d OfficeHome -s Rw -t Cl -a vit_base_patch16_224 --epochs 15 --seed 0 --seed2 1 -b 24 --no-pool --log logs/cdan_mcc_sdat_vit_5/OfficeHome_Rw2Cl --log_name Rw2Cl_cdan_mcc_sdat_vit --gpu 4 --rho 0.02 --lr 0.002 
python cdan_mcc_sdat_com.py data/office-home -d OfficeHome -s Rw -t Pr -a vit_base_patch16_224 --epochs 15 --seed 0 --seed2 1 -b 24 --no-pool --log logs/cdan_mcc_sdat_vit_5/OfficeHome_Rw2Pr --log_name Rw2Pr_cdan_mcc_sdat_vit --gpu 6 --rho 0.02 --lr 0.002 

B. Ablation emperiments

1. Non-instance aware

cdan_jan_com_fusion_learn.py

CUDA_VISIBLE_DEVICES=0 python cdan_jan_com_fusion_learn.py data/office-home -d OfficeHome -s Ar -t Cl -a resnet50 --epochs 10 --distill_epochs 5 --seed 2 --log logs/cdan12/OfficeHome_Ar2Cl 
CUDA_VISIBLE_DEVICES=1 python cdan_jan_com_fusion_learn.py data/office-home -d OfficeHome -s Ar -t Pr -a resnet50 --epochs 10 --distill_epochs 5 --seed 0 --log logs/cdan12/OfficeHome_Ar2Pr 
CUDA_VISIBLE_DEVICES=3 python cdan_jan_com_fusion_learn.py data/office-home -d OfficeHome -s Ar -t Rw -a resnet50 --epochs 10 --distill_epochs 5 --seed 0 --log logs/cdan12/OfficeHome_Ar2Rw 
CUDA_VISIBLE_DEVICES=4 python cdan_jan_com_fusion_learn.py data/office-home -d OfficeHome -s Cl -t Ar -a resnet50 --epochs 10 --distill_epochs 5 --seed 0 --log logs/cdan12/OfficeHome_Cl2Ar 
CUDA_VISIBLE_DEVICES=5 python cdan_jan_com_fusion_learn.py data/office-home -d OfficeHome -s Cl -t Pr -a resnet50 --epochs 10 --distill_epochs 5 --seed 0 --log logs/cdan12/OfficeHome_Cl2Pr 
CUDA_VISIBLE_DEVICES=6 python cdan_jan_com_fusion_learn.py data/office-home -d OfficeHome -s Cl -t Rw -a resnet50 --epochs 10 --distill_epochs 5 --seed 0 --log logs/cdan12/OfficeHome_Cl2Rw 
CUDA_VISIBLE_DEVICES=0 python cdan_jan_com_fusion_learn.py data/office-home -d OfficeHome -s Pr -t Ar -a resnet50 --epochs 10 --distill_epochs 5 --seed 0 --log logs/cdan12/OfficeHome_Pr2Ar
CUDA_VISIBLE_DEVICES=2 python cdan_jan_com_fusion_learn.py data/office-home -d OfficeHome -s Pr -t Cl -a resnet50 --epochs 10 --distill_epochs 5 --seed 0 --log logs/cdan12/OfficeHome_Pr2Cl 
CUDA_VISIBLE_DEVICES=4 python cdan_jan_com_fusion_learn.py data/office-home -d OfficeHome -s Pr -t Rw -a resnet50 --epochs 10 --distill_epochs 5 --seed 0 --log logs/cdan12/OfficeHome_Pr2Rw 
CUDA_VISIBLE_DEVICES=5 python cdan_jan_com_fusion_learn.py data/office-home -d OfficeHome -s Rw -t Ar -a resnet50 --epochs 10 --distill_epochs 5 --seed 0 --log logs/cdan12/OfficeHome_Rw2Ar 
CUDA_VISIBLE_DEVICES=0 python cdan_jan_com_fusion_learn.py data/office-home -d OfficeHome -s Rw -t Cl -a resnet50 --epochs 10 --distill_epochs 5 --seed 0 --log logs/cdan12/OfficeHome_Rw2Cl 
CUDA_VISIBLE_DEVICES=3 python cdan_jan_com_fusion_learn.py data/office-home -d OfficeHome -s Rw -t Pr -a resnet50 --epochs 10 --distill_epochs 5 --seed 0 --log logs/cdan12/OfficeHome_Rw2Pr 

2. Averaging ensemble model

cdan_jan_average.py

CUDA_VISIBLE_DEVICES=0 python cdan_jan_average.py data/office-home -d OfficeHome -s Ar -t Cl -a resnet50 --epochs 15 --distill_epochs 5 --seed 0 --log logs/cdan13/OfficeHome_Ar2Cl 
CUDA_VISIBLE_DEVICES=1 python cdan_jan_average.py data/office-home -d OfficeHome -s Ar -t Pr -a resnet50 --epochs 15 --distill_epochs 5 --seed 0 --log logs/cdan13/OfficeHome_Ar2Pr 
CUDA_VISIBLE_DEVICES=4 python cdan_jan_average.py data/office-home -d OfficeHome -s Ar -t Rw -a resnet50 --epochs 15 --distill_epochs 5 --seed 0 --log logs/cdan13/OfficeHome_Ar2Rw 
CUDA_VISIBLE_DEVICES=5 python cdan_jan_average.py data/office-home -d OfficeHome -s Cl -t Ar -a resnet50 --epochs 15 --distill_epochs 5 --seed 0 --log logs/cdan13/OfficeHome_Cl2Ar 
CUDA_VISIBLE_DEVICES=6 python cdan_jan_average.py data/office-home -d OfficeHome -s Cl -t Pr -a resnet50 --epochs 15 --distill_epochs 5 --seed 0 --log logs/cdan13/OfficeHome_Cl2Pr 
CUDA_VISIBLE_DEVICES=3 python cdan_jan_average.py data/office-home -d OfficeHome -s Cl -t Rw -a resnet50 --epochs 15 --distill_epochs 5 --seed 0 --log logs/cdan13/OfficeHome_Cl2Rw 
CUDA_VISIBLE_DEVICES=0 python cdan_jan_average.py data/office-home -d OfficeHome -s Pr -t Ar -a resnet50 --epochs 15 --distill_epochs 5 --seed 0 --log logs/cdan13/OfficeHome_Pr2Ar
CUDA_VISIBLE_DEVICES=3 python cdan_jan_average.py data/office-home -d OfficeHome -s Pr -t Cl -a resnet50 --epochs 15 --distill_epochs 5 --seed 0 --log logs/cdan13/OfficeHome_Pr2Cl 
CUDA_VISIBLE_DEVICES=2 python cdan_jan_average.py data/office-home -d OfficeHome -s Pr -t Rw -a resnet50 --epochs 15 --distill_epochs 5 --seed 0 --log logs/cdan13/OfficeHome_Pr2Rw 
CUDA_VISIBLE_DEVICES=3 python cdan_jan_average.py data/office-home -d OfficeHome -s Rw -t Ar -a resnet50 --epochs 15 --distill_epochs 5 --seed 0 --log logs/cdan13/OfficeHome_Rw2Ar 
CUDA_VISIBLE_DEVICES=4 python cdan_jan_average.py data/office-home -d OfficeHome -s Rw -t Cl -a resnet50 --epochs 15 --distill_epochs 5 --seed 0 --log logs/cdan13/OfficeHome_Rw2Cl 
CUDA_VISIBLE_DEVICES=5 python cdan_jan_average.py data/office-home -d OfficeHome -s Rw -t Pr -a resnet50 --epochs 15 --distill_epochs 5 --seed 0 --log logs/cdan13/OfficeHome_Rw2Pr 


3. Use fully connected linear layer other that shuffle linear layer

cdan_jan_linear_layer.py

CUDA_VISIBLE_DEVICES=6 python cdan_jan_linear_layer.py data/office-home -d OfficeHome -s Ar -t Cl -a resnet50 --epochs 15 --distill_epochs 5 --seed 0 --log logs/cdan11/OfficeHome_Ar2Cl 
CUDA_VISIBLE_DEVICES=0 python cdan_jan_linear_layer.py data/office-home -d OfficeHome -s Ar -t Pr -a resnet50 --epochs 15 --distill_epochs 5 --seed 0 --log logs/cdan11/OfficeHome_Ar2Pr 
CUDA_VISIBLE_DEVICES=1 python cdan_jan_linear_layer.py data/office-home -d OfficeHome -s Ar -t Rw -a resnet50 --epochs 15 --distill_epochs 5 --seed 0 --log logs/cdan11/OfficeHome_Ar2Rw 
CUDA_VISIBLE_DEVICES=6 python cdan_jan_linear_layer.py data/office-home -d OfficeHome -s Cl -t Ar -a resnet50 --epochs 15 --distill_epochs 5 --seed 0 --log logs/cdan11/OfficeHome_Cl2Ar 
CUDA_VISIBLE_DEVICES=0 python cdan_jan_linear_layer.py data/office-home -d OfficeHome -s Cl -t Pr -a resnet50 --epochs 15 --distill_epochs 5 --seed 0 --log logs/cdan15/OfficeHome_Cl2Pr 
CUDA_VISIBLE_DEVICES=2 python cdan_jan_linear_layer.py data/office-home -d OfficeHome -s Cl -t Rw -a resnet50 --epochs 15 --distill_epochs 5 --seed 0 --log logs/cdan15/OfficeHome_Cl2Rw 
CUDA_VISIBLE_DEVICES=3 python cdan_jan_linear_layer.py data/office-home -d OfficeHome -s Pr -t Ar -a resnet50 --epochs 15 --distill_epochs 5 --seed 0 --log logs/cdan15/OfficeHome_Pr2Ar
CUDA_VISIBLE_DEVICES=3 python cdan_jan_linear_layer.py data/office-home -d OfficeHome -s Pr -t Cl -a resnet50 --epochs 15 --distill_epochs 5 --seed 0 --log logs/cdan15/OfficeHome_Pr2Cl 
CUDA_VISIBLE_DEVICES=3 python cdan_jan_linear_layer.py data/office-home -d OfficeHome -s Pr -t Rw -a resnet50 --epochs 20 --distill_epochs 5 --seed 0 --log logs/cdan15/OfficeHome_Pr2Rw 
CUDA_VISIBLE_DEVICES=6 python cdan_jan_linear_layer.py data/office-home -d OfficeHome -s Rw -t Ar -a resnet50 --epochs 15 --distill_epochs 5 --seed 0 --log logs/cdan15/OfficeHome_Rw2Ar 
CUDA_VISIBLE_DEVICES=4 python cdan_jan_linear_layer.py data/office-home -d OfficeHome -s Rw -t Cl -a resnet50 --epochs 20 --distill_epochs 5 --seed 0 --log logs/cdan15/OfficeHome_Rw2Cl 
CUDA_VISIBLE_DEVICES=5 python cdan_jan_linear_layer.py data/office-home -d OfficeHome -s Rw -t Pr -a resnet50 --epochs 15 --distill_epochs 5 --seed 0 --log logs/cdan15/OfficeHome_Rw2Pr 

4. Group_num = 128 in shuffle linear layer

cdan_jan_com.py

CUDA_VISIBLE_DEVICES=2 python cdan_jan_com.py data/office-home -d OfficeHome -s Ar -t Cl -a resnet50 --epochs 20 --distill_epochs 5 --seed 0 --log logs/cdan2/OfficeHome_Ar2Cl
CUDA_VISIBLE_DEVICES=0 python cdan_jan_com.py data/office-home -d OfficeHome -s Ar -t Pr -a resnet50 --epochs 20 --distill_epochs 5 --seed 0 --log logs/cdan2/OfficeHome_Ar2Pr 
CUDA_VISIBLE_DEVICES=4 python cdan_jan_com.py data/office-home -d OfficeHome -s Ar -t Rw -a resnet50 --epochs 20 --distill_epochs 5 --seed 0 --log logs/cdan2/OfficeHome_Ar2Rw 
CUDA_VISIBLE_DEVICES=1 python cdan_jan_com.py data/office-home -d OfficeHome -s Cl -t Ar -a resnet50 --epochs 20 --distill_epochs 5 --seed 0 --log logs/cdan2/OfficeHome_Cl2Ar
CUDA_VISIBLE_DEVICES=7 python cdan_jan_com.py data/office-home -d OfficeHome -s Cl -t Pr -a resnet50 --epochs 20 --distill_epochs 5 --seed 0 --log logs/cdan2/OfficeHome_Cl2Pr 
CUDA_VISIBLE_DEVICES=2 python cdan_jan_com.py data/office-home -d OfficeHome -s Cl -t Rw -a resnet50 --epochs 20 --distill_epochs 5 --seed 0 --log logs/cdan2/OfficeHome_Cl2Rw 
CUDA_VISIBLE_DEVICES=3 python cdan_jan_com.py data/office-home -d OfficeHome -s Pr -t Ar -a resnet50 --epochs 20 --distill_epochs 5 --seed 0 --log logs/cdan2/OfficeHome_Pr2Ar 
CUDA_VISIBLE_DEVICES=2 python cdan_jan_com.py data/office-home -d OfficeHome -s Pr -t Cl -a resnet50 --epochs 20 --distill_epochs 5 --seed 0 --log logs/cdan2/OfficeHome_Pr2Cl 
CUDA_VISIBLE_DEVICES=6 python cdan_jan_com.py data/office-home -d OfficeHome -s Pr -t Rw -a resnet50 --epochs 20 --distill_epochs 5 --seed 0 --log logs/cdan2/OfficeHome_Pr2Rw 
CUDA_VISIBLE_DEVICES=2 python cdan_jan_com.py data/office-home -d OfficeHome -s Rw -t Ar -a resnet50 --epochs 20 --distill_epochs 5 --seed 0 --log logs/cdan2/OfficeHome_Rw2Ar 
CUDA_VISIBLE_DEVICES=0 python cdan_jan_com.py data/office-home -d OfficeHome -s Rw -t Cl -a resnet50 --epochs 20 --distill_epochs 5 --seed 0 --log logs/cdan2/OfficeHome_Rw2Cl 
CUDA_VISIBLE_DEVICES=4 python cdan_jan_com.py data/office-home -d OfficeHome -s Rw -t Pr -a resnet50 --epochs 20 --distill_epochs 5 --seed 0 --log logs/cdan2/OfficeHome_Rw2Pr 

5. Group_num = 32 in shuffle linear layer

cdan_jan_com_share_group.py

CUDA_VISIBLE_DEVICES=1 python cdan_jan_com_share_group.py data/office-home -d OfficeHome -s Ar -t Cl -a resnet50 --epochs 20 --distill_epochs 5 --seed 2 --log logs/cdan3/OfficeHome_Ar2Cl --group_num 32 
CUDA_VISIBLE_DEVICES=0 python cdan_jan_com_share_group.py data/office-home -d OfficeHome -s Ar -t Pr -a resnet50 --epochs 20 --distill_epochs 5 --seed 0 --log logs/cdan3/OfficeHome_Ar2Pr --group_num 32 
CUDA_VISIBLE_DEVICES=1 python cdan_jan_com_share_group.py data/office-home -d OfficeHome -s Ar -t Rw -a resnet50 --epochs 20 --distill_epochs 5 --seed 0 --log logs/cdan3/OfficeHome_Ar2Rw --group_num 32 
CUDA_VISIBLE_DEVICES=5 python cdan_jan_com_share_group.py data/office-home -d OfficeHome -s Cl -t Ar -a resnet50 --epochs 20 --distill_epochs 5 --seed 0 --log logs/cdan3/OfficeHome_Cl2Ar --group_num 32 
CUDA_VISIBLE_DEVICES=6 python cdan_jan_com_share_group.py data/office-home -d OfficeHome -s Cl -t Pr -a resnet50 --epochs 20 --distill_epochs 5 --seed 0 --log logs/cdan3/OfficeHome_Cl2Pr --group_num 32 
CUDA_VISIBLE_DEVICES=2 python cdan_jan_com_share_group.py data/office-home -d OfficeHome -s Cl -t Rw -a resnet50 --epochs 20 --distill_epochs 5 --seed 0 --log logs/cdan3/OfficeHome_Cl2Rw --group_num 32 
CUDA_VISIBLE_DEVICES=0 python cdan_jan_com_share_group.py data/office-home -d OfficeHome -s Pr -t Ar -a resnet50 --epochs 20 --distill_epochs 5 --seed 0 --log logs/cdan3/OfficeHome_Pr2Ar --group_num 32 
CUDA_VISIBLE_DEVICES=3 python cdan_jan_com_share_group.py data/office-home -d OfficeHome -s Pr -t Cl -a resnet50 --epochs 20 --distill_epochs 5 --seed 0 --log logs/cdan3/OfficeHome_Pr2Cl --group_num 32
CUDA_VISIBLE_DEVICES=4 python cdan_jan_com_share_group.py data/office-home -d OfficeHome -s Pr -t Rw -a resnet50 --epochs 20 --distill_epochs 5 --seed 0 --log logs/cdan3/OfficeHome_Pr2Rw --group_num 32
CUDA_VISIBLE_DEVICES=5 python cdan_jan_com_share_group.py data/office-home -d OfficeHome -s Rw -t Ar -a resnet50 --epochs 20 --distill_epochs 5 --seed 0 --log logs/cdan3/OfficeHome_Rw2Ar --group_num 32 
CUDA_VISIBLE_DEVICES=6 python cdan_jan_com_share_group.py data/office-home -d OfficeHome -s Rw -t Cl -a resnet50 --epochs 20 --distill_epochs 5 --seed 0 --log logs/cdan3/OfficeHome_Rw2Cl --group_num 32 
CUDA_VISIBLE_DEVICES=1 python cdan_jan_com_share_group.py data/office-home -d OfficeHome -s Rw -t Pr -a resnet50 --epochs 20 --distill_epochs 5 --seed 0 --log logs/cdan3/OfficeHome_Rw2Pr --group_num 32 

6. Group_num = 16 in shuffle linear layer

cdan_jan_com_share_group.py

CUDA_VISIBLE_DEVICES=2 python cdan_jan_com_share_group.py data/office-home -d OfficeHome -s Ar -t Cl -a resnet50 --epochs 20 --distill_epochs 5 --seed 2 --log logs/cdan4/OfficeHome_Ar2Cl --group_num 16
CUDA_VISIBLE_DEVICES=0 python cdan_jan_com_share_group.py data/office-home -d OfficeHome -s Ar -t Pr -a resnet50 --epochs 20 --distill_epochs 5 --seed 0 --log logs/cdan4/OfficeHome_Ar2Pr --group_num 16 
CUDA_VISIBLE_DEVICES=3 python cdan_jan_com_share_group.py data/office-home -d OfficeHome -s Ar -t Rw -a resnet50 --epochs 20 --distill_epochs 5 --seed 0 --log logs/cdan4/OfficeHome_Ar2Rw --group_num 16 
CUDA_VISIBLE_DEVICES=0 python cdan_jan_com_share_group.py data/office-home -d OfficeHome -s Cl -t Ar -a resnet50 --epochs 20 --distill_epochs 5 --seed 0 --log logs/cdan4/OfficeHome_Cl2Ar --group_num 16 
CUDA_VISIBLE_DEVICES=1 python cdan_jan_com_share_group.py data/office-home -d OfficeHome -s Cl -t Pr -a resnet50 --epochs 20 --distill_epochs 5 --seed 0 --log logs/cdan4/OfficeHome_Cl2Pr --group_num 16 
CUDA_VISIBLE_DEVICES=2 python cdan_jan_com_share_group.py data/office-home -d OfficeHome -s Cl -t Rw -a resnet50 --epochs 20 --distill_epochs 5 --seed 0 --log logs/cdan4/OfficeHome_Cl2Rw --group_num 16 
CUDA_VISIBLE_DEVICES=3 python cdan_jan_com_share_group.py data/office-home -d OfficeHome -s Pr -t Ar -a resnet50 --epochs 20 --distill_epochs 5 --seed 0 --log logs/cdan4/OfficeHome_Pr2Ar --group_num 16 
CUDA_VISIBLE_DEVICES=4 python cdan_jan_com_share_group.py data/office-home -d OfficeHome -s Pr -t Cl -a resnet50 --epochs 20 --distill_epochs 5 --seed 0 --log logs/cdan4/OfficeHome_Pr2Cl --group_num 16 
CUDA_VISIBLE_DEVICES=5 python cdan_jan_com_share_group.py data/office-home -d OfficeHome -s Pr -t Rw -a resnet50 --epochs 20 --distill_epochs 5 --seed 0 --log logs/cdan4/OfficeHome_Pr2Rw --group_num 16 
CUDA_VISIBLE_DEVICES=6 python cdan_jan_com_share_group.py data/office-home -d OfficeHome -s Rw -t Ar -a resnet50 --epochs 20 --distill_epochs 5 --seed 0 --log logs/cdan4/OfficeHome_Rw2Ar --group_num 16 
CUDA_VISIBLE_DEVICES=0 python cdan_jan_com_share_group.py data/office-home -d OfficeHome -s Rw -t Cl -a resnet50 --epochs 20 --distill_epochs 5 --seed 0 --log logs/cdan4/OfficeHome_Rw2Cl --group_num 16 
CUDA_VISIBLE_DEVICES=1 python cdan_jan_com_share_group.py data/office-home -d OfficeHome -s Rw -t Pr -a resnet50 --epochs 20 --distill_epochs 5 --seed 0 --log logs/cdan4/OfficeHome_Rw2Pr --group_num 16 

7. There is one layer in fusion sub-network

cdan_jan_fusion_1_layer.py

CUDA_VISIBLE_DEVICES=3 python cdan_jan_fusion_1_layer.py data/office-home -d OfficeHome -s Ar -t Cl -a resnet50 --epochs 20 --distill_epochs 5 --seed 0 --log logs/cdan15/OfficeHome_Ar2Cl
CUDA_VISIBLE_DEVICES=4 python cdan_jan_fusion_1_layer.py data/office-home -d OfficeHome -s Ar -t Pr -a resnet50 --epochs 20 --distill_epochs 5 --seed 0 --log logs/cdan15/OfficeHome_Ar2Pr 
CUDA_VISIBLE_DEVICES=6 python cdan_jan_fusion_1_layer.py data/office-home -d OfficeHome -s Ar -t Rw -a resnet50 --epochs 20 --distill_epochs 5 --seed 0 --log logs/cdan15/OfficeHome_Ar2Rw 
CUDA_VISIBLE_DEVICES=2 python cdan_jan_fusion_1_layer.py data/office-home -d OfficeHome -s Cl -t Ar -a resnet50 --epochs 20 --distill_epochs 5 --seed 0 --log logs/cdan15/OfficeHome_Cl2Ar
CUDA_VISIBLE_DEVICES=3 python cdan_jan_fusion_1_layer.py data/office-home -d OfficeHome -s Cl -t Pr -a resnet50 --epochs 20 --distill_epochs 5 --seed 0 --log logs/cdan15/OfficeHome_Cl2Pr 
CUDA_VISIBLE_DEVICES=4 python cdan_jan_fusion_1_layer.py data/office-home -d OfficeHome -s Cl -t Rw -a resnet50 --epochs 20 --distill_epochs 5 --seed 0 --log logs/cdan15/OfficeHome_Cl2Rw 
CUDA_VISIBLE_DEVICES=5 python cdan_jan_fusion_1_layer.py data/office-home -d OfficeHome -s Pr -t Ar -a resnet50 --epochs 20 --distill_epochs 5 --seed 0 --log logs/cdan15/OfficeHome_Pr2Ar 
CUDA_VISIBLE_DEVICES=0 python cdan_jan_fusion_1_layer.py data/office-home -d OfficeHome -s Pr -t Cl -a resnet50 --epochs 20 --distill_epochs 5 --seed 0 --log logs/cdan15/OfficeHome_Pr2Cl 
CUDA_VISIBLE_DEVICES=1 python cdan_jan_fusion_1_layer.py data/office-home -d OfficeHome -s Pr -t Rw -a resnet50 --epochs 20 --distill_epochs 5 --seed 0 --log logs/cdan15/OfficeHome_Pr2Rw 
CUDA_VISIBLE_DEVICES=2 python cdan_jan_fusion_1_layer.py data/office-home -d OfficeHome -s Rw -t Ar -a resnet50 --epochs 20 --distill_epochs 5 --seed 0 --log logs/cdan15/OfficeHome_Rw2Ar 
CUDA_VISIBLE_DEVICES=0 python cdan_jan_fusion_1_layer.py data/office-home -d OfficeHome -s Rw -t Cl -a resnet50 --epochs 20 --distill_epochs 5 --seed 0 --log logs/cdan15/OfficeHome_Rw2Cl 
CUDA_VISIBLE_DEVICES=4 python cdan_jan_fusion_1_layer.py data/office-home -d OfficeHome -s Rw -t Pr -a resnet50 --epochs 20 --distill_epochs 5 --seed 0 --log logs/cdan15/OfficeHome_Rw2Pr 

8.There are two layers in fusion sub-network

cdan_jan_com.py

CUDA_VISIBLE_DEVICES=2 python cdan_jan_com.py data/office-home -d OfficeHome -s Ar -t Cl -a resnet50 --epochs 20 --distill_epochs 5 --seed 0 --log logs/cdan2/OfficeHome_Ar2Cl
CUDA_VISIBLE_DEVICES=0 python cdan_jan_com.py data/office-home -d OfficeHome -s Ar -t Pr -a resnet50 --epochs 20 --distill_epochs 5 --seed 0 --log logs/cdan2/OfficeHome_Ar2Pr 
CUDA_VISIBLE_DEVICES=4 python cdan_jan_com.py data/office-home -d OfficeHome -s Ar -t Rw -a resnet50 --epochs 20 --distill_epochs 5 --seed 0 --log logs/cdan2/OfficeHome_Ar2Rw 
CUDA_VISIBLE_DEVICES=1 python cdan_jan_com.py data/office-home -d OfficeHome -s Cl -t Ar -a resnet50 --epochs 20 --distill_epochs 5 --seed 0 --log logs/cdan2/OfficeHome_Cl2Ar
CUDA_VISIBLE_DEVICES=7 python cdan_jan_com.py data/office-home -d OfficeHome -s Cl -t Pr -a resnet50 --epochs 20 --distill_epochs 5 --seed 0 --log logs/cdan2/OfficeHome_Cl2Pr 
CUDA_VISIBLE_DEVICES=2 python cdan_jan_com.py data/office-home -d OfficeHome -s Cl -t Rw -a resnet50 --epochs 20 --distill_epochs 5 --seed 0 --log logs/cdan2/OfficeHome_Cl2Rw 
CUDA_VISIBLE_DEVICES=3 python cdan_jan_com.py data/office-home -d OfficeHome -s Pr -t Ar -a resnet50 --epochs 20 --distill_epochs 5 --seed 0 --log logs/cdan2/OfficeHome_Pr2Ar 
CUDA_VISIBLE_DEVICES=2 python cdan_jan_com.py data/office-home -d OfficeHome -s Pr -t Cl -a resnet50 --epochs 20 --distill_epochs 5 --seed 0 --log logs/cdan2/OfficeHome_Pr2Cl 
CUDA_VISIBLE_DEVICES=6 python cdan_jan_com.py data/office-home -d OfficeHome -s Pr -t Rw -a resnet50 --epochs 20 --distill_epochs 5 --seed 0 --log logs/cdan2/OfficeHome_Pr2Rw 
CUDA_VISIBLE_DEVICES=2 python cdan_jan_com.py data/office-home -d OfficeHome -s Rw -t Ar -a resnet50 --epochs 20 --distill_epochs 5 --seed 0 --log logs/cdan2/OfficeHome_Rw2Ar 
CUDA_VISIBLE_DEVICES=0 python cdan_jan_com.py data/office-home -d OfficeHome -s Rw -t Cl -a resnet50 --epochs 20 --distill_epochs 5 --seed 0 --log logs/cdan2/OfficeHome_Rw2Cl 
CUDA_VISIBLE_DEVICES=4 python cdan_jan_com.py data/office-home -d OfficeHome -s Rw -t Pr -a resnet50 --epochs 20 --distill_epochs 5 --seed 0 --log logs/cdan2/OfficeHome_Rw2Pr 

9. There are three layers in fusion sub-network

cdan_jan_fusion_3_layer.py

CUDA_VISIBLE_DEVICES=3 python cdan_jan_fusion_3_layer.py data/office-home -d OfficeHome -s Ar -t Cl -a resnet50 --epochs 20 --distill_epochs 5 --seed 0 --log logs/cdan16/OfficeHome_Ar2Cl
CUDA_VISIBLE_DEVICES=4 python cdan_jan_fusion_3_layer.py data/office-home -d OfficeHome -s Ar -t Pr -a resnet50 --epochs 20 --distill_epochs 5 --seed 0 --log logs/cdan16/OfficeHome_Ar2Pr 
CUDA_VISIBLE_DEVICES=4 python cdan_jan_fusion_3_layer.py data/office-home -d OfficeHome -s Ar -t Rw -a resnet50 --epochs 20 --distill_epochs 5 --seed 0 --log logs/cdan16/OfficeHome_Ar2Rw 
CUDA_VISIBLE_DEVICES=1 python cdan_jan_fusion_3_layer.py data/office-home -d OfficeHome -s Cl -t Ar -a resnet50 --epochs 20 --distill_epochs 5 --seed 0 --log logs/cdan16/OfficeHome_Cl2Ar
CUDA_VISIBLE_DEVICES=7 python cdan_jan_fusion_3_layer.py data/office-home -d OfficeHome -s Cl -t Pr -a resnet50 --epochs 20 --distill_epochs 5 --seed 0 --log logs/cdan16/OfficeHome_Cl2Pr 
CUDA_VISIBLE_DEVICES=2 python cdan_jan_fusion_3_layer.py data/office-home -d OfficeHome -s Cl -t Rw -a resnet50 --epochs 20 --distill_epochs 5 --seed 0 --log logs/cdan16/OfficeHome_Cl2Rw 
CUDA_VISIBLE_DEVICES=3 python cdan_jan_fusion_3_layer.py data/office-home -d OfficeHome -s Pr -t Ar -a resnet50 --epochs 20 --distill_epochs 5 --seed 0 --log logs/cdan16/OfficeHome_Pr2Ar 
CUDA_VISIBLE_DEVICES=1 python cdan_jan_fusion_3_layer.py data/office-home -d OfficeHome -s Pr -t Cl -a resnet50 --epochs 20 --distill_epochs 5 --seed 0 --log logs/cdan16/OfficeHome_Pr2Cl 
CUDA_VISIBLE_DEVICES=6 python cdan_jan_fusion_3_layer.py data/office-home -d OfficeHome -s Pr -t Rw -a resnet50 --epochs 20 --distill_epochs 5 --seed 0 --log logs/cdan16/OfficeHome_Pr2Rw 
CUDA_VISIBLE_DEVICES=2 python cdan_jan_fusion_3_layer.py data/office-home -d OfficeHome -s Rw -t Ar -a resnet50 --epochs 20 --distill_epochs 5 --seed 0 --log logs/cdan16/OfficeHome_Rw2Ar 
CUDA_VISIBLE_DEVICES=0 python cdan_jan_fusion_3_layer.py data/office-home -d OfficeHome -s Rw -t Cl -a resnet50 --epochs 20 --distill_epochs 5 --seed 0 --log logs/cdan16/OfficeHome_Rw2Cl 
CUDA_VISIBLE_DEVICES=4 python cdan_jan_fusion_3_layer.py data/office-home -d OfficeHome -s Rw -t Pr -a resnet50 --epochs 20 --distill_epochs 5 --seed 0 --log logs/cdan16/OfficeHome_Rw2Pr 