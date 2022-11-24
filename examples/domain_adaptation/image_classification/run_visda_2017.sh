In the following, the first row means the python file in 'examples', other rows are the running scripts

A. Ensemble model in main experiments

1.Ensemble of CDANs and JAN 

cdan_jan_com.py

CUDA_VISIBLE_DEVICES=5 python cdan_jan_com.py data/visda-2017 -d VisDA2017 -s Synthetic -t Real -a resnet50 --epochs 20 --distill_epochs 5 --seed 0 --per-class-eval --train-resizing cen.crop --log logs/cdan2/VisDA2017 

2. Ensemble of two CDANs

cdan_two_com.py

CUDA_VISIBLE_DEVICES=0 python cdan_two_com.py data/visda-2017 -d VisDA2017 -s Synthetic -t Real -a resnet50 --epochs 20 --distill_epochs 5 --seed 0 --per-class-eval --train-resizing cen.crop --log logs/cdan1/VisDA2017 

3. Ensemble of three CDANs

cdan_three_com.py

CUDA_VISIBLE_DEVICES=6 python cdan_three_com.py data/visda-2017 -d VisDA2017 -s Synthetic -t Real -a resnet50 --epochs 20 --distill_epochs 10 --seed 0 --per-class-eval --train-resizing cen.crop --log logs/cdan_three/VisDA2017 

4. Ensemble of two CDAN+MCC+SDAT

cdan_mcc_sdat_com.py

python cdan_mcc_sdat_com.py data/visda-2017 -d VisDA2017 -s Synthetic -t Real -a vit_base_patch16_224 --trade-off1 0.5 --trade-off2 0.5 --tem 2 --epochs 7 --seed 0 --seed2 1 --lr 0.002 --per-class-eval --train-resizing cen.crop --log logs/cdan_mcc_sdat_vit_21/VisDA2017 --log_name visda_cdan_mcc_sdat_vit --gpu 4 --no-pool --rho 0.02 