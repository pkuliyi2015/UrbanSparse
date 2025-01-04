for i in {1..5}
do
    CUDA_VISIBLE_DEVICES=6 /home_nfs/regionteam/miniconda3/envs/GeoBloom/bin/python model/train_urbansparse.py --dataset Shanghai --no-pred
done