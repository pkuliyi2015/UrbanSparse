for i in {1..5}
do
    CUDA_VISIBLE_DEVICES=0 /home_nfs/regionteam/miniconda3/envs/GeoBloom/bin/python model/train_urbansparse.py --dataset Beijing
done