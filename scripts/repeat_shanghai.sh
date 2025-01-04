for i in {1..10}
do
    CUDA_VISIBLE_DEVICES=0 python model/train_urbansparse.py --dataset Shanghai
done