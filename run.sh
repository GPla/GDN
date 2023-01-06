gpu_n=$1
DATASET=$2

seed=5
BATCH_SIZE=32
SLIDE_WIN=5
dim=64
out_layer_num=1
SLIDE_STRIDE=1
topk=5
out_layer_inter_dim=128
val_ratio=0.2
decay=0

for HOST in 1-1 1-2 1-3 1-4 1-7 1-8 2-1 2-2 2-3 2-4 2-5 2-6 2-7 2-8 2-9 3-3 3-4 3-6 3-7 3-10
do
echo $HOST

path_pattern="${DATASET}/${HOST}"
COMMENT="${DATASET}/${HOST}"

EPOCH=50
report='best'

if [[ "$gpu_n" == "cpu" ]]; then
    python main.py \
        -dataset $DATASET \
        -save_path_pattern $path_pattern \
        -slide_stride $SLIDE_STRIDE \
        -slide_win $SLIDE_WIN \
        -batch $BATCH_SIZE \
        -epoch $EPOCH \
        -comment $COMMENT \
        -random_seed $seed \
        -decay $decay \
        -dim $dim \
        -out_layer_num $out_layer_num \
        -out_layer_inter_dim $out_layer_inter_dim \
        -decay $decay \
        -val_ratio $val_ratio \
        -report $report \
        -topk $topk \
        -device 'cpu'
else
    CUDA_VISIBLE_DEVICES=$gpu_n  python main.py \
        -dataset $DATASET \
        -host $HOST \
        -save_path_pattern $path_pattern \
        -slide_stride $SLIDE_STRIDE \
        -slide_win $SLIDE_WIN \
        -batch $BATCH_SIZE \
        -epoch $EPOCH \
        -comment $COMMENT \
        -random_seed $seed \
        -decay $decay \
        -dim $dim \
        -out_layer_num $out_layer_num \
        -out_layer_inter_dim $out_layer_inter_dim \
        -decay $decay \
        -val_ratio $val_ratio \
        -report $report \
        -topk $topk
fi
done