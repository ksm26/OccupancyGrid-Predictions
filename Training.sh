export CUDA_VISIBLE_DEVICES=0

# Parameter details: 
#   --is_training : 1 for Training, 0 otherwise
#   --use_std_rnn : Use standard ConvLSTM instead of SpatioTemporalLSTM, this is an baseline
#   --device : cuda
#   --saveckpt_step : Save checkpoint after certain number of epochs
#   --pretrained_network: Path to pretrained model
#   --traindata : Path to training dataset
#   --testdata : Path to testing dataset
#   --saveckpt_path : Path to save checkpoints
#   --savetestimages : Path to save Test images
#   --seq_len : Total length of the sequence
#   --input_len : Input length to the prediction model
#   --seqimg_gap : Gap between two sequences (least value: 1)
#   --img_width : Original image width
#   --img_height : Original image height
#   --resize_img : Bool for resizing the original image
#   --num_hidden : Number of hidden layers
#   --use_combinedStaticSemantic : Combined Static and Semantic objects
#   --use_StaticSemantic : Separate Static and Semantic objects
#   --use_StaticFull : Input is static and full image and predict separate Static and Semantic objects
#   --use_semantic_masking : Use masking for semantic labels in separate static-semantic-prediction
#   --reverse_scheduled_sampling : Boolean for choosing the training scheme
#   --scheduled_sampling : Training scheme


python3 -u main.py \
    --is_training 1 \
    --use_std_rnn 0 \
    --device cuda \
    --saveckpt_step 1 \
    --traindata /.../star_predrnn/dataset/nuscenes_train \
    --testdata  /.../star_predrnn/dataset/nuscenes_test \
    --saveckpt_path /.../star_predrnn/checkpoints/StaticSemantic_masked \
    --savetestimages /.../star_predrnn/results/StaticSemantic_masked \
    --seq_len 10 \
    --input_len 6 \
    --seqimg_gap 2 \
    --img_channels 1 \
    --img_width 600 \
    --img_height 600 \
    --resize_img 1 \
    --resize_img_ht 256 \
    --resize_img_wd 256 \
    --model_name predrnn \
    --use_combinedStaticSemantic 0 \
    --use_StaticSemantic 1 \
    --use_StaticFull 0 \
    --use_semantic_masking 0 \
    --filter_size 5 \
    --stride 1 \
    --reverse_scheduled_sampling 1 \
    --r_sampling_step1 25000 \
    --r_sampling_step2 50000 \
    --r_exp_alpha 5000 \
    --scheduled_sampling 1 \
    --sampling_stop_iter 50000 \
    --sampling_start_value 1.0 \
    --sampling_changing_rate 0.00002 \
    --k_loss 10 \
    --lr 0.0003 \
    --batch_size 1 \
    --epochs 30 \
    --display_batch_interval 500 \
    --optim_step 8 