experiment_name="$1"

# for ccm pretrained weight (finetuning)

# CUDA_VISIBLE_DEVICES=1 python train.py --experiments_dir ${experiment_name}_freeze --encoder_name openai/clip-vit-large-patch14 --features_dir features --mlp_path pic2word.pt

# lr=5e-4
# CUDA_VISIBLE_DEVICES=1 python train.py --experiments_dir ${experiment_name}_finetuning_${lr} --encoder_name openai/clip-vit-large-patch14 --features_dir l14_features --train_mlp --mlp_path pic2word.pt --lr $lr
mkdir "${experiment_name}/results"
for var in 88560 79704 61992
do
  CUDA_VISIBLE_DEVICES=1 python infer.py --features_path "features/val.hdf5" --model_path "${experiment_name}/rag_7M_gpt2" --checkpoint_path checkpoint-${var}
  CUDA_VISIBLE_DEVICES=1 python coco-caption/run_eval.py coco-caption/annotations/captions_valKarpathy.json "${experiment_name}/rag_7M_gpt2/checkpoint-${var}/val_preds.json" > "${experiment_name}/results/val_${var}.txt"

#   CUDA_VISIBLE_DEVICES=0 python infer.py --model_path "${experiment_name}/rag_7M_gpt2" --checkpoint_path checkpoint-${var} --infer_test
#   CUDA_VISIBLE_DEVICES=0 python coco-caption/run_eval.py coco-caption/annotations/captions_testKarpathy.json "${experiment_name}/rag_7M_gpt2/checkpoint-${var}/test_preds.json" > "${experiment_name}/results/test_${var}.txt"
done