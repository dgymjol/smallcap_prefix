
experiment_name="$1"
gpu="$2"


PYTHONPATH=/workspace/smallcap_prefix CUDA_VISIBLE_DEVICES=${gpu} python train.py --i2t_features_dir /workspace/smallcap_prefix/i2t_features_77 --features_dir /workspace/smallcap_prefix/features --mlp_path /workspace/smallcap_prefix/pic2word.pt --experiments_dir cat_openai_finetuning --train_mlp
PYTHONPATH=/workspace/smallcap_prefix CUDA_VISIBLE_DEVICES=${gpu} python train.py --i2t_features_dir /workspace/smallcap_prefix/i2t_features_77 --features_dir /workspace/smallcap_prefix/features --mlp_path /workspace/smallcap_prefix/pic2word.pt --experiments_dir cat_openai_finetuning


for var in 17711 35422 53133 70844 88555 106266 123977 141688
do
  PYTHONPATH=/workspace/smallcap_prefix CUDA_VISIBLE_DEVICES=${gpu} python infer.py \
                  --i2t_features_path /workspace/smallcap_prefix/i2t_features_77/val.hdf5 \
                  --features_path /workspace/smallcap_prefix/features/val.hdf5 \
                  --model_path /workspace/smallcap_prefix/${experiment_name}/rag_7M_gpt2 \
                  --checkpoint_path checkpoint-${var}
  CUDA_VISIBLE_DEVICES=${gpu} python coco-caption/run_eval.py coco-caption/annotations/captions_valKarpathy.json "${experiment_name}/rag_7M_gpt2/checkpoint-${var}/val_preds.json" > "${experiment_name}/val_${var}.txt"

done

for var in 17711 35422 53133 70844 88555 106266 123977 141688
do
  PYTHONPATH=/workspace/smallcap_prefix CUDA_VISIBLE_DEVICES=${gpu} python infer.py \
                  --i2t_features_path /workspace/smallcap_prefix/i2t_features_77/val.hdf5 \
                  --features_path /workspace/smallcap_prefix/features/val.hdf5 \
                  --model_path /workspace/smallcap_prefix/${experiment_name}/rag_7M_gpt2 \
                  --checkpoint_path checkpoint-${var} \
                  --infer_test
  CUDA_VISIBLE_DEVICES=${gpu} python coco-caption/run_eval.py coco-caption/annotations/captions_testKarpathy.json "${experiment_name}/rag_7M_gpt2/checkpoint-${var}/test_preds.json" > "${experiment_name}/test_${var}.txt"

done