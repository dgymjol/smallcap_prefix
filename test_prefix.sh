experiment_name="$1"

# for ccm pretrained weight (finetuning)

# CUDA_VISIBLE_DEVICES=1 python train.py --experiments_dir ${experiment_name}_freeze --encoder_name openai/clip-vit-large-patch14 --features_dir oofeatures --mlp_path pic2word.pt

# lr=5e-4
CUDA_VISIBLE_DEVICES=0 python train.py --experiments_dir ${experiment_name}
# mkdir "${experiment_name}/results"
# for var in 26568 35424 44280 53136 61992 70848 79704 88560
# do
#   CUDA_VISIBLE_DEVICES=0 python infer.py --features_path "features/val.hdf5" --model_path "${experiment_name}/rag_7M_gpt2" --checkpoint_path checkpoint-${var}
#   CUDA_VISIBLE_DEVICES=0 python coco-caption/run_eval.py coco-caption/annotations/captions_valKarpathy.json "${experiment_name}/rag_7M_gpt2/checkpoint-${var}/val_preds.json" > "${experiment_name}/results/val_${var}.txt"

#   CUDA_VISIBLE_DEVICES=0 python infer.py --model_path "${experiment_name}/rag_7M_gpt2" --checkpoint_path checkpoint-${var} --infer_test
#   CUDA_VISIBLE_DEVICES=0 python coco-caption/run_eval.py coco-caption/annotations/captions_testKarpathy.json "${experiment_name}/rag_7M_gpt2/checkpoint-${var}/test_preds.json" > "${experiment_name}/results/test_${var}.txt"
# done

# lr=1e-4
# CUDA_VISIBLE_DEVICES=1 python train.py --experiments_dir ${experiment_name}_finetuning_${lr} --encoder_name openai/clip-vit-large-patch14 --features_dir l14_features --train_mlp --mlp_path pic2word.pt --lr $lr
# mkdir "${experiment_name}_finetuning_${lr}/results"
# for var in 8856 17712 26568 35424 44280 53136 61992 70848 79704 88560
# do
#   CUDA_VISIBLE_DEVICES=1 python infer.py --encoder_name "openai/clip-vit-large-patch14" --features_path "l14_features/val.hdf5" --model_path "${experiment_name}_finetuning_${lr}/rag_7M_gpt2" --checkpoint_path checkpoint-${var}
#   CUDA_VISIBLE_DEVICES=1 python coco-caption/run_eval.py coco-caption/annotations/captions_valKarpathy.json "${experiment_name}_finetuning_${lr}/rag_7M_gpt2/checkpoint-${var}/val_preds.json" > "${experiment_name}_finetuning_${lr}/results/val_${var}.txt"

#   CUDA_VISIBLE_DEVICES=1 python infer.py --encoder_name "openai/clip-vit-large-patch14" --model_path "${experiment_name}_finetuning_${lr}/rag_7M_gpt2" --checkpoint_path checkpoint-${var} --infer_test
#   CUDA_VISIBLE_DEVICES=1 python coco-caption/run_eval.py coco-caption/annotations/captions_testKarpathy.json "${experiment_name}_finetuning_${lr}/rag_7M_gpt2/checkpoint-${var}/test_preds.json" > "${experiment_name}_finetuning_${lr}/results/test_${var}.txt"
# done

# lr=5e-5
# CUDA_VISIBLE_DEVICES=1 python train.py --experiments_dir ${experiment_name}_finetuning_${lr} --encoder_name openai/clip-vit-large-patch14 --features_dir l14_features --train_mlp --mlp_path pic2word.pt --lr $lr
# mkdir "${experiment_name}_finetuning_${lr}/results"
# for var in 8856 17712 26568 35424 44280 53136 61992 70848 79704 88560
# do
#   CUDA_VISIBLE_DEVICES=1 python infer.py --encoder_name "openai/clip-vit-large-patch14" --features_path "l14_features/val.hdf5" --model_path "${experiment_name}_finetuning_${lr}/rag_7M_gpt2" --checkpoint_path checkpoint-${var}
#   CUDA_VISIBLE_DEVICES=1 python coco-caption/run_eval.py coco-caption/annotations/captions_valKarpathy.json "${experiment_name}_finetuning_${lr}/rag_7M_gpt2/checkpoint-${var}/val_preds.json" > "${experiment_name}_finetuning_${lr}/results/val_${var}.txt"

#   CUDA_VISIBLE_DEVICES=1 python infer.py --encoder_name "openai/clip-vit-large-patch14" --model_path "${experiment_name}_finetuning_${lr}/rag_7M_gpt2" --checkpoint_path checkpoint-${var} --infer_test
#   CUDA_VISIBLE_DEVICES=1 python coco-caption/run_eval.py coco-caption/annotations/captions_testKarpathy.json "${experiment_name}_finetuning_${lr}/rag_7M_gpt2/checkpoint-${var}/test_preds.json" > "${experiment_name}_finetuning_${lr}/results/test_${var}.txt"
# done

# lr=1e-5
# CUDA_VISIBLE_DEVICES=1 python train.py --experiments_dir ${experiment_name}_finetuning_${lr} --encoder_name openai/clip-vit-large-patch14 --features_dir l14_features --train_mlp --mlp_path pic2word.pt --lr $lr
# mkdir "${experiment_name}_finetuning_${lr}/results"
# for var in 8856 17712 26568 35424 44280 53136 61992 70848 79704 88560
# do
#   CUDA_VISIBLE_DEVICES=1 python infer.py --encoder_name "openai/clip-vit-large-patch14" --features_path "l14_features/val.hdf5" --model_path "${experiment_name}_finetuning_${lr}/rag_7M_gpt2" --checkpoint_path checkpoint-${var}
#   CUDA_VISIBLE_DEVICES=1 python coco-caption/run_eval.py coco-caption/annotations/captions_valKarpathy.json "${experiment_name}_finetuning_${lr}/rag_7M_gpt2/checkpoint-${var}/val_preds.json" > "${experiment_name}_finetuning_${lr}/results/val_${var}.txt"

#   CUDA_VISIBLE_DEVICES=1 python infer.py --encoder_name "openai/clip-vit-large-patch14" --model_path "${experiment_name}_finetuning_${lr}/rag_7M_gpt2" --checkpoint_path checkpoint-${var} --infer_test
#   CUDA_VISIBLE_DEVICES=1 python coco-caption/run_eval.py coco-caption/annotations/captions_testKarpathy.json "${experiment_name}_finetuning_${lr}/rag_7M_gpt2/checkpoint-${var}/test_preds.json" > "${experiment_name}_finetuning_${lr}/results/test_${var}.txt"
# done


# for ccm pretrained weight (freeze)


# lr=5e-4
# CUDA_VISIBLE_DEVICES=1 python train.py --experiments_dir ${experiment_name}_freeze_${lr} --encoder_name openai/clip-vit-large-patch14 --features_dir l14_features --mlp_path pic2word.pt --lr $lr
# mkdir "${experiment_name}_freeze_${lr}/results"
# for var in 8856 17712 26568 35424 44280 53136 61992 70848 79704 88560
# do
#   CUDA_VISIBLE_DEVICES=1 python infer.py --encoder_name "openai/clip-vit-large-patch14" --features_path "l14_features/val.hdf5" --model_path "${experiment_name}_freeze_${lr}/rag_7M_gpt2" --checkpoint_path checkpoint-${var}
#   CUDA_VISIBLE_DEVICES=1 python coco-caption/run_eval.py coco-caption/annotations/captions_valKarpathy.json "${experiment_name}_freeze_${lr}/rag_7M_gpt2/checkpoint-${var}/val_preds.json" > "${experiment_name}_freeze_${lr}/results/val_${var}.txt"

#   CUDA_VISIBLE_DEVICES=1 python infer.py --encoder_name "openai/clip-vit-large-patch14" --model_path "${experiment_name}_freeze_${lr}/rag_7M_gpt2" --checkpoint_path checkpoint-${var} --infer_test
#   CUDA_VISIBLE_DEVICES=1 python coco-caption/run_eval.py coco-caption/annotations/captions_testKarpathy.json "${experiment_name}_freeze_${lr}/rag_7M_gpt2/checkpoint-${var}/test_preds.json" > "${experiment_name}_freeze_${lr}/results/test_${var}.txt"
# done

# lr=1e-4
# CUDA_VISIBLE_DEVICES=1 python train.py --experiments_dir ${experiment_name}_freeze_${lr} --encoder_name openai/clip-vit-large-patch14 --features_dir l14_features --mlp_path pic2word.pt --lr $lr
# mkdir "${experiment_name}_freeze_${lr}/results"
# for var in 8856 17712 26568 35424 44280 53136 61992 70848 79704 88560
# do
#   CUDA_VISIBLE_DEVICES=1 python infer.py --encoder_name "openai/clip-vit-large-patch14" --features_path "l14_features/val.hdf5" --model_path "${experiment_name}_freeze_${lr}/rag_7M_gpt2" --checkpoint_path checkpoint-${var}
#   CUDA_VISIBLE_DEVICES=1 python coco-caption/run_eval.py coco-caption/annotations/captions_valKarpathy.json "${experiment_name}_freeze_${lr}/rag_7M_gpt2/checkpoint-${var}/val_preds.json" > "${experiment_name}_freeze_${lr}/results/val_${var}.txt"

#   CUDA_VISIBLE_DEVICES=1 python infer.py --encoder_name "openai/clip-vit-large-patch14" --model_path "${experiment_name}_freeze_${lr}/rag_7M_gpt2" --checkpoint_path checkpoint-${var} --infer_test
#   CUDA_VISIBLE_DEVICES=1 python coco-caption/run_eval.py coco-caption/annotations/captions_testKarpathy.json "${experiment_name}_freeze_${lr}/rag_7M_gpt2/checkpoint-${var}/test_preds.json" > "${experiment_name}_freeze_${lr}/results/test_${var}.txt"
# done

# lr=5e-5
# CUDA_VISIBLE_DEVICES=1 python train.py --experiments_dir ${experiment_name}_freeze_${lr} --encoder_name openai/clip-vit-large-patch14 --features_dir l14_features --mlp_path pic2word.pt --lr $lr
# mkdir "${experiment_name}_freeze_${lr}/results"
# for var in 8856 17712 26568 35424 44280 53136 61992 70848 79704 88560
# do
#   CUDA_VISIBLE_DEVICES=1 python infer.py --encoder_name "openai/clip-vit-large-patch14" --features_path "l14_features/val.hdf5" --model_path "${experiment_name}_freeze_${lr}/rag_7M_gpt2" --checkpoint_path checkpoint-${var}
#   CUDA_VISIBLE_DEVICES=1 python coco-caption/run_eval.py coco-caption/annotations/captions_valKarpathy.json "${experiment_name}_freeze_${lr}/rag_7M_gpt2/checkpoint-${var}/val_preds.json" > "${experiment_name}_freeze_${lr}/results/val_${var}.txt"

#   CUDA_VISIBLE_DEVICES=1 python infer.py --encoder_name "openai/clip-vit-large-patch14" --model_path "${experiment_name}_freeze_${lr}/rag_7M_gpt2" --checkpoint_path checkpoint-${var} --infer_test
#   CUDA_VISIBLE_DEVICES=1 python coco-caption/run_eval.py coco-caption/annotations/captions_testKarpathy.json "${experiment_name}_freeze_${lr}/rag_7M_gpt2/checkpoint-${var}/test_preds.json" > "${experiment_name}_freeze_${lr}/results/test_${var}.txt"
# done

# lr=1e-5
# CUDA_VISIBLE_DEVICES=1 python train.py --experiments_dir ${experiment_name}_freeze_${lr} --encoder_name openai/clip-vit-large-patch14 --features_dir l14_features --mlp_path pic2word.pt --lr $lr
# mkdir "${experiment_name}_freeze_${lr}/results"
# for var in 8856 17712 26568 35424 44280 53136 61992 70848 79704 88560
# do
#   CUDA_VISIBLE_DEVICES=1 python infer.py --encoder_name "openai/clip-vit-large-patch14" --features_path "l14_features/val.hdf5" --model_path "${experiment_name}_freeze_${lr}/rag_7M_gpt2" --checkpoint_path checkpoint-${var}
#   CUDA_VISIBLE_DEVICES=1 python coco-caption/run_eval.py coco-caption/annotations/captions_valKarpathy.json "${experiment_name}_freeze_${lr}/rag_7M_gpt2/checkpoint-${var}/val_preds.json" > "${experiment_name}_freeze_${lr}/results/val_${var}.txt"
#   CUDA_VISIBLE_DEVICES=1 python infer.py --encoder_name "openai/clip-vit-large-patch14" --model_path "${experiment_name}_freeze_${lr}/rag_7M_gpt2" --checkpoint_path checkpoint-${var} --infer_test
#   CUDA_VISIBLE_DEVICES=1 python coco-caption/run_eval.py coco-caption/annotations/captions_testKarpathy.json "${experiment_name}_freeze_${lr}/rag_7M_gpt2/checkpoint-${var}/test_preds.json" > "${experiment_name}_freeze_${lr}/results/test_${var}.txt"
# done


# d=14

# # CUDA_VISIBLE_DEVICES=0 python train.py --experiments_dir ${experiment_name}_freeze_${d} --encoder_name openai/clip-vit-large-patch14 --features_dir l14_features --mlp_path pic2word.pt --attention_size $d
# mkdir "${experiment_name}_freeze_${d}/results"
# for var in 70848 79704 88560
# do
#   CUDA_VISIBLE_DEVICES=0 python infer.py --encoder_name "openai/clip-vit-large-patch14" --features_path "l14_features/val.hdf5" --model_path "${experiment_name}_freeze_${d}/rag_14.0M_gpt2" --checkpoint_path checkpoint-${var}
#   CUDA_VISIBLE_DEVICES=0 python coco-caption/run_eval.py coco-caption/annotations/captions_valKarpathy.json "${experiment_name}_freeze_${d}/rag_14.0M_gpt2/checkpoint-${var}/val_preds.json" > "${experiment_name}_freeze_${d}/results/val_${var}.txt"

#   # CUDA_VISIBLE_DEVICES=0 python infer.py --encoder_name "openai/clip-vit-large-patch14" --model_path "${experiment_name}_freeze_${d}/rag_14.0M_gpt2" --checkpoint_path checkpoint-${var} --infer_test
#   # CUDA_VISIBLE_DEVICES=0 python coco-caption/run_eval.py coco-caption/annotations/captions_testKarpathy.json "${experiment_name}_freeze_${d}/rag_14.0M_gpt2/checkpoint-${var}/test_preds.json" > "${experiment_name}_freeze_${d}/results/test_${var}.txt"
# done


# # CUDA_VISIBLE_DEVICES=0 python train.py --experiments_dir ${experiment_name}_finetuning_${d} --encoder_name openai/clip-vit-large-patch14 --features_dir l14_features --train_mlp --mlp_path pic2word.pt --attention_size $d 
# mkdir "${experiment_name}_finetuning_${d}/results"
# for var in 26568 35424 44280 53136 61992 70848 79704 88560
# do
#   CUDA_VISIBLE_DEVICES=0 python infer.py --encoder_name "openai/clip-vit-large-patch14" --features_path "l14_features/val.hdf5" --model_path "${experiment_name}_finetuning_${d}/rag_14.0M_gpt2" --checkpoint_path checkpoint-${var}
#   CUDA_VISIBLE_DEVICES=0 python coco-caption/run_eval.py coco-caption/annotations/captions_valKarpathy.json "${experiment_name}_finetuning_${d}/rag_14.0M_gpt2/checkpoint-${var}/val_preds.json" > "${experiment_name}_finetuning_${d}/results/val_${var}.txt"

#   # CUDA_VISIBLE_DEVICES=0 python infer.py --encoder_name "openai/clip-vit-large-patch14" --model_path "${experiment_name}_finetuning_${d}/rag_14.0M_gpt2" --checkpoint_path checkpoint-${var} --infer_test
#   # CUDA_VISIBLE_DEVICES=0 python coco-caption/run_eval.py coco-caption/annotations/captions_testKarpathy.json "${experiment_name}_finetuning_${d}/rag_14.0M_gpt2/checkpoint-${var}/test_preds.json" > "${experiment_name}_finetuning_${d}/results/test_${var}.txt"
# done



# d=28
# # CUDA_VISIBLE_DEVICES=0 python train.py --experiments_dir ${experiment_name}_finetuning_${d} --encoder_name openai/clip-vit-large-patch14 --features_dir l14_features --train_mlp --mlp_path pic2word.pt --attention_size $d 
# mkdir "${experiment_name}_finetuning_${d}/results"
# for var in 8856 17712 26568 35424 44280 53136 61992 70848 79704 88560
# do
#   CUDA_VISIBLE_DEVICES=0 python infer.py --encoder_name "openai/clip-vit-large-patch14" --features_path "l14_features/val.hdf5" --model_path "${experiment_name}_finetuning_${d}/rag_28.0M_gpt2" --checkpoint_path checkpoint-${var}
#   CUDA_VISIBLE_DEVICES=0 python coco-caption/run_eval.py coco-caption/annotations/captions_valKarpathy.json "${experiment_name}_finetuning_${d}/rag_28.0M_gpt2/checkpoint-${var}/val_preds.json" > "${experiment_name}_finetuning_${d}/results/val_${var}.txt"

#   # CUDA_VISIBLE_DEVICES=0 python infer.py --encoder_name "openai/clip-vit-large-patch14" --model_path "${experiment_name}_finetuning_${d}/rag_28.0M_gpt2" --checkpoint_path checkpoint-${var} --infer_test
#   # CUDA_VISIBLE_DEVICES=0 python coco-caption/run_eval.py coco-caption/annotations/captions_testKarpathy.json "${experiment_name}_finetuning_${d}/rag_28.0M_gpt2/checkpoint-${var}/test_preds.json" > "${experiment_name}_finetuning_${d}/results/test_${var}.txt"
# done

# # CUDA_VISIBLE_DEVICES=0 python train.py --experiments_dir ${experiment_name}_freeze_${d} --encoder_name openai/clip-vit-large-patch14 --features_dir l14_features --mlp_path pic2word.pt --attention_size $d
# # mkdir "${experiment_name}_freeze_${d}/results"
# # for var in 8856 17712 26568 35424 44280 53136 61992 70848 79704 88560
# # do
# #   CUDA_VISIBLE_DEVICES=0 python infer.py --encoder_name "openai/clip-vit-large-patch14" --features_path "l14_features/val.hdf5" --model_path "${experiment_name}_freeze_${d}/rag_28.0M_gpt2" --checkpoint_path checkpoint-${var}
# #   CUDA_VISIBLE_DEVICES=0 python coco-caption/run_eval.py coco-caption/annotations/captions_valKarpathy.json "${experiment_name}_freeze_${d}/rag_28.0M_gpt2/checkpoint-${var}/val_preds.json" > "${experiment_name}_freeze_${d}/results/val_${var}.txt"

# #   # CUDA_VISIBLE_DEVICES=0 python infer.py --encoder_name "openai/clip-vit-large-patch14" --model_path "${experiment_name}_freeze_${d}/rag_28.0M_gpt2" --checkpoint_path checkpoint-${var} --infer_test
# #   # CUDA_VISIBLE_DEVICES=0 python coco-caption/run_eval.py coco-caption/annotations/captions_testKarpathy.json "${experiment_name}_freeze_${d}/rag_28.0M_gpt2/checkpoint-${var}/test_preds.json" > "${experiment_name}_freeze_${d}/results/test_${var}.txt"
# # done