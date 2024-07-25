pip install flash_attn
xtuner convert pth_to_hf /data/home/xiaotong/xtuner/xtuner/configs/sft_exp/llama3_8b_full_alpaca_fa2.py \
    work_dirs/llama3_8b_full_alpaca_fa2/iter_795.pth \
    work_dirs/llama3_8b_full_alpaca_fa2/iter_795_hf
xtuner convert pth_to_hf /data/home/xiaotong/xtuner/xtuner/configs/sft_exp/phi3_mini_4k_instruct_full_alpaca_fa2.py \
    work_dirs/phi3_mini_4k_instruct_full_alpaca_fa2/iter_873.pth \
    work_dirs/phi3_mini_4k_instruct_full_alpaca_fa2/iter_873_hf

pip uninstall flash_attn -y
xtuner convert pth_to_hf /data/home/xiaotong/xtuner/xtuner/configs/sft_exp/llama3_8b_full_alpaca_nofa2.py \
    work_dirs/llama3_8b_full_alpaca_nofa2/iter_795.pth \
    work_dirs/llama3_8b_full_alpaca_nofa2/iter_795_hf
xtuner convert pth_to_hf /data/home/xiaotong/xtuner/xtuner/configs/sft_exp/phi3_mini_4k_instruct_full_alpaca_nofa2.py \
    work_dirs/phi3_mini_4k_instruct_full_alpaca_nofa2/iter_873.pth \
    work_dirs/phi3_mini_4k_instruct_full_alpaca_nofa2/iter_873_hf