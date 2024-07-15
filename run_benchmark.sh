pip install flash_attn
NPROC_PER_NODE=8 xtuner train /data/home/xiaotong/xtuner/xtuner/configs/sft_exp/llama3_8b_full_alpaca_fa2.py --deepspeed deepspeed_zero3
NPROC_PER_NODE=8 xtuner train /data/home/xiaotong/xtuner/xtuner/configs/sft_exp/phi3_mini_4k_instruct_full_alpaca_fa2.py --deepspeed deepspeed_zero3
pip uninstall flash_attn -y
NPROC_PER_NODE=8 xtuner train /data/home/xiaotong/xtuner/xtuner/configs/sft_exp/llama3_8b_full_alpaca_nofa2.py --deepspeed deepspeed_zero3
NPROC_PER_NODE=8 xtuner train /data/home/xiaotong/xtuner/xtuner/configs/sft_exp/phi3_mini_4k_instruct_full_alpaca_nofa2.py --deepspeed deepspeed_zero3