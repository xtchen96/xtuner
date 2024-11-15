## Experiment: fine-tune LLama3-8b with 8K length dataset

### Prerequisite

install xtuner. need a Python environment with cuda, torch and llm related libraries installed (transformers, accelerate, etc.). Need to login to huggingface to download llama3 models.

pytorch 2.5 will cause `'Adafactor is already registered in optimizer at torch.optim'` error. solution is to use pytorch 2.4.

Another error `error: 'libGL.so.1: cannot open shared object file: No such file or directory'`. solution: `sudo apt update && sudo apt install libgl1`

```
pip install -e '.[all]'
```

### Dataset preparation

```
cd llama3-8b-lorasft-8k
```

Option 1: select any available dataset on HF, e.g. default one is `'tatsu-lab/alpaca'`. The script is `llama3_8b_default_alpaca_dataset.py`

Option 2:
create 8k length dataset from a short test dataset (use `alpaca_en_demo.json` now). Change the data path before running:

```
python create_long_alpaca.py 
```

which repeats the content in 'output' field until it reaches 8k length. Also set `max_length = 8192`. The script is `llama3_8b_customized_dataset.py`

### Fine-tuning script

run the lora-sft script 

```
NPROC_PER_NODE=4 xtuner train llama3_8b_default_alpaca_dataset.py --deepspeed deepspeed_zero3_offload
```

The key parameter is to set `sequence_parallel_size = 4`, which divides the 8k length input data equally into 4 pieces, so that the memory fits within 24GiB on rtx 3090ti/4090.

upon successful running, the cli output should be like

```
11/15 18:22:07 - mmengine - INFO - Iter(train) [  10/1545]  lr: 4.0002e-05  eta: 4:17:00  time: 10.0461  data_time: 0.0105  memory: 7529  loss: 1.8848
11/15 18:23:42 - mmengine - INFO - Iter(train) [  20/1545]  lr: 8.4446e-05  eta: 4:07:45  time: 9.4501  data_time: 0.0106  memory: 7528  loss: 1.8937
11/15 18:25:16 - mmengine - INFO - Iter(train) [  30/1545]  lr: 1.2889e-04  eta: 4:03:37  time: 9.4491  data_time: 0.0105  memory: 7528  loss: 1.8641
11/15 18:26:51 - mmengine - INFO - Iter(train) [  40/1545]  lr: 1.7333e-04  eta: 4:00:45  time: 9.4475  data_time: 0.0106  memory: 7528  loss: 1.8745
```

The memory usage should be around 16GiB across 4 gpus.