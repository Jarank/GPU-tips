"""Compare speed of different models with batch size 12"""
import torch
import psutil
import datetime
import os
import pandas as pd
import argparse
from torch.utils.data import DataLoader
import json
from utils import MODEL_LIST, RandomDataset, train, inference

torch.backends.cudnn.benchmark = True
# https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
# This flag allows you to enable the inbuilt cudnn auto-tuner to find the best algorithm to use for your hardware.
# If you check it using the profile tool, the cnn method such as winograd, fft, etc. is used for the first iteration and the best operation is selected for the device.


precisions = ["float", "half", "double"]
# For post-volta architectures, there is a possibility to use tensor-core at half precision.
# Due to the gradient overflow problem, apex is recommended for practical use.
device_name = str(torch.cuda.get_device_name(0))
# Training settings
parser = argparse.ArgumentParser(description="PyTorch Benchmarking")
parser.add_argument("--WARM_UP", "-w", type=int, default=5, required=False, help="Num of warm up")
parser.add_argument("--NUM_TEST", "-n", type=int, default=50, required=False, help="Num of Test")
parser.add_argument(
    "--BATCH_SIZE", "-b", type=int, default=12, required=False, help="Num of batch size"
)
parser.add_argument(
    "--NUM_CLASSES", "-c", type=int, default=1000, required=False, help="Num of class"
)
parser.add_argument("--NUM_GPU", "-g", type=int, default=1, required=False, help="Num of gpus")
parser.add_argument(
    "--folder",
    "-f",
    type=str,
    default="result",
    required=False,
    help="folder to save results",
)
args = parser.parse_args()
args.BATCH_SIZE *= args.NUM_GPU


if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    folder_name = args.folder
    device_name = f"{device_name}_{args.NUM_GPU}_gpus_"
    system_configs = f"{psutil.cpu_freq()}\n\
                    cpu_count: {psutil.cpu_count()}\n\
                    memory_available: {psutil.virtual_memory().available}"
    gpu_configs = [
        torch.cuda.device_count(),
        torch.version.cuda,
        torch.backends.cudnn.version(),
        torch.cuda.get_device_name(0),
    ]
    gpu_configs = list(map(str, gpu_configs))
    CONFIGS = [
        "Number of GPUs on current device : ",
        "CUDA Version : ",
        "Cudnn Version : ",
        "Device Name : ",
    ]
    random_loader = DataLoader(
        dataset=RandomDataset(args.BATCH_SIZE * (args.WARM_UP + args.NUM_TEST)),
        batch_size=args.BATCH_SIZE,
        shuffle=False,
        num_workers=8,
    )
    os.makedirs(folder_name, exist_ok=True)
    with open(os.path.join(folder_name, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    now = datetime.datetime.now()

    start_time = now.strftime("%Y/%m/%d %H:%M:%S")

    print(f"benchmark start : {start_time}")
    for idx, value in enumerate(zip(CONFIGS, gpu_configs)):
        gpu_configs[idx] = "".join(value)
        print(gpu_configs[idx])
    print(system_configs)

    with open(os.path.join(folder_name, "system_info.txt"), "w") as f:
        f.writelines(f"benchmark start : {start_time}\n")
        f.writelines("system_configs\n\n")
        f.writelines(system_configs)
        f.writelines("\ngpu_configs\n\n")
        f.writelines(s + "\n" for s in gpu_configs)

    for precision in precisions:
        train_result = train(precision, device, random_loader, args)
        train_result_df = pd.DataFrame(train_result)
        path = f"{folder_name}/{device_name}_{precision}_model_train_benchmark.csv"
        train_result_df.to_csv(path, index=False)

        inference_result = inference(precision, device, random_loader, args)
        inference_result_df = pd.DataFrame(inference_result)
        path = f"{folder_name}/{device_name}_{precision}_model_inference_benchmark.csv"
        inference_result_df.to_csv(path, index=False)

    now = datetime.datetime.now()

    end_time = now.strftime("%Y/%m/%d %H:%M:%S")
    print(f"benchmark end : {end_time}")
    with open(os.path.join(folder_name, "system_info.txt"), "a") as f:
        f.writelines(f"benchmark end : {end_time}\n")
