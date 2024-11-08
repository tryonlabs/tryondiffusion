import argparse
import subprocess
import os
import pathlib

parser = argparse.ArgumentParser(description='run ootd')
parser.add_argument('--gpu_id', '-g', type=int, default=0, required=False)
parser.add_argument('--model_path', type=str, default="", required=True)
parser.add_argument('--cloth_path', type=str, default="", required=True)
parser.add_argument('--output_path', type=str, default="", required=True)
parser.add_argument('--model_type', type=str, default="hd", required=False)
parser.add_argument('--category', '-c', type=int, default=0, required=False)
parser.add_argument('--scale', type=float, default=2.0, required=False)
parser.add_argument('--step', type=int, default=20, required=False)
parser.add_argument('--sample', type=int, default=4, required=False)
parser.add_argument('--seed', type=int, default=-1, required=False)
args = parser.parse_args()

print(args)

if __name__ == '__main__':
    ootdiffusion_dir = "/home/ubuntu/ootdiffusion"

    command = (f"{os.path.join(str(pathlib.Path.home()), 'miniconda3/envs/ootdiffusion/bin/python')} "
               f"run.py --model_path {args.model_path} --cloth_path {args.cloth_path} "
               f"--output_path {args.output_path} --model_type {args.model_type} --category {args.category} "
               f"--image_scale {args.scale} --gpu_id {args.gpu_id} --n_samples {args.sample} --seed {args.seed} "
               f"--n_steps {args.step}")

    print("command:", command, command.split(" "))

    p = subprocess.Popen(command.split(" "), stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                         cwd=ootdiffusion_dir)
    out, err = p.communicate()
    print(out, err)


