import argparse
import torch

from pipefuser.pipelines.pixartalpha import DistriPixArtAlphaPipeline
from pipefuser.utils import DistriConfig
from torch.profiler import profile, record_function, ProfilerActivity
from pipefuser.modules.conv.conv_chunk.chunk_conv2d import PatchConv2d

import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        default="PixArt-alpha/PixArt-XL-2-1024-MS",
        type=str,
        help="Path to the pretrained model.",
    )
    parser.add_argument(
        "--parallelism",
        "-p",
        default="patch",
        type=str,
        choices=["patch", "naive_patch", "pipefusion", "tensor", "sequence"],
        help="Parallelism to use.",
    )
    parser.add_argument(
        "--sync_mode",
        type=str,
        default="corrected_async_gn",
        choices=[
            "separate_gn",
            "async_gn",
            "corrected_async_gn",
            "sync_gn",
            "full_sync",
            "no_sync",
        ],
        help="Different GroupNorm synchronization modes",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--pp_num_patch", type=int, default=2, help="patch number in pipefusion."
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="The height of image",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="The width of image",
    )
    parser.add_argument(
        "--no_use_resolution_binning",
        action="store_true",
    )
    parser.add_argument(
        "--ulysses_degree",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--pipefusion_warmup_step",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--use_use_ulysses_low",
        action="store_true",
    )
    parser.add_argument(
        "--use_profiler",
        action="store_true",
    )
    parser.add_argument(
        "--use_cuda_graph",
        action="store_true",
    )
    parser.add_argument(
        "--use_parallel_vae",
        action="store_true",
    )
    parser.add_argument(
        "--output_type",
        type=str,
        default="latent",
        choices=["latent", "pil"],
        help="latent saves memory, pil will results a memory burst in vae",
    )
    parser.add_argument("--attn_num", default=None, nargs="*", type=int)
    parser.add_argument(
        "--scheduler",
        "-s",
        default="dpm-solver",
        type=str,
        choices=["dpm-solver", "ddim"],
        help="Scheduler to use.",
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default="An astronaut riding a green horse",
    )
    parser.add_argument("--output_file", type=str, default=None)

    args = parser.parse_args()

    # torch.backends.cudnn.benchmark=True
    torch.backends.cudnn.deterministic = True

    enable_parallel_vae = args.use_parallel_vae

    # for DiT the height and width are fixed according to the model
    distri_config = DistriConfig(
        height=args.height,
        width=args.width,
        warmup_steps=args.pipefusion_warmup_step,
        do_classifier_free_guidance=True,
        split_batch=False,
        parallelism=args.parallelism,
        mode=args.sync_mode,
        pp_num_patch=args.pp_num_patch,
        use_resolution_binning=not args.no_use_resolution_binning,
        use_cuda_graph=args.use_cuda_graph,
        attn_num=args.attn_num,
        scheduler=args.scheduler,
        ulysses_degree=args.ulysses_degree,
        batch_size=2
    )

    pipeline = DistriPixArtAlphaPipeline.from_pretrained(
        distri_config=distri_config,
        pretrained_model_name_or_path=args.model_id,
        enable_parallel_vae=enable_parallel_vae,
        use_profiler=args.use_profiler,
    )

    pipeline.set_progress_bar_config(disable=distri_config.rank != 0)
    output = pipeline(
        prompt=["An astronaut riding a green horse", "An astronaut riding a red horse"],
        generator=torch.Generator(device="cuda").manual_seed(42),
        output_type=args.output_type,
        num_inference_steps=args.num_inference_steps
    )


    if args.parallelism == "pipefusion":
        case_name = f"{args.parallelism}_hw_{args.height}_sync_{args.sync_mode}_u{args.ulysses_degree}_w{distri_config.world_size}_mb{args.pp_num_patch}_warm{args.pipefusion_warmup_step}"
    else:
        case_name = f"{args.parallelism}_hw_{args.height}_sync_{args.sync_mode}_u{args.ulysses_degree}_w{distri_config.world_size}"
    if args.output_file:
        case_name = args.output_file + "_" + case_name
    if enable_parallel_vae:
        case_name += "_patchvae"

    if distri_config.rank == 0:
        if args.output_type == "pil":
            for i in range(len(output.images)):
                print(f"save images to ./results/{case_name}_{i}.png")
                output.images[i].save(f"./results/{case_name}_{i}.png")

if __name__ == "__main__":
    main()
