# import torch

# from distrifuser.pipelines import DistriSDXLPipeline
# from distrifuser.utils import DistriConfig


# import argparse

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model_id", default="stabilityai/stable-diffusion-xl-base-1.0", type=str, help="Path or Id to the pretrained model.")
#     args = parser.parse_args()
        
#     distri_config = DistriConfig(height=1024, width=1024, warmup_steps=4)

#     pipeline = DistriSDXLPipeline.from_pretrained(
#         distri_config=distri_config,
#         pretrained_model_name_or_path=args.model_id,
#         variant="fp16",
#         use_safetensors=True,
#     )

#     pipeline.set_progress_bar_config(disable=distri_config.rank != 0)
#     image = pipeline(
#         prompt="Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
#         generator=torch.Generator(device="cuda").manual_seed(233),
#     ).images[0]

#     if distri_config.rank == 0:
#         image.save("astronaut.png")
        
import argparse
import torch

from distrifuser.pipelines.sdxl import DistriSDXLPipeline
from distrifuser.utils import DistriConfig
from torch.profiler import profile, record_function, ProfilerActivity

import time

HAS_LONG_CTX_ATTN = False
try:
    from yunchang import set_seq_parallel_pg

    HAS_LONG_CTX_ATTN = True
except ImportError:
    print("yunchang not found")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        default="stabilityai/stable-diffusion-xl-base-1.0",
        type=str,
        help="Path to the pretrained model.",
    )
    parser.add_argument(
        "--parallelism",
        "-p",
        default="patch",
        type=str,
        choices=["patch", "naive_patch", "pipeline", "tensor"],
        help="Parallelism to use.",
    )
    parser.add_argument(
        "--use_seq_parallel_attn",
        action="store_true",
        default=False,
        help="Enable sequence parallel attention.",
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
    parser.add_argument("--num_micro_batch", type=int, default=2)
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="The height of image",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=256,
        help="The width of image",
    )
    parser.add_argument(
        "--no_use_resolution_binning",
        action="store_true",
    )
    parser.add_argument(
        "--ulysses_degree",
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
        "--output_type",
        type=str,
        default="latent",
        choices=["latent", "pil"],
        help="latent saves memory, pil will results a memory burst in vae",
    )
    parser.add_argument(
        "--attn_num",
        default=None,
        nargs='*',
        type=int
    )
    parser.add_argument(
        "--scheduler",
        "-s",
        default="dpmsolver_multistep",
        type=str,
        choices=["dpmsolver_multistep", "ddim"],
        help="Scheduler to use.",
    )
    

    args = parser.parse_args()

    # torch.backends.cudnn.benchmark=True
    torch.backends.cudnn.deterministic = True

    # for DiT the height and width are fixed according to the model
    distri_config = DistriConfig(
        height=args.height,
        width=args.width,
        warmup_steps=4,
        do_classifier_free_guidance=True,
        split_batch=False,
        parallelism=args.parallelism,
        mode=args.sync_mode,
        num_micro_batch=args.num_micro_batch,
        use_seq_parallel_attn=args.use_seq_parallel_attn,
        use_resolution_binning=not args.no_use_resolution_binning,
        use_cuda_graph=args.use_cuda_graph,
        attn_num=args.attn_num,
        scheduler=args.scheduler,
    )

    if distri_config.use_seq_parallel_attn and HAS_LONG_CTX_ATTN:
        ulysses_degree = args.ulysses_degree
        ring_degree = distri_config.world_size // ulysses_degree
        set_seq_parallel_pg(
            ulysses_degree,
            ring_degree,
            distri_config.rank,
            distri_config.world_size,
            use_ulysses_low=args.use_use_ulysses_low,
        )

    pipeline = DistriSDXLPipeline.from_pretrained(
        distri_config=distri_config,
        pretrained_model_name_or_path=args.model_id,
        # variant="fp16",
        # use_safetensors=True,
    )

    pipeline.set_progress_bar_config(disable=distri_config.rank != 0)
    # warmup
    output = pipeline(
        prompt="An astronaut riding a green horse",
        generator=torch.Generator(device="cuda").manual_seed(42),
        output_type=args.output_type,
    )

    torch.cuda.reset_peak_memory_stats()
    if args.parallelism == "pipeline":
        case_name = f"{args.parallelism}_hw_{args.height}_sync_{args.sync_mode}_sp_{args.use_seq_parallel_attn}_u{args.ulysses_degree}_w{distri_config.world_size}_mb{args.num_micro_batch}"
    else:
        case_name = f"{args.parallelism}_hw_{args.height}_sync_{args.sync_mode}_sp_{args.use_seq_parallel_attn}_u{args.ulysses_degree}_w{distri_config.world_size}"
    if args.use_profiler:
        start_time = time.time()
        with profile(
            activities=[ProfilerActivity.CUDA],
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                f"./profile/{case_name}"
            ),
            profile_memory=True,
            with_stack=True,
            record_shapes=True,
        ) as prof:
            output = pipeline(
                prompt="An astronaut riding a green horse",
                generator=torch.Generator(device="cuda").manual_seed(42),
                num_inference_steps=args.num_inference_steps,
                output_type="latent",
            )
        # if distri_config.rank == 0:
        #     prof.export_memory_timeline(
        #         f"{distri_config.mode}_{args.height}_{distri_config.world_size}_mem.html"
        #     )
        end_time = time.time()
    else:
        MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT = 100000
        torch.cuda.memory._record_memory_history(
            max_entries=MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT
        )
        start_time = time.time()
        output = pipeline(
            prompt="An astronaut riding a green horse",
            generator=torch.Generator(device="cuda").manual_seed(42),
            num_inference_steps=args.num_inference_steps,
            output_type=args.output_type,
        )

        end_time = time.time()
        torch.cuda.memory._dump_snapshot(
            f"{distri_config.mode}_{distri_config.world_size}.pickle"
        )
        torch.cuda.memory._record_memory_history(enabled=None)

    elapsed_time = end_time - start_time

    peak_memory = torch.cuda.max_memory_allocated(device="cuda")

    if distri_config.rank == 0:

        print(
            f"{case_name} epoch time: {elapsed_time:.2f} sec, memory: {peak_memory/1e9} GB"
        )
        if args.output_type == "pil":
            print(f"save images to ./results/astronaut_{case_name}.png")
            output.images[0].save(f"./results/astronaut_{case_name}.png")


if __name__ == "__main__":
    main()
