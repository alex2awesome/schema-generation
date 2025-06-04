import torch
import torch.distributed as dist
import os

def main():
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank != -1:
        # Initialize DDP
        if not dist.is_initialized():
            torch.cuda.set_device(local_rank)
            # Changed backend to gloo for testing
            dist.init_process_group(backend='gloo', init_method='env://') 
            print(f"Rank {dist.get_rank()} initialized on device {torch.cuda.current_device()} with GLOO backend")
    else:
        print("Not in distributed mode (LOCAL_RANK not set). Running as standalone.")
        # For standalone, we can just use device 0 or let PyTorch pick
        local_rank = 0 # Default to device 0 if not distributed

    if not torch.cuda.is_available():
        print("CUDA is not available.")
        return

    # Ensure a device is selected even in non-DDP mode for consistency
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device) # Set the current device
    print(f"Currently operating on device: {torch.cuda.current_device()}")

    try:
        # Create a tensor on the assigned GPU
        x = torch.tensor([1.0, 2.0]).to(device)
        y = x * 2
        current_rank_str = f"Rank {dist.get_rank()}" if dist.is_initialized() else "Standalone"
        print(f"{current_rank_str}: Tensor operations successful on {device}. Result: {y}")

        if dist.is_initialized():
            # Simple all_reduce
            dist.all_reduce(y, op=dist.ReduceOp.SUM)
            # Ensure all processes have completed all_reduce before printing
            dist.barrier() 
            if dist.get_rank() == 0:
                print(f"Rank 0: All-reduced result after barrier (GLOO): {y}")
            # Clean up DDP group
            dist.destroy_process_group()
            print(f"Rank {local_rank}: DDP group destroyed (GLOO).")

    except RuntimeError as e:
        current_rank_str = f"Rank {dist.get_rank()}" if dist.is_initialized() else "Standalone"
        print(f"{current_rank_str}: CUDA Error on device {device} (GLOO): {e}")
        if dist.is_initialized():
            dist.destroy_process_group()
        raise
    except Exception as e:
        current_rank_str = f"Rank {dist.get_rank()}" if dist.is_initialized() else "Standalone"
        print(f"{current_rank_str}: Generic Error on device {device} (GLOO): {e}")
        if dist.is_initialized():
            dist.destroy_process_group()
        raise

if __name__ == "__main__":
    main() 