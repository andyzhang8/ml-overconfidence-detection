import torch
import argparse
import os

def merge_ensemble_checkpoints(checkpoint_prefix, output_path):
    print(f"Loading: {checkpoint_prefix}", flush=True)
    
    merged_state_dicts = {}
    
    for i in range(5):
        checkpoint_path = f"{checkpoint_prefix}_{i}.pth"
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found.")
        
        print(f"Loading {checkpoint_path}...", flush=True)
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        
        merged_state_dicts[f"model_{i}"] = state_dict

    print(f"saving to {output_path}", flush=True)
    torch.save(merged_state_dicts, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge Deep Ensemble model checkpoints into a single file.")
    parser.add_argument("--prefix", type=str, required=True, help="Checkpoint file prefix (without _0.pth, _1.pth, etc.)")
    parser.add_argument("--output", type=str, required=True, help="Output merged checkpoint file path")

    args = parser.parse_args()

    merge_ensemble_checkpoints(args.prefix, args.output)
