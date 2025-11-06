import subprocess

def run_fine_tuned_doctor_training():
    command = [
        "mlx_lm.lora",
        "--model", "mlx-community/Ministral-8B-Instruct-2410-4bit",
        "--data", "./fine_tuned_systems/train_data/fine_tuned_doctor",
        "--train",
        "--fine-tune-type", "lora",
        "--batch-size", "4",
        "--num-layers", "16",
        "--iters", "500",
        "--adapter-path", "./fine_tuned_systems/adapters/fine_tuned_doctor"
    ]

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")
    except FileNotFoundError:
        print("Python or the specified module was not found. Ensure everything is installed correctly.")