import subprocess

def ask_fine_tuned_docter(query):
    command = [
        "mlx_lm.generate",
        "--model", "mlx-community/Ministral-8B-Instruct-2410-4bit",
        "--max-tokens", "500",
        "--adapter-path", "./fine_tuned_systems/adapters/fine_tuned_doctor/",
        "--prompt", query
    ]

    try:
        return subprocess.run(command, check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")
    except FileNotFoundError:
        print("Python or the specified module was not found. Ensure everything is installed correctly.")

def run_fine_tuned_doctor():
    print("Fine Tuned Model is now running... 👨‍⚕️\n")
    print("Type your question, or type 'quit' to exit.\n")

    while True:
        query = input("❓ > ")

        # Exit condition
        if query.strip().lower() in ["quit", "exit", "q"]:
            print("👋 Goodbye!")
            break

        # Empty input → ignore
        if not query.strip():
            continue

        answer = ask_fine_tuned_docter(query)
        print(answer)

        print("\n-----------------------------------\n")