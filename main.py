import inquirer
from memory_systems.conversator import run_conversator
from memory_systems.roman_expert import run_roman_expert
from rag_systems.eredivisie_rag import run_eredivisie_rag
from fine_tuned_systems.fine_tuned_doctor import run_fine_tuned_doctor
from fine_tune_scripts.fine_tuned_doctor_tuner import run_fine_tuned_doctor_training
from rag_systems.nederland_expert import run_nederland_expert

if __name__ == "__main__":
    print("Welcome to the Local LLM MLX Test Interface! 🕹️")
    print("You can experiment here with different type of systems built using Local LLMs and MLX. 🛠️")
    print("------------------------------------------------------------------------------------------\n")
    print("Firstly we wanna know what you want to do?")
    questions = [
        inquirer.List('action',
                      message="What do you want to do?",
                      choices=['Fine tune a model', 'Test a system'],
                      ),
    ]
    answers = inquirer.prompt(questions)
    if answers['action'] == 'Fine tune a model':
        print("Choose a tuner to run from the options below:")
        questions = [
            inquirer.List('tuner',
                          message="What model do you want to tune",
                          choices=['Fine-tuned-Doctor'],
                          ),
        ]
        answers = inquirer.prompt(questions)

        if answers['tuner'] == 'Fine-tuned-Doctor':
            print("👨‍⚕️🦾 Running Fine-tuned-Doctor training...")
            run_fine_tuned_doctor_training()

    elif answers['action'] == 'Test a system':
        print("Choose a system to test from the options below:")
        while True:
            questions = [
                inquirer.List('model',
                              message="What model do you want to explore?",
                              choices=['Fine-tuned-Doctor', 'Eredivisie RAG', 'Nederland Expert RAG', 'Conversator', 'Roman Expert'],
                              ),
            ]
            answers = inquirer.prompt(questions)

            if answers['model'] == 'Fine-tuned-Doctor':
                run_fine_tuned_doctor()

            elif answers['model'] == 'Eredivisie RAG':
                run_eredivisie_rag()

            elif answers['model'] == 'Nederland Expert RAG':
                run_nederland_expert()

            elif answers['model'] == 'Conversator':
                run_conversator()

            elif answers['model'] == 'Roman Expert':
                run_roman_expert()

            if not inquirer.confirm("Do you want to try another system?", default=True):
                break

    print("Thank you for using the Local LLM MLX Test Interface. Goodbye! 👋")