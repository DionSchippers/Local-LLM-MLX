import inquirer
from rag_systems.eredivisie_rag import run_eredivisie_rag

if __name__ == "__main__":
    print("Welcome to the Local LLM MLX Test Interface! 🕹️")
    print("You can experiment here with different type of systems built using Local LLMs and MLX. 🛠️")
    print("Choose a system to test from the options below:")
    while True:
        questions = [
            inquirer.List('model',
                          message="What model do you want to explore?",
                          choices=['Fine-tuned-Doctor', 'Eredivisie RAG'],
                          ),
        ]
        answers = inquirer.prompt(questions)

        if answers['model'] == 'Fine-tuned-Doctor':
            from fine_tuned_doctor import run_fine_tuned_doctor
            run_fine_tuned_doctor()

        elif answers['model'] == 'Eredivisie RAG':
            run_eredivisie_rag()

        if not inquirer.confirm("Do you want to try another system?", default=True):
            break

    print("Thank you for using the Local LLM MLX Test Interface. Goodbye! 👋")