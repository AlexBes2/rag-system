import requests

url = "http://localhost:11434/api/generate"
model = "llama3"

while True:
    question = input("Введите вопрос: ")

    if question.lower() in ["exit", "quit"]:
        break

    data = {
        "model": model,
        "prompt": question,
        "stream": False
    }

    response = requests.post(url, json=data)

    if response.status_code == 200:
        answer = response.json()["response"]
        print("\nОтвет модели:\n")
        print(answer)
        print("\n" + "-"*50 + "\n")
    else:
        print("Ошибка:", response.text)