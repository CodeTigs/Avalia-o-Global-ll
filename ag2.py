import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score


url = "https://raw.githubusercontent.com/marcelovca90-inatel/AG2/refs/heads/main/iris.csv"
df = pd.read_csv(url)

print("Dados carregados com sucesso!\n")
print(df.head())

mapping = {
    "Iris-setosa": 1,
    "Iris-versicolor": 2,
    "Iris-virginica": 3
}

df["species"] = df["species"].replace(mapping).astype("int64")

print("\nEspécies convertidas para números:")
print(df.head())

# todas as colunas menos 'species' são atributos
feature_cols = [col for col in df.columns if col != "species"]

X = df[feature_cols]
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    shuffle=True,
    random_state=42
)

# Treinar o modelo (Decision Tree)
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n===== AVALIAÇÃO DO MODELO =====")
print("Acurácia:", accuracy_score(y_test, y_pred))
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

def classificar_usuario(model):
    while True:
        print("\n===== CLASSIFICAÇÃO MANUAL =====")
        print("(Use ponto para decimais, ex: 5.1)\n")

        # leitura dos 4 valores
        try:
            sepal_length = float(input("Sepal Length (cm): "))
            sepal_width  = float(input("Sepal Width  (cm): "))
            petal_length = float(input("Petal Length (cm): "))
            petal_width  = float(input("Petal Width  (cm): "))
        except ValueError:
            print("Erro: insira somente números (use ponto como separador decimal).")
            continue  # volta para o início do while

        # monta o DataFrame de entrada usando os MESMOS nomes de colunas de treino
        valores = [sepal_length, sepal_width, petal_length, petal_width]
        entrada_dict = dict(zip(feature_cols, valores))
        entrada = pd.DataFrame([entrada_dict])

        # faz a previsão
        pred = model.predict(entrada)[0]

        especie = {
            1: "Iris-setosa",
            2: "Iris-versicolor",
            3: "Iris-virginica"
        }[pred]

        print(f"\n→ O modelo classificou como: **{especie}**")

        # pergunta se quer continuar
        opcao = input("\nDeseja classificar outra flor? (s/n): ").strip().lower()
        if opcao != "s":
            print("\nEncerrando... Obrigado por usar o classificador Iris!")
            break

# Chamar função
classificar_usuario(model)

