import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("dark_background")
import seaborn as sns

# Config
sns.set_theme(style="darkgrid")
plt.rcParams["figure.facecolor"] = "black"

# Cargar datos
df = pd.read_csv("churn.csv")

# ---------------------------
# 1. Overview general
# ---------------------------
print("Shape:", df.shape)
print("\nTipos de datos:\n", df.dtypes)
print("\nMissing values:\n", df.isnull().sum())
print("\nDuplicados:", df.duplicated().sum())

# ---------------------------
# 2. Target analysis
# ---------------------------
churn_rate = df["Exited"].mean()
print(f"\nChurn Rate: {churn_rate:.2%}")

plt.rcParams["figure.facecolor"] = "black"
plt.rcParams["axes.facecolor"] = "black"
plt.rcParams["savefig.facecolor"] = "black"
plt.rcParams["text.color"] = "white"
plt.rcParams["axes.labelcolor"] = "white"
plt.rcParams["xtick.color"] = "white"
plt.rcParams["ytick.color"] = "white"
plt.rcParams["axes.edgecolor"] = "white"

sns.countplot(x="Exited", data=df)
plt.title("Distribución de Churn (Exited)")
plt.show()

# ---------------------------
# 3. Variables numéricas
# ---------------------------
num_cols = ["CreditScore", "Age", "Balance", "EstimatedSalary"]

df[num_cols].hist(figsize=(10,8))
plt.suptitle("Distribución de Variables Numéricas")
plt.show()

# Boxplot vs churn
for col in num_cols:
    plt.figure()
    sns.boxplot(x="Exited", y=col, data=df)
    plt.title(f"{col} vs Churn")
    plt.show()

# ---------------------------
# 4. Variables categóricas
# ---------------------------
cat_cols = ["Geography", "Gender", "NumOfProducts", "IsActiveMember"]

for col in cat_cols:
    plt.figure()
    sns.countplot(x=col, hue="Exited", data=df)
    plt.title(f"{col} vs Churn")
    plt.xticks(rotation=45)
    plt.show()

# ---------------------------
# 5. Correlación
# ---------------------------
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Matriz de correlación")
plt.show()


# Copia de seguridad
df_model = df.copy()

# ---------------------------
# 1. Eliminar variables irrelevantes
# ---------------------------
df_model.drop(["RowNumber", "CustomerId", "Surname"], axis=1, inplace=True)

# ---------------------------
# 2. Feature Engineering (NEGOCIO)
# ---------------------------

# Edad en grupos (comportamiento distinto por segmento)
df_model["AgeGroup"] = pd.cut(
    df_model["Age"],
    bins=[18, 30, 45, 60, 100],
    labels=["Young", "Adult", "Mature", "Senior"]
)

# Cliente de alto valor (balance alto)
df_model["HighValueCustomer"] = (df_model["Balance"] > df_model["Balance"].median()).astype(int)

# Engagement (combinación clave)
df_model["EngagementScore"] = (
    df_model["NumOfProducts"] * df_model["IsActiveMember"]
)

# Ratio balance / salario (capacidad financiera)
df_model["BalanceSalaryRatio"] = df_model["Balance"] / (df_model["EstimatedSalary"] + 1)

# Flag clientes "dormidos"
df_model["InactiveHighBalance"] = (
    (df_model["IsActiveMember"] == 0) & (df_model["Balance"] > 0)
).astype(int)

# ---------------------------
# 3. Encoding
# ---------------------------

df_model = pd.get_dummies(df_model, drop_first=True)

# ---------------------------
# 4. Separar X e y
# ---------------------------

X = df_model.drop("Exited", axis=1)
y = df_model["Exited"]

print("Shape final:", X.shape)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# ---------------------------
# 1. Split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------
# 2. Escalado (para modelos lineales)
# ---------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------
# 3. Modelos
# ---------------------------

models = {
    "Logistic": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
    "XGBoost": xgb.XGBClassifier(eval_metric="logloss", random_state=42)
}

results = {}

for name, model in models.items():
    
    if name == "Logistic":
        model.fit(X_train_scaled, y_train)
        y_proba = model.predict_proba(X_test_scaled)[:,1]
    else:
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)[:,1]
    
    auc = roc_auc_score(y_test, y_proba)
    results[name] = auc
    
    print(f"\n{name} - AUC: {auc:.4f}")
    
    # Clasificación base (threshold 0.5)
    y_pred = (y_proba >= 0.5).astype(int)
    
    print(classification_report(y_test, y_pred))
    print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred))

# ---------------------------
# 4. Comparación final
# ---------------------------
print("\nComparación de modelos (AUC):")
for k, v in results.items():
    print(f"{k}: {v:.4f}")
    

def calculate_profit(y_true, y_pred, benefit_tp=100, cost_fp=-20):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    
    return (tp * benefit_tp) + (fp * cost_fp)

thresholds = np.linspace(0, 1, 100)
profits = []

# Usamos probabilidades del mejor modelo (ej: XGBoost)
y_proba_best = models["XGBoost"].predict_proba(X_test)[:,1]

for t in thresholds:
    y_pred = (y_proba_best >= t).astype(int)
    profit = calculate_profit(y_test.values, y_pred)
    profits.append(profit)

# Mejor threshold
best_threshold = thresholds[np.argmax(profits)]
best_profit = max(profits)

print(f"Best Threshold: {best_threshold:.2f}")
print(f"Max Profit: {best_profit}")

plt.plot(thresholds, profits)
plt.xlabel("Threshold")
plt.ylabel("Profit")
plt.title("Profit vs Threshold")
plt.axvline(best_threshold, color='red', linestyle='--', label=f'Best Threshold: {best_threshold:.2f}')
plt.legend()
plt.show()


plt.plot(thresholds, profits)
plt.xlabel("Threshold")
plt.ylabel("Profit")
plt.title("Optimización de Profit")
plt.axvline(best_threshold, linestyle="--")
plt.show()

import mlflow
import mlflow.sklearn

mlflow.set_experiment("Churn_Optimization_Project")

import mlflow.xgboost

with mlflow.start_run():

    model = xgb.XGBClassifier(eval_metric="logloss", random_state=42)
    model.fit(X_train, y_train)

    # Probabilidades
    y_proba = model.predict_proba(X_test)[:,1]

    # AUC
    auc = roc_auc_score(y_test, y_proba)

    # Profit optimization
    thresholds = np.linspace(0, 1, 100)
    profits = []

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        profit = calculate_profit(y_test.values, y_pred)
        profits.append(profit)

    best_threshold = thresholds[np.argmax(profits)]
    best_profit = max(profits)

    # ---------------------------
    # LOGS (esto es lo importante)
    # ---------------------------

    mlflow.log_param("model", "XGBoost")

    mlflow.log_metric("AUC", auc)
    mlflow.log_metric("best_threshold", best_threshold)
    mlflow.log_metric("max_profit", best_profit)

    # Guardar modelo
    mlflow.xgboost.log_model(model, "model")

    print("Run guardada en MLflow")
    
import joblib

joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(X.columns.tolist(), "features.pkl")

from fastapi import FastAPI
import joblib
import numpy as np
import pandas as pd

app = FastAPI(title="Churn Prediction API")

# Cargar artefactos
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

@app.get("/")
def home():
    return {"message": "Churn API funcionando"}

@app.post("/predict")
def predict(data: dict):
    
    # Convertir input a DataFrame
    df = pd.DataFrame([data])
    
    # Asegurar columnas correctas
    df = df.reindex(columns=features, fill_value=0)
    
    # Escalar (si aplica)
    try:
        df_scaled = scaler.transform(df)
        proba = model.predict_proba(df_scaled)[:,1][0]
    except:
        proba = model.predict_proba(df)[:,1][0]
    
    prediction = int(proba >= 0.5)
    
    return {
        "churn_probability": float(proba),
        "prediction": prediction
    }

import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import xgboost as xgb

# ---------------------------
# Load data
# ---------------------------
df = pd.read_csv("churn.csv")

df.drop(["RowNumber", "CustomerId", "Surname"], axis=1, inplace=True)

# ---------------------------
# Feature Engineering
# ---------------------------
df["AgeGroup"] = pd.cut(df["Age"], bins=[18,30,45,60,100], labels=[0,1,2,3])
df["HighValueCustomer"] = (df["Balance"] > df["Balance"].median()).astype(int)
df["EngagementScore"] = df["NumOfProducts"] * df["IsActiveMember"]
df["BalanceSalaryRatio"] = df["Balance"] / (df["EstimatedSalary"] + 1)
df["InactiveHighBalance"] = ((df["IsActiveMember"] == 0) & (df["Balance"] > 0)).astype(int)

df = pd.get_dummies(df, drop_first=True)

X = df.drop("Exited", axis=1)
y = df["Exited"]

# ---------------------------
# Split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# ---------------------------
# Profit function
# ---------------------------
def calculate_profit(y_true, y_pred, benefit_tp=100, cost_fp=-20):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return (tp * benefit_tp) + (fp * cost_fp)

# ---------------------------
# MLflow
# ---------------------------
mlflow.set_experiment("Churn_Optimization_Project")

with mlflow.start_run():

    model = xgb.XGBClassifier(eval_metric="logloss", random_state=42)
    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, y_proba)

    # Threshold optimization
    thresholds = np.linspace(0, 1, 100)
    profits = []

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        profits.append(calculate_profit(y_test.values, y_pred))

    best_threshold = thresholds[np.argmax(profits)]
    best_profit = max(profits)

    # Logs
    mlflow.log_metric("AUC", auc)
    mlflow.log_metric("best_threshold", best_threshold)
    mlflow.log_metric("max_profit", best_profit)

    mlflow.xgboost.log_model(model, "model")

# Save artifacts
joblib.dump(model, "model.pkl")
joblib.dump(X.columns.tolist(), "features.pkl")

print("Training completo + modelo guardado")

from fastapi import FastAPI


app = FastAPI(title="Churn Prediction API")

model = joblib.load("model.pkl")
features = joblib.load("features.pkl")

# threshold óptimo (podrías cargarlo desde MLflow en versión pro)
BEST_THRESHOLD = 0.27

@app.get("/")
def home():
    return {"message": "API funcionando"}

@app.post("/predict")
def predict(data: dict):

    df = pd.DataFrame([data])
    df = df.reindex(columns=features, fill_value=0)

    proba = model.predict_proba(df)[:,1][0]
    prediction = int(proba >= BEST_THRESHOLD)

    return {
        "churn_probability": float(proba),
        "prediction": prediction,
        "decision_threshold": BEST_THRESHOLD
    }
    
def generate_explanation(row):
    
    reasons = []

    if row["Age"] > 50:
        reasons.append("edad elevada")

    if row["IsActiveMember"] == 0:
        reasons.append("baja actividad")

    if row["NumOfProducts"] == 1:
        reasons.append("pocos productos contratados")

    if row["Balance"] > 100000:
        reasons.append("alto saldo en cuenta")

    if not reasons:
        return "Cliente con riesgo moderado sin factores dominantes"

    return "Cliente con riesgo de churn debido a: " + ", ".join(reasons)

def generate_retention_message(row):

    if row["IsActiveMember"] == 0:
        return "Te ofrecemos beneficios exclusivos si volvés a usar tu cuenta."

    if row["NumOfProducts"] == 1:
        return "Descubrí nuevos productos diseñados para vos con beneficios especiales."

    if row["Balance"] > 100000:
        return "Tenemos asesoramiento personalizado para ayudarte a gestionar mejor tu dinero."

    return "Queremos seguir acompañándote con mejores beneficios."

@app.post("/predict")
def predict(data: dict):

    df = pd.DataFrame([data])
    df = df.reindex(columns=features, fill_value=0)

    proba = model.predict_proba(df)[:,1][0]
    prediction = int(proba >= BEST_THRESHOLD)

    explanation = generate_explanation(data)
    message = generate_retention_message(data)

    return {
        "churn_probability": float(proba),
        "prediction": prediction,
        "explanation": explanation,
        "retention_action": message
    }
    
from openai import OpenAI

client = OpenAI(api_key="TU_API_KEY_AQUI")

def generate_llm_explanation(data):

    prompt = f"""
    Sos un analista de riesgo bancario.

    Explicá por qué este cliente tiene riesgo de churn:
    {data}

    Sé claro y orientado a negocio.
    """

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

def decision_engine(proba, threshold, customer):

    if proba < threshold:
        return {
            "action": "no_action",
            "reason": "Low risk"
        }

    # Segmentación de estrategia
    if customer["Balance"] > 100000:
        return {
            "action": "premium_retention",
            "offer": "Asesor financiero personalizado"
        }

    if customer["IsActiveMember"] == 0:
        return {
            "action": "reactivation_campaign",
            "offer": "Bonificación por actividad"
        }

    return {
        "action": "standard_retention",
        "offer": "Promoción general"
    }
    
def simulate_campaign(y_true, y_proba, threshold):

    selected = y_proba >= threshold

    contacted = selected.sum()
    churners_captured = ((y_true == 1) & selected).sum()

    conversion_rate = churners_captured / (contacted + 1)

    return {
        "contacted_clients": int(contacted),
        "saved_clients": int(churners_captured),
        "conversion_rate": conversion_rate
    }
    
@app.post("/predict_pro")
def predict_pro(data: dict):

    df = pd.DataFrame([data])
    df = df.reindex(columns=features, fill_value=0)

    proba = model.predict_proba(df)[:,1][0]

    explanation = generate_explanation(data)
    message = generate_retention_message(data)

    decision = decision_engine(proba, BEST_THRESHOLD, data)

    return {
        "churn_probability": float(proba),
        "explanation": explanation,
        "recommended_action": decision,
        "retention_message": message
    }
    
import datetime

def log_prediction(data, proba):
    with open("logs.txt", "a") as f:
        f.write(f"{datetime.datetime.now()} - {proba} - {data}\n")
        
knowledge_base = {
    "high_value": "Ofrecer asesoramiento personalizado",
    "inactive": "Campaña de reactivación",
    "low_products": "Cross-selling"
}

def ab_test_simulation(y_true, y_proba, threshold):

    # Grupo A (random)
    random_group = np.random.rand(len(y_true)) < 0.5
    
    # Grupo B (modelo)
    model_group = ~random_group

    y_pred_model = (y_proba >= threshold).astype(int)

    profit_A = calculate_profit(y_true[random_group], random_group[random_group])
    profit_B = calculate_profit(y_true[model_group], y_pred_model[model_group])

    return {
        "random_strategy_profit": profit_A,
        "model_strategy_profit": profit_B,
        "uplift": profit_B - profit_A
    }
    
def detect_drift(train_mean, new_mean, threshold=0.1):
    return abs(train_mean - new_mean) > threshold

def churn_rate_by_gender(df):
    return df.groupby("Gender")["Exited"].mean()

df["CLV"] = df["Balance"] * 0.1 + df["EstimatedSalary"] * 0.05

def calculate_profit_clv(y_true, y_pred, clv):

    tp = (y_true == 1) & (y_pred == 1)
    fp = (y_true == 0) & (y_pred == 1)

    profit = (clv[tp].sum()) - (20 * fp.sum())

    return profit

import shap

explainer = shap.Explainer(model)
shap_values = explainer(X_test)

shap.plots.bar(shap_values)

