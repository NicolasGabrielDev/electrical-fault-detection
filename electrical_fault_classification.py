import os
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
warnings.filterwarnings("ignore")

import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

KAGGLE_DATASET = "esathyaprakash/electrical-fault-detection-and-classification"
TARGET_COLUMN = "fault_type"
TEST_SIZE = 0.2
RANDOM_STATE = 42
MINIMUM_RECALL_PER_CLASS = 0.90
RESULTS_PATH = Path("model_comparison_results.csv")

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use("seaborn-whitegrid")

pd.set_option("display.max_columns", 100)
pd.set_option("display.float_format", "{:.4f}".format)


def select_dataset_file(dataset_directory):
    csv_files = sorted(dataset_directory.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError("Nenhum arquivo CSV foi encontrado no diretório baixado do dataset.")
    for preferred_name in ["classData.csv", "classdata.csv"]:
        for csv_file in csv_files:
            if csv_file.name == preferred_name:
                return csv_file
    return max(csv_files, key=lambda csv_file: csv_file.stat().st_size)


def build_fault_type(row):
    label_columns = ["G", "C", "B", "A"]
    active_labels = [label for label in label_columns if int(row[label]) == 1]
    return "Sem falha" if not active_labels else "".join(active_labels)


def prepare_target(dataframe):
    label_columns = ["G", "C", "B", "A"]
    if TARGET_COLUMN in dataframe.columns:
        return dataframe.copy(), []
    if set(label_columns).issubset(dataframe.columns):
        prepared = dataframe.copy()
        prepared[TARGET_COLUMN] = prepared.apply(build_fault_type, axis=1)
        return prepared, label_columns
    candidate_columns = ["faultType", "Fault Type", "Fault_Type", "Output (S)", "Output"]
    for candidate_column in candidate_columns:
        if candidate_column in dataframe.columns:
            return dataframe.rename(columns={candidate_column: TARGET_COLUMN}).copy(), []
    raise ValueError("A coluna-alvo não pôde ser identificada.")


def build_preprocessor(training_features):
    numeric_training_features = training_features.select_dtypes(include=np.number).columns.tolist()
    categorical_training_features = training_features.select_dtypes(exclude=np.number).columns.tolist()
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer(transformers=[
        ("numeric", numeric_transformer, numeric_training_features),
        ("categorical", categorical_transformer, categorical_training_features),
    ])


def summarize_outliers(dataframe, columns):
    summaries = []
    for column in columns:
        q1 = dataframe[column].quantile(0.25)
        q3 = dataframe[column].quantile(0.75)
        iqr = q3 - q1
        outlier_count = ((dataframe[column] < q1 - 1.5 * iqr) | (dataframe[column] > q3 + 1.5 * iqr)).sum()
        summaries.append({"variavel": column, "outliers": outlier_count, "taxa_outliers": outlier_count / len(dataframe)})
    return pd.DataFrame(summaries).sort_values("taxa_outliers", ascending=False)


def get_feature_names(fitted_pipeline):
    return fitted_pipeline.named_steps["preprocessor"].get_feature_names_out()


def plot_confusion_matrix(model_name, y_test, y_pred, class_labels):
    matrix = confusion_matrix(y_test, y_pred, labels=class_labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title(f"Matriz de confusão - {model_name}")
    ax.set_xlabel("Previsto")
    ax.set_ylabel("Real")
    ax.set_xticks(range(len(class_labels)))
    ax.set_xticklabels(class_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(class_labels)))
    ax.set_yticklabels(class_labels)
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            ax.text(col_index, row_index, matrix[row_index, col_index], ha="center", va="center", color="black")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()


def plot_feature_importance(model_name, trained_models, top_n=12):
    pipeline = trained_models[model_name]
    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        print(f"{model_name} não expõe feature_importances_.")
        return pd.DataFrame()
    importance_data = pd.DataFrame({
        "variavel": get_feature_names(pipeline),
        "importancia": model.feature_importances_,
    }).sort_values("importancia", ascending=False)
    plot_data = importance_data.head(top_n).sort_values("importancia")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(plot_data["variavel"], plot_data["importancia"], color="#8a5a44")
    ax.set_title(f"Importância das variáveis - {model_name}")
    ax.set_xlabel("Importância")
    plt.tight_layout()
    plt.show()
    return importance_data


def main():
    dataset_path = kagglehub.dataset_download(KAGGLE_DATASET)
    dataset_directory = Path(dataset_path)
    print("Caminho dos arquivos do dataset:", dataset_directory)

    dataset_file_path = select_dataset_file(dataset_directory)
    print("Arquivo de dataset selecionado:", dataset_file_path)

    raw_data = pd.read_csv(dataset_file_path)
    print("Linhas e colunas:", raw_data.shape)
    print("Colunas:", raw_data.columns.tolist())

    data, target_source_columns = prepare_target(raw_data)
    print("Colunas de origem da variável-alvo:", target_source_columns or [TARGET_COLUMN])

    print("Dimensões do dataset:", data.shape)
    print("Linhas duplicadas:", data.duplicated().sum())
    print(data.isna().sum().to_frame("valores_ausentes").to_string())
    print(data[TARGET_COLUMN].value_counts().to_frame("quantidade").to_string())

    analysis_feature_data = data.drop(columns=[TARGET_COLUMN] + target_source_columns)
    numeric_columns_for_analysis = analysis_feature_data.select_dtypes(include=np.number).columns.tolist()
    class_counts = data[TARGET_COLUMN].value_counts()

    fig, ax = plt.subplots(figsize=(9, 4))
    class_counts.plot(kind="bar", ax=ax, color="#2f6f8f")
    ax.set_title("Distribuição das classes")
    ax.set_xlabel("Tipo de falha")
    ax.set_ylabel("Registros")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    data[numeric_columns_for_analysis].hist(figsize=(12, 8), bins=30, color="#3b7a57")
    plt.suptitle("Histogramas das variáveis numéricas", y=1.02)
    plt.tight_layout()
    plt.show()

    correlation_matrix = data[numeric_columns_for_analysis].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(correlation_matrix, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_title("Matriz de correlação")
    ax.set_xticks(range(len(correlation_matrix.columns)))
    ax.set_xticklabels(correlation_matrix.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(correlation_matrix.columns)))
    ax.set_yticklabels(correlation_matrix.columns)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()

    selected_boxplot_columns = numeric_columns_for_analysis[:6]
    fig, axes = plt.subplots(len(selected_boxplot_columns), 1, figsize=(10, 3 * len(selected_boxplot_columns)))
    if len(selected_boxplot_columns) == 1:
        axes = [axes]
    for axis, column in zip(axes, selected_boxplot_columns):
        data.boxplot(column=column, by=TARGET_COLUMN, ax=axis, rot=45)
        axis.set_title(f"{column} por tipo de falha")
        axis.set_xlabel("Tipo de falha")
        axis.set_ylabel(column)
    plt.suptitle("")
    plt.tight_layout()
    plt.show()

    outlier_summary = summarize_outliers(data, numeric_columns_for_analysis)
    print(outlier_summary.to_string())

    modeling_data = data.drop_duplicates().copy()
    feature_columns_to_drop = [TARGET_COLUMN] + target_source_columns
    features = modeling_data.drop(columns=feature_columns_to_drop)
    target = modeling_data[TARGET_COLUMN]

    print("Linhas após remoção de duplicatas:", len(modeling_data))

    required_three_phase_columns = ["Ia", "Ib", "Ic", "Va", "Vb", "Vc"]
    missing_columns = [c for c in required_three_phase_columns if c not in features.columns]
    if missing_columns:
        raise ValueError(f"Colunas necessárias para engenharia de features não encontradas: {missing_columns}")

    features = features.copy()
    features["I_seq_zero"] = features[["Ia", "Ib", "Ic"]].mean(axis=1)
    features["V_seq_zero"] = features[["Va", "Vb", "Vc"]].mean(axis=1)
    features["I_unbalance"] = features[["Ia", "Ib", "Ic"]].std(axis=1)
    features["V_unbalance"] = features[["Va", "Vb", "Vc"]].std(axis=1)
    features["I_magnitude"] = np.sqrt((features[["Ia", "Ib", "Ic"]] ** 2).sum(axis=1))
    features["V_magnitude"] = np.sqrt((features[["Va", "Vb", "Vc"]] ** 2).sum(axis=1))

    print("Colunas numéricas atualizadas:", features.select_dtypes(include=np.number).columns.tolist())

    minimum_class_count = target.value_counts().min()
    stratify_target = target if minimum_class_count >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=stratify_target,
    )
    print("Linhas de treino:", X_train.shape[0])
    print("Linhas de teste:", X_test.shape[0])

    models = {
        "Árvore de Decisão": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Floresta Aleatória": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "Regressão Logística": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
    }

    trained_models = {}
    for model_name, model in models.items():
        pipeline = Pipeline(steps=[
            ("preprocessor", build_preprocessor(X_train)),
            ("model", model),
        ])
        pipeline.fit(X_train, y_train)
        trained_models[model_name] = pipeline
        print(f"Modelo treinado: {model_name}")

    evaluation_rows = []
    predictions = {}

    for model_name, pipeline in trained_models.items():
        y_pred = pipeline.predict(X_test)
        predictions[model_name] = y_pred
        evaluation_rows.append({
            "model": model_name,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "f1_score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        })
        print(f"\n{model_name}")
        print(classification_report(y_test, y_pred, zero_division=0))

    results = pd.DataFrame(evaluation_rows).sort_values(["f1_score", "recall"], ascending=False).reset_index(drop=True)
    print(results.to_string())

    class_labels = sorted(y_test.unique())
    per_class_reports = {}

    for model_name, y_pred in predictions.items():
        print(f"\nAnálise por classe - {model_name}")
        plot_confusion_matrix(model_name, y_test, y_pred, class_labels)
        report = classification_report(y_test, y_pred, labels=class_labels, output_dict=True, zero_division=0)
        report_table = pd.DataFrame(report).T.loc[class_labels, ["precision", "recall", "f1-score", "support"]]
        per_class_reports[model_name] = report_table
        print(report_table.to_string())
        for class_name, row in report_table[report_table["recall"] < MINIMUM_RECALL_PER_CLASS].iterrows():
            print(
                f"ALERTA: {model_name} teve recall {row['recall']:.4f} "
                f"para a classe {class_name}, abaixo do limiar configurado de {MINIMUM_RECALL_PER_CLASS:.2f}."
            )

    fold_count = min(5, int(y_train.value_counts().min()))
    cross_validation_rows = []

    if fold_count >= 2:
        cross_validator = StratifiedKFold(n_splits=fold_count, shuffle=True, random_state=RANDOM_STATE)
        for model_name, model in models.items():
            pipeline = Pipeline(steps=[("preprocessor", build_preprocessor(X_train)), ("model", model)])
            scores = cross_val_score(pipeline, X_train, y_train, cv=cross_validator, scoring="f1_weighted", n_jobs=1)
            cross_validation_rows.append({"model": model_name, "cv_f1_mean": scores.mean(), "cv_f1_std": scores.std()})

    cross_validation_results = pd.DataFrame(cross_validation_rows)
    if cross_validation_results.empty:
        print("A validação cruzada foi ignorada porque pelo menos uma classe tem menos de dois registros.")
    else:
        print(cross_validation_results.to_string())

    if not cross_validation_results.empty:
        results = results.merge(cross_validation_results, on="model", how="left")

    results_for_report = results.rename(columns={
        "model": "modelo",
        "accuracy": "acurácia",
        "precision": "precisão",
        "recall": "recall",
        "f1_score": "f1-score",
        "cv_f1_mean": "f1_médio_validação_cruzada",
        "cv_f1_std": "desvio_f1_validação_cruzada",
    })
    results_for_report.to_csv(RESULTS_PATH, index=False)
    best_model_name = results.iloc[0]["model"]
    best_model = trained_models[best_model_name]

    print(results_for_report.to_string())
    print("Resultados salvos em:", RESULTS_PATH)
    print("Melhor modelo:", best_model_name)

    metric_columns = ["accuracy", "precision", "recall", "f1_score"]
    results.set_index("model")[metric_columns].plot(kind="bar", figsize=(10, 5))
    plt.title("Comparação de métricas entre modelos")
    plt.ylabel("Pontuação")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    for model_name in ["Árvore de Decisão", "Floresta Aleatória", "Gradient Boosting"]:
        if model_name in trained_models:
            plot_feature_importance(model_name, trained_models)

    if RESULTS_PATH.exists():
        comparison_from_csv = pd.read_csv(RESULTS_PATH)
        logistic_row = comparison_from_csv.loc[comparison_from_csv["modelo"] == "Regressão Logística"].iloc[0]
        best_row = comparison_from_csv.sort_values("f1-score", ascending=False).iloc[0]
        majority_class_rate = target.value_counts(normalize=True).max()
        logistic_analysis = pd.DataFrame([
            {"indicador": "Acurácia da Regressão Logística", "valor": logistic_row["acurácia"]},
            {"indicador": "F1 ponderado da Regressão Logística", "valor": logistic_row["f1-score"]},
            {"indicador": "F1 médio em validação cruzada da Regressão Logística", "valor": logistic_row["f1_médio_validação_cruzada"]},
            {"indicador": "F1 ponderado do melhor modelo", "valor": best_row["f1-score"]},
            {"indicador": "Diferença de F1 para o melhor modelo", "valor": best_row["f1-score"] - logistic_row["f1-score"]},
            {"indicador": "Proporção da maior classe", "valor": majority_class_rate},
        ])
        print(logistic_analysis.to_string())


if __name__ == "__main__":
    main()