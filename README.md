# 📡 Telecom X – Predicción de Cancelación de Clientes (*Churn Prediction*)

Proyecto de análisis y modelado predictivo para anticipar la cancelación de clientes en una empresa de telecomunicaciones.  
Se utilizan técnicas de **machine learning**, análisis exploratorio y herramientas de interpretabilidad como **SHAP**, con el objetivo de identificar los factores clave que influyen en el abandono y proponer **estrategias de retención efectivas**.

---

---

## 🧠 Objetivo del Proyecto
Desarrollar un modelo capaz de **predecir si un cliente cancelará su servicio**, utilizando información sobre:
- Características de contrato
- Historial de pagos
- Uso de servicios

Esto permitirá a la empresa **tomar acciones proactivas** para reducir la tasa de cancelación (*churn rate*).

---

## 🧪 Proceso de Desarrollo

### 1️⃣ Carga y Limpieza de Datos
- Eliminación de identificadores únicos.
- Codificación de variables categóricas con `get_dummies`.
- Balanceo de clases con **SMOTE**.
- Normalización con **StandardScaler**.

### 2️⃣ Análisis Exploratorio (EDA)
- Histogramas, boxplots y gráficos de correlación.
- Identificación de variables influyentes.
- Análisis de churn según contrato, servicios y método de pago.

### 3️⃣ Selección de Características
- `SelectKBest` con `f_classif`.
- **k óptimo = 29** variables seleccionadas por rendimiento.

### 4️⃣ Entrenamiento de Modelos
- **Regresión Logística** (`LogisticRegression`)
- **Random Forest** (`RandomForestClassifier`)
- **Gradient Boosting** (`GradientBoostingClassifier`)
- **K-Nearest Neighbors** (KNN)
- **Ensemble** (*VotingClassifier*: LR + GB)

### 5️⃣ Optimización de Hiperparámetros
- `GridSearchCV` con validación cruzada estratificada.

### 6️⃣ Evaluación
- Métricas: Accuracy, Precision, Recall, F1 Score.
- Matrices de confusión.
- Curvas ROC y AUC.
- Interpretación con SHAP y coeficientes de la Regresión Logística.

---

## 🏆 Resultados

| Modelo                  | Accuracy | Precision | Recall  | F1 Score | AUC   |
|-------------------------|----------|-----------|---------|----------|-------|
| Gradient Boosting       | 0.7887   | 0.5907    | 0.5864  | 0.5886   | 0.850 |
| Regresión Logística     | 0.7956   | 0.5997    | 0.6221  | 0.6107   | 0.840 |
| **Ensemble (LR + GB)**  | **0.8025** | **0.6277** | 0.5740  | 0.5996   | **0.851** |

✅ **Modelo ganador:** *Ensemble LR + GB* → combina la interpretabilidad de LR y el rendimiento de GB, logrando el mejor **AUC**.

---

## 🔍 Principales Factores de Cancelación
Identificados con **SHAP** y análisis de correlación:

- Contrato *month-to-month*.
- Método de pago: *electronic check*.
- Ausencia de servicios adicionales (soporte técnico, seguridad, respaldo online).
- Baja antigüedad (pocos meses en la empresa).
- Cargos mensuales elevados y bajo engagement.

---

## 💡 Estrategias de Retención Sugeridas
- Incentivar contratos de largo plazo (1–2 años).
- Ofrecer **bundles** de servicios de valor agregado (soporte + seguridad + respaldo).
- Promocionar el uso de pagos automáticos (tarjeta o transferencia).
- Seguimiento personalizado a clientes con cargos altos.
- Programas de fidelización para clientes nuevos.

---

## 📦 Requerimientos
# Manipulación y análisis de datos
pandas
numpy

# Visualización
matplotlib
seaborn

# Machine Learning y preprocesamiento
scikit-learn
imbalanced-learn   # Para SMOTE

# Interpretabilidad
shap

# Entorno de trabajo (opcional, si usas notebooks)
jupyter




