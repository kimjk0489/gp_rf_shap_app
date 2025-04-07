import streamlit as st
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import shap
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.optim import optimize_acqf

# --- 페이지 설정 ---
st.set_page_config(page_title="Slurry 조성 최적화 GP", layout="wide")
st.title("Slurry 조성 최적화 GP")

# --- 데이터 불러오기 ---
CSV_PATH = "slurry_data_wt%_ALL.csv"
try:
    df = pd.read_csv(CSV_PATH)
except FileNotFoundError:
    df = pd.DataFrame(columns=["carbon_black_wt%", "graphite_wt%", "CMC_wt%", "solvent_wt%", "yield_stress"])

# --- 사용자 입력 폼 ---
st.sidebar.header("새로운 실험 조성 추가")
with st.sidebar.form("new_data_form"):
    new_cb = st.number_input("Carbon Black [wt%]", min_value=0.0, step=0.1)
    new_cmc = st.number_input("CMC [wt%]", min_value=0.0, step=0.05)
    new_solvent = st.number_input("Solvent [wt%]", min_value=0.0, step=0.5)
    total_input = new_cb + new_cmc + new_solvent
    new_graphite = max(0.0, 100.0 - total_input)
    st.markdown(f"Graphite: **{new_graphite:.2f} wt%**")
    new_yield = st.number_input("Yield Stress [Pa]", min_value=0.0, step=10.0)
    submitted = st.form_submit_button("데이터 추가")

if submitted:
    if total_input > 100:
        st.sidebar.error("⚠️ 조성 합이 100을 초과했습니다.")
    else:
        new_row = {
            "carbon_black_wt%": new_cb,
            "graphite_wt%": new_graphite,
            "CMC_wt%": new_cmc,
            "solvent_wt%": new_solvent,
            "yield_stress": new_yield,
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(CSV_PATH, index=False)
        st.sidebar.success("✅ 데이터 저장 완료")

# --- 전처리 ---
x_cols = ["carbon_black_wt%", "graphite_wt%", "CMC_wt%", "solvent_wt%"]
y_cols = ["yield_stress"]
X_raw = df[x_cols].values
Y_raw = df[y_cols].values

param_bounds = {
    "carbon_black_wt%": (1.75, 10.0),
    "graphite_wt%": (18.0, 38.0),
    "CMC_wt%": (0.7, 1.5),
    "solvent_wt%": (58.0, 78.0),
}
bounds_array = np.array([param_bounds[k] for k in x_cols])
x_scaler = MinMaxScaler()
x_scaler.fit(bounds_array.T)

X_scaled = x_scaler.transform(X_raw)
train_x = torch.tensor(X_scaled, dtype=torch.double)
train_y = torch.tensor(Y_raw, dtype=torch.double)

# --- GP 모델 학습 ---
model = SingleTaskGP(train_x, train_y)
mll = ExactMarginalLogLikelihood(model.likelihood, model)
fit_gpytorch_mll(mll)

# --- Random Forest 학습 ---
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_scaled, Y_raw.ravel())

# --- 후보 추천 ---
input_dim = train_x.shape[1]
bounds = torch.tensor([[0.0] * input_dim, [1.0] * input_dim], dtype=torch.double)
scales = bounds_array[:, 1] - bounds_array[:, 0]
offset = np.sum(bounds_array[:, 0])
rhs = 100.0 - offset
indices = torch.arange(len(x_cols), dtype=torch.long)
coefficients = torch.tensor(scales, dtype=torch.double)
rhs_tensor = torch.tensor(rhs, dtype=torch.double)
inequality_constraints = [
    (indices, coefficients, rhs_tensor),
    (indices, -coefficients, -rhs_tensor),
]

def is_duplicate(candidate_scaled, train_scaled, tol=1e-3):
    return any(np.allclose(candidate_scaled, x, atol=tol) for x in train_scaled)

candidate_wt = None
if st.button("Candidate"):
    best_y = train_y.max().item()
    acq_fn = ExpectedImprovement(model=model, best_f=best_y, maximize=True)
    for _ in range(10):
        candidate_scaled, _ = optimize_acqf(
            acq_function=acq_fn,
            bounds=bounds,
            q=1,
            num_restarts=10,
            raw_samples=100,
            inequality_constraints=inequality_constraints,
        )
        candidate_np = candidate_scaled.detach().numpy()[0]
        if is_duplicate(candidate_np, train_x.numpy()):
            continue
        y_pred = model.posterior(torch.tensor(candidate_np.reshape(1, -1), dtype=torch.double)).mean.item()
        if y_pred > 0:
            candidate_wt = x_scaler.inverse_transform(candidate_np.reshape(1, -1))[0]
            break
    if candidate_wt is not None:
        st.subheader("Candidate")
        for i, col in enumerate(x_cols):
            st.write(f"{col}: **{candidate_wt[i]:.2f} wt%**")
        st.write(f"**총합**: {np.sum(candidate_wt):.2f} wt%")
        st.write(f"**예측 Yield Stress (GP)**: {y_pred:.2f} Pa")

# --- 예측 곡선 시각화 ---
cb_idx = x_cols.index("carbon_black_wt%")
x_vals_scaled = np.linspace(0, 1, 100)
mean_scaled = np.mean(X_scaled, axis=0)
X_test_scaled = np.tile(mean_scaled, (100, 1))
X_test_scaled[:, cb_idx] = x_vals_scaled
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.double)

model.eval()
with torch.no_grad():
    posterior = model.posterior(X_test_tensor)
    mean = posterior.mean.numpy().flatten()
    std = posterior.variance.sqrt().numpy().flatten()

cb_vals_wt = x_scaler.inverse_transform(X_test_scaled)[:, cb_idx]
train_x_cb = x_scaler.inverse_transform(train_x.numpy())[:, cb_idx]
train_y_np = train_y.numpy().flatten()
y_rf_pred = rf_model.predict(X_test_scaled)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(cb_vals_wt, mean, label="GP Predicted Mean", color="blue")
ax.fill_between(cb_vals_wt, mean - 1.96 * std, mean + 1.96 * std, color="blue", alpha=0.2, label="95% CI")
ax.plot(cb_vals_wt, y_rf_pred, label="RF Prediction", color="green", linestyle="--")
ax.scatter(train_x_cb, train_y_np, color="red", label="Observed Data")
if candidate_wt is not None:
    cand_scaled = x_scaler.transform(candidate_wt.reshape(1, -1))
    pred_y_gp = model.posterior(torch.tensor(cand_scaled, dtype=torch.double)).mean.item()
    pred_y_rf = rf_model.predict(cand_scaled)[0]
    ax.scatter(candidate_wt[cb_idx], pred_y_gp, color="yellow", label="GP Candidate")
    ax.scatter(candidate_wt[cb_idx], pred_y_rf, color="green", marker="x", label="RF Candidate")

ax.set_xlabel("Carbon Black [wt%]")
ax.set_ylabel("Yield Stress [Pa]")
ax.set_title("GP vs RF Prediction")
ax.grid(True)
ax.legend()
st.pyplot(fig)

# --- Random Forest Feature Importance ---
st.subheader("Random Forest 기반 조성 해석")

st.markdown("### Gini Importance (트리 분할 기준 중요도)")
gini_importances = rf_model.feature_importances_
sorted_idx = np.argsort(gini_importances)
fig_gini, ax_gini = plt.subplots()
ax_gini.barh(np.array(x_cols)[sorted_idx], gini_importances[sorted_idx], color="skyblue")
ax_gini.set_xlabel("Importance")
ax_gini.set_title("Gini Feature Importance (Random Forest)")
st.pyplot(fig_gini)

st.markdown("### SHAP Importance (예측 기여도 기반 중요도)")
explainer = shap.Explainer(rf_model, X_scaled)
shap_values = explainer(X_scaled, check_additivity=False)
fig_shap = plt.figure()
shap.summary_plot(shap_values, X_scaled, feature_names=x_cols, show=False)
st.pyplot(fig_shap)