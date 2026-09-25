import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Load_data import load_data


# ============================================================
# CLAIMWISE - LOG TRANSFORMATION
# ============================================================

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

CHART_DIR = os.path.join(
    PROJECT_DIR,
    "static",
    "charts"
)

os.makedirs(CHART_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

df = load_data()


# ============================================================
# ORIGINAL LOSS
# ============================================================

original_loss = df["loss"]


print("==========================================")
print("     CLAIMWISE - LOG TRANSFORMATION")
print("==========================================")

print("\nOriginal Loss Statistics:")

print("Mean:", round(original_loss.mean(), 2))
print("Std:", round(original_loss.std(), 2))
print("Minimum:", round(original_loss.min(), 2))
print("Median:", round(original_loss.median(), 2))
print("Maximum:", round(original_loss.max(), 2))


# ============================================================
# APPLY LOG TRANSFORMATION
# ============================================================

df["log_loss"] = np.log(df["loss"])


# ============================================================
# LOG LOSS STATISTICS
# ============================================================

log_loss = df["log_loss"]

print("\nLog Transformed Loss Statistics:")

print("Mean:", round(log_loss.mean(), 6))
print("Std:", round(log_loss.std(), 6))
print("Minimum:", round(log_loss.min(), 6))
print("25%:", round(log_loss.quantile(0.25), 6))
print("Median:", round(log_loss.median(), 6))
print("75%:", round(log_loss.quantile(0.75), 6))
print("Maximum:", round(log_loss.max(), 6))


# ============================================================
# PLOT ORIGINAL DISTRIBUTION
# ============================================================

plt.figure(figsize=(10, 6))

plt.hist(
    original_loss,
    bins=50
)

plt.title("Original Claim Loss Distribution")
plt.xlabel("Loss")
plt.ylabel("Number of Claims")

plt.tight_layout()

plt.savefig(
    os.path.join(
        CHART_DIR,
        "original_loss_distribution.png"
    )
)

plt.close()


# ============================================================
# PLOT LOG DISTRIBUTION
# ============================================================

plt.figure(figsize=(10, 6))

plt.hist(
    log_loss,
    bins=50
)

plt.title("Log Transformed Claim Loss Distribution")
plt.xlabel("Log(Loss)")
plt.ylabel("Number of Claims")

plt.tight_layout()

plt.savefig(
    os.path.join(
        CHART_DIR,
        "log_loss_distribution.png"
    )
)

plt.close()


# ============================================================
# COMPLETION
# ============================================================

print("\nCharts created successfully.")

print(
    "\nOriginal chart:"
    "\nstatic/charts/original_loss_distribution.png"
)

print(
    "\nLog transformed chart:"
    "\nstatic/charts/log_loss_distribution.png"
)

print("\nLog transformation completed successfully.")