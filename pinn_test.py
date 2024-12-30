import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("pinn_predictions.csv")

plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(data["time"], data["x_true"], label="True X")
plt.plot(data["time"], data["x_pred"], label="Predicted X", linestyle="dashed")
plt.title("X Position")
plt.xlabel("Time")
plt.ylabel("X")
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(data["time"], data["y_true"], label="True Y")
plt.plot(data["time"], data["y_pred"], label="Predicted Y", linestyle="dashed")
plt.title("Y Position")
plt.xlabel("Time")
plt.ylabel("Y")
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(data["time"], data["z_true"], label="True Z")
plt.plot(data["time"], data["z_pred"], label="Predicted Z", linestyle="dashed")
plt.title("Z Position")
plt.xlabel("Time")
plt.ylabel("Z")
plt.legend()

plt.tight_layout()
plt.show()
