import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("/data.lfpn/ibraun/Code/lvmeshfitting/test_overfitting/SCD0003701_Ed/dice_history.csv")

time = np.arange(0,len(df["Myocardium Dice"]))*100

# Plot both value columns over time
plt.figure(figsize=(10, 6))
plt.plot(time, df["Myocardium Dice"], label="Myocardium", marker='o')
plt.plot(time, df["Blood Pool Dice"], label="Blood Pool", marker='x')

#plt.yscale('log')  # Logarithmic scale on y-axis
plt.xscale('log')  # Logarithmic scale on y-axis


# Labels and legend
plt.xlabel("Fitting Steps")
plt.ylabel("Dice scores")
plt.title("Values Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("/data.lfpn/ibraun/Code/lvmeshfitting/test_overfitting/SCD0003701_Ed/change_in_dice.png")
# Show the plot
plt.show()