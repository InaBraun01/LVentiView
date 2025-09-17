import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load CSV
df_ED = pd.read_csv("/data.lfpn/ibraun/Code/paper_volume_calculation/test/test_ED/SCD0003701/dice_history.csv")
df_ES = pd.read_csv("/data.lfpn/ibraun/Code/paper_volume_calculation/test/test_ES/SCD0003701/dice_history.csv")

df_bp = pd.read_csv("/data.lfpn/ibraun/Code/paper_volume_calculation/test/test_2/SCD0003701/bp_dice_history.csv")
df_myo = pd.read_csv("/data.lfpn/ibraun/Code/paper_volume_calculation/test/test_2/SCD0003701/myo_dice_history.csv")

time = np.arange(0,len(df_ED["Myocardium Dice"]))*10

# Plot both value columns over time
plt.figure(figsize=(10, 6))
#plt.plot(time, df_ED["Myocardium Dice"], label="ED: Myocardium Old", c="blue",marker='o',alpha =0.5)
plt.plot(time, df_ED["Blood Pool Dice"], label="ED:Blood Pool Old", c="blue", marker='x',alpha =0.5)

#plt.plot(time, df_ES["Myocardium Dice"], label="ES: Myocardium Old", c="green",marker='o',alpha =0.5)
plt.plot(time, df_ES["Blood Pool Dice"], label="ES: Blood Pool Old", c="green", marker='x',alpha =0.5)

#plt.plot(time, df_myo["0"], label="ED: Myocardium New", c="pink",marker='o',alpha =0.5)
plt.plot(time, df_bp["0"], label="ED:Blood Pool New", c="pink", marker='x',alpha =0.5)

#plt.plot(time, df_myo["1"], label="ES: Myocardium New", c="orange",marker='o',alpha =0.5)
plt.plot(time, df_bp["1"], label="ES: Blood Pool New", c="orange", marker='x',alpha =0.5)

#plt.yscale('log')  # Logarithmic scale on y-axis
#plt.xscale('log')  # Logarithmic scale on y-axis


# Labels and legend
plt.xlabel("Fitting Steps")
plt.ylabel("Dice scores")
plt.title("Values Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("bp_dice.png")
# Show the plot
# plt.show()