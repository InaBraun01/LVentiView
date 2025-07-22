import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# Helper function to convert month-year to datetime
def ym_to_date(year, month):
    return datetime(year, month, 1)

# Define the timeline
topics = [
    ("GUI Paper", ym_to_date(2025, 7), ym_to_date(2025, 9)),
    ("Patient-specific PINN", ym_to_date(2025, 8), ym_to_date(2025, 11)),
    ("Generalized PINN", ym_to_date(2025, 11), ym_to_date(2026, 2)),
    ("Add scar tissue", ym_to_date(2026, 2), ym_to_date(2026, 7)),
]

# Convert to matplotlib's date format
bar_data = [(mdates.date2num(start), (end - start).days) for _, start, end in topics]

fig, ax = plt.subplots(figsize=(10, 3.5))

# Plot each topic in grayscale
gray_shades = ['#333333', '#555555', '#777777', '#999999']

for i, ((label, _, _), (start, duration), color) in enumerate(zip(topics, bar_data, gray_shades)):
    ax.broken_barh([(start, duration)], (i - 0.4, 0.8), facecolors=color, edgecolors='black')
    ax.text(start + duration / 2, i, label, va='center', ha='center', color='white', fontsize=10, weight='bold')

# Set y-axis
ax.set_yticks(range(len(topics)))
ax.set_yticklabels([t[0] for t in topics], fontsize=12)
ax.set_ylim(-0.5, len(topics) - 0.5)
ax.invert_yaxis()

# Format x-axis with months
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=12)

# Remove unnecessary spines and add grid
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)

# Add title
ax.set_title('Project Timeline', fontsize=16, weight='bold')

plt.tight_layout()
plt.savefig("First_TAC_time_line.pdf")
plt.show()
