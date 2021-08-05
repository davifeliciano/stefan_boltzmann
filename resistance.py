import pandas as pd
import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt

plt.rcParams.update({
    'text.usetex': True,
})

resistance_df = pd.read_csv('data/resistance.csv')

lamp_voltage = resistance_df.iloc[:, 0].to_numpy()
lamp_current = resistance_df.iloc[:, 1].to_numpy()

resistance_reg = linregress(lamp_current, lamp_voltage)

current_fit = np.linspace(0, 50)
slope = resistance_reg.slope
intercept = resistance_reg.intercept
voltage_fit = slope * current_fit + intercept

fig, ax = plt.subplots()

ax.set(xlabel=r'$I$ (mA)',
       ylabel=r'$U$ (mV)',
       xlim=(0, 50),
       ylim=(0, 40))

if intercept > 0:
    sign = '+'
else:
    sign = '-'

intercept = abs(intercept)

ax.plot(lamp_current, lamp_voltage, 'ro',
        ms=2.8, label='Medidas', zorder=1)

ax.plot(current_fit, voltage_fit,
        linewidth=0.7,
        color='red',
        label=rf'$ U(I) = {slope:.2f}I {sign} {intercept:.2f}$',
        zorder=1)

ax.grid(linestyle=':',
        linewidth=0.4,
        zorder=-1)

ax.legend(loc='upper left')

fig.savefig('plots/resistance.png', dpi=300)

plt.show()
