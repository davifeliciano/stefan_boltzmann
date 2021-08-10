import pandas as pd
import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt

plt.rcParams.update({
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{siunitx}'
})

resistance_df = pd.read_csv('data/resistance.csv')

lamp_voltage = resistance_df.iloc[:, 0].to_numpy()
lamp_current = resistance_df.iloc[:, 1].to_numpy()

resistance_reg = linregress(lamp_current, lamp_voltage)

slope = resistance_reg.slope
intercept = resistance_reg.intercept
rvalue = resistance_reg.rvalue

n = lamp_voltage.size

sum_x_squares = np.sum(lamp_current ** 2)

delta_y = np.sqrt(np.sum((lamp_voltage - slope * lamp_current -
                  intercept) ** 2) / (n - 2))

delta = n * sum_x_squares - (np.sum(lamp_current)) ** 2

slope_error = delta_y * np.sqrt(sum_x_squares / delta)

print('Regression Results')
print(f'slope = {slope:.2f}')
print(f'slope_error = {slope_error:.2f}')
print(f'correlation = {rvalue}')

fig, ax = plt.subplots()

ax.set(xlabel=r'$I$ (\si{\milli\ampere})',
       ylabel=r'$U$ (\si{\milli\volt})',
       xlim=(0, 50),
       ylim=(0, 40))

if intercept > 0:
    sign = '+'
else:
    sign = '-'

intercept = abs(intercept)

current = np.linspace(0, 50)
voltage_fit = slope * current + intercept

ax.plot(lamp_current, lamp_voltage, 'ro',
        ms=2.8, label='Medidas', zorder=1)

ax.plot(current, voltage_fit,
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
