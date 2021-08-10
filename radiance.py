import pandas as pd
import numpy as np
from scipy.stats import linregress
from scipy.optimize import bisect, curve_fit
from scipy.misc import derivative
import matplotlib.pyplot as plt

plt.rcParams.update({
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{siunitx}'
})

l_a_ratio = 1.47e7
delta_l_a_ratio = 0.09e7


# for temperatures between 90 and 750
def resistivity_1(temp):
    return (- 1.06871 + 2.06884e-2 * temp + 1.27971e-6 * temp ** 2 +
            8.53101e-9 * temp ** 3 - 5.14195e-12 * temp ** 4) * 1e-8


def resistance_1(temp):
    return resistivity_1(temp) * l_a_ratio


# for temperatures between 700 and 3600
def resistivity_2(temp):
    return (- 1.72573 + 2.1435e-2 * temp + 5.74811e-6 * temp ** 2 -
            1.13698e-9 * temp ** 3 + 1.1167e-13 * temp ** 4) * 1e-8


def resistance_2(temp):
    return resistivity_2(temp) * l_a_ratio


radiance_df = pd.read_csv('data/radiance.csv')

lamp_voltage = radiance_df.iloc[:, 0].to_numpy()
delta_lamp_voltage = radiance_df.iloc[:, 1].to_numpy()

lamp_current = radiance_df.iloc[:, 2].to_numpy()
delta_lamp_current = radiance_df.iloc[:, 3].to_numpy()

lamp_resistance = lamp_voltage / lamp_current
delta_lamp_resistance = np.sqrt((delta_lamp_voltage / lamp_voltage) **
                                2 + (delta_lamp_current / lamp_current) ** 2) * lamp_resistance

lamp_temp = np.zeros_like(lamp_resistance)
delta_lamp_temp = np.zeros_like(lamp_temp)

# estimating lamp temperatures and its errors
for i in range(lamp_resistance.size):

    if lamp_resistance[i] <= resistance_1(750):
        a, b = (90, 750)
        resistivity = resistivity_1
        resistance = resistance_1
    else:
        a, b = (750, 3600)
        resistivity = resistivity_1
        resistance = resistance_2

    lamp_temp[i] = bisect(lambda temp: resistance(temp) - lamp_resistance[i],
                          a, b, full_output=False)

    delta_lamp_temp[i] = resistivity(lamp_temp[i]) * np.sqrt((delta_lamp_resistance[i] / lamp_resistance[i]) ** 2 +
                                                             (delta_l_a_ratio / l_a_ratio) ** 2) / derivative(resistivity, lamp_temp[i], dx=0.01)

sensor_voltage = radiance_df.iloc[:, 4].to_numpy()
delta_sensor_voltage = radiance_df.iloc[:, 5].to_numpy()

# linear regression for sensor voltage
radiance_reg = linregress(
    np.log10(lamp_temp), np.log10(sensor_voltage))

print('Radiance Linear Regression Results')
print(f'slope = {radiance_reg.slope}')
print(f'intercept = {radiance_reg.slope}')
print(f'r = {radiance_reg.rvalue}')

# non linear least squares curve fit for sensor voltage
radiance_fit = curve_fit(lambda temp, a, b: a * temp ** b,
                         lamp_temp, sensor_voltage, sigma=delta_sensor_voltage)

coeff, exponent = radiance_fit[0]
delta_coeff, delta_exponent = np.sqrt(np.diag(radiance_fit[1]))

print('\nRadiance Curve Fit Results')
print(f'{coeff:.2e} * temp ^ {exponent:.2f}')
print(f'coefficient error = {delta_coeff:.1e}')
print(f'exponent error = {delta_exponent:.2f}')


# converts number to scientific notation in latex
def sci_to_latex(float):
    number, exponent = f'{float:.2e}'.split(sep='e')
    return f'{number} \\times 10^{{{exponent}}}'


# ploting sensor voltage vs lamp temperatures
fig_radiance, ax_radiance = plt.subplots()

xlim = (500, 1900)

ax_radiance.set(xlabel=r'$T$ (\si{\kelvin})',
                ylabel=r'$U_{p}$ (\si{\volt})',
                xlim=xlim,
                xscale='log',
                yscale='log')

ax_radiance.errorbar(lamp_temp, sensor_voltage, xerr=delta_lamp_temp,
                     fmt='ro', ms=2.8, label='Medidas', zorder=1)

temp = np.linspace(*xlim)

ax_radiance.plot(temp, coeff * temp ** exponent, linewidth=0.7, color='red',
                 label=rf'$U_p(T) = ({sci_to_latex(coeff)}) T^{{{exponent:.2f}}}$', zorder=1)

ax_radiance.grid(linestyle=':',
                 linewidth=0.4,
                 zorder=-1)

ax_radiance.legend(loc='upper left')

lamp_power = lamp_voltage * lamp_current
delta_lamp_power = delta_lamp_resistance

# linear regression for lamp power draw
power_reg = linregress(
    np.log10(lamp_temp), np.log10(lamp_power))

print('\nPower Linear Regression Results')
print(f'slope = {power_reg.slope}')
print(f'intercept = {power_reg.slope}')
print(f'r = {power_reg.rvalue}')

# non linear least squares curve fit for sensor voltage
power_fit = curve_fit(lambda temp, a, b: a * temp ** b,
                      lamp_temp, lamp_power, sigma=delta_lamp_power)

coeff_2, exponent_2 = power_fit[0]
delta_coeff_2, delta_exponent_2 = np.sqrt(np.diag(power_fit[1]))

print('\nPower Curve Fit Results')
print(f'{coeff_2:.2e} * temp ^ {exponent_2:.2f}')
print(f'coefficient error = {delta_coeff_2:.1e}')
print(f'exponent error = {delta_exponent_2:.2f}')

# ploting lamp power draw vs lamp temps
fig_power, ax_power = plt.subplots()

ax_power.set(xlabel=r'$T$ (\si{\kelvin})',
             ylabel=r'$P$ (\si{\watt})',
             xlim=xlim,
             xscale='log',
             yscale='log')

ax_power.errorbar(lamp_temp, lamp_power, xerr=delta_lamp_temp,
                  fmt='ro', ms=2.8, label='Medidas', zorder=1)

ax_power.plot(temp, coeff_2 * temp ** exponent_2, linewidth=0.7, color='red',
              label=rf'$P(T) = ({sci_to_latex(coeff_2)}) T^{{{exponent_2:.2f}}}$', zorder=1)

ax_power.grid(linestyle=':',
              linewidth=0.4,
              zorder=-1)

ax_power.legend(loc='upper left')

# saving images
fig_radiance.savefig('plots/radiance.png', dpi=300)
fig_power.savefig('plots/power.png', dpi=300)

# creating dataframes with final data
temp_array = np.array([lamp_temp, delta_lamp_temp])
temp_df = pd.DataFrame(temp_array.T, columns=('Temperaturas', 'Incertezas'))

reg_array = np.array([list(radiance_reg), list(power_reg)])
reg_df = pd.DataFrame(reg_array[:, :3],
                      columns=('Angular', 'Linear', 'Correlação'))

fit_array = np.array([[coeff, delta_coeff, exponent, delta_exponent],
                      [coeff_2, delta_coeff_2, exponent_2, delta_exponent_2]])

fit_df = pd.DataFrame(fit_array,
                      columns=('A', 'Incerteza A', 'Expoente', 'Incerteza Expoente'))

# saving csvs
temp_df.to_csv('temperatures.csv', index=False)
reg_df.to_csv('regression.csv', index=False)
fit_df.to_csv('fit.csv', index=False, float_format='%.2E')

plt.show()
