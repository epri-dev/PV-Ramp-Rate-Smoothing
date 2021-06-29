# -*- coding: utf-8 -*-
"""
@author: pdfr001
"""

import ramp_rate_control
import ramp_rate_optimization
import matplotlib.pyplot as plt
import pandas as pd


plt.close('all')


#%% Settings
settings = {
              #constraints
              "max_ramp": 0.1, #max rate of change allowed per ramp interval, 0.1 corresponds to 10%
              "ramp_interval": 10, #(minutes), interval at which ramp rate is calculated. Average power is taken over the interval
              "AC_upper_bound_on": 1, #(1-true,0-false) enforce upper bound of AC power
              "AC_lower_bound_on": 1, #(1-true,0-false) enforce lower bound of AC power
              "AC_upper_bound": 1.05, #times AC nameplate
              "AC_lower_bound": -0.01, #times AC nameplate (use negative if grid draw allowed)
              
              'short_forecast': 0, #(1-true,0-false) Select whether controller uses short-term power forecast
              "forecast_shift_periods": 3, #forecasting window in terms of periods of the ramp interval
              
              
              'battery_energy': 0.2, #(hours). 1 corresponds to an X kWh battery, for a nameplate PV power of X 
              'battery_power': 1, #power rating of battery as a fraction of the PV system power. 1 for full rating
              'round_trip_efficiency': 0.9, #round trip efficiency of the storage system, including power electronics. 1 for no loss
              
              'curtail_as_control': 0, #(1-true,0-false) With this setting, curtailment is considered part of the control and is used to correct up-ramp violations. 
              'curtail_if_violation': 0, #(1-true,0-false) exclusive from 'curtail_as_control'. This setting specifies what happens in the case of an up-ramp violation
                                          #for false - violations are sent through to the grid. for true - power is curtailed
                                          #a violation is counted in either case. the sum energy sent to the grid is reduced if this setting is true
              
              }

settings0 = settings.copy()

#%% Data Import
data_import = 1
if data_import:
    df = pd.read_csv("./sample_data.csv") #read sample 1-minute power signal
    df.index = pd.date_range(start='1/1/2019 00:00', end='12/31/2019 23:59', freq='t')
    df.columns = ['Time_stamp','Power'] #rename the columns
    df = df[['Power']] #keep only the power column
    df['Power_scaled'] = df['Power'].divide(500) #normalize by the AC nameplate rating
    
    
#%% Execute Smoothing
plot_results = 1
violation_count, total_energy = ramp_rate_control.run_smooth_controller(df['Power_scaled'], settings.copy(), plot_results)
print("Violations: %.0f" % violation_count)
print("Violation Percent %.2f (of all intervals-including nighttime)" % (100*violation_count/(len(df.index)/settings['ramp_interval'])))
print("Total Energy to Grid Percent %.2f (loss due to curtail and battery loss)" % (100*total_energy/df['Power_scaled'].resample('h').mean().sum()))


#%% Execute Parameter optimization (optimizes for violations, not energy)
train_min, test_min, [kp_best, ki_best, kf_best, soc_rest_best] = ramp_rate_optimization.optimize_params(df['Power_scaled'], settings)


#%% Execute battery size sweep
battery_sweep = [0.2, 0.1, 0.05]#[0.3, 0.2, 0.1, 0.05, 0.025]
battery_size_sweep, violation_sweeptrain, violation_sweeptest, energy_output_sweep = ramp_rate_optimization.size_sweep(df['Power_scaled'], settings.copy(), battery_sweep)

fig, ax = plt.subplots()
plt.plot(battery_size_sweep, violation_sweeptrain, marker='.', label='Training')
plt.plot(battery_size_sweep, violation_sweeptest, marker='.', label='Testing')
ax.legend()
