# -*- coding: utf-8 -*-
"""
@author: pdfr001
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def run_smooth_controller(pv_input, settings, plotoutput, kp=1.2, ki=1.8, kf=0.3, soc_rest=0.5):    
    #memory variables
    previous_power = 0
    battery_soc = 0
    #outputs lists
    outpower = []
    battpower = []
    battsoc = []
    curtail = []
    violation_list = []
    
    #conversion factors
    power_to_energy_conversion_factor = settings['ramp_interval']/60
    batt_half_round_trip_eff = settings['round_trip_efficiency']**0.5
    
    
    #pre-processing:
    #discretize PV input by length of ramp_interval
    PV_ramp_interval = pv_input.resample(str(settings['ramp_interval'])+'t', label='right').mean()
    #simulate a perfect forecasting signal by taking rolling sum of future pv power values
    forecast_pv_energy = PV_ramp_interval[::-1].rolling(window=settings['forecast_shift_periods'], min_periods=0).sum()[::-1].multiply(settings['ramp_interval']/60)
    
    if settings['short_forecast'] == 0: #disable forecasting if settings is zero
        kf=0
        
    #iterate through time-series
    for pv_power, forecast_power in np.nditer([PV_ramp_interval.values, forecast_pv_energy.values]):
        #calculate controller error
        delta_power = pv_power - previous_power #proportional error
        soc_increment = battery_soc + (pv_power-previous_power)*power_to_energy_conversion_factor #integral error
        future_error = previous_power*settings['forecast_shift_periods']*power_to_energy_conversion_factor - forecast_power #derivitive error
        error = kp*delta_power + ki*(soc_increment-soc_rest*settings['battery_energy']) - kf*future_error
        
        #calculate the desired output power, enforce ramp rate limit
        if error > 0:
            out_power = previous_power + min(settings['max_ramp'],abs(error))
        else:
            out_power = previous_power - min(settings['max_ramp'],abs(error))
        
        #enforce grid power limits
        if settings['AC_upper_bound_on']:
            if out_power > settings['AC_upper_bound']:
                out_power = settings['AC_upper_bound']
        if settings['AC_lower_bound_on']:
            if out_power < settings['AC_lower_bound']:
                out_power = settings['AC_lower_bound']
        
        #calculate desired (unconstrained) battery power
        battery_power_terminal = out_power - pv_power # positive is power leaving battery (discharging)
        
        #adjust battery power to factor in battery constraints
        #check SOC limit - reduce battery power if either soc exceeds either 0 or 100%
        #check full
        if (battery_soc - battery_power_terminal*batt_half_round_trip_eff*power_to_energy_conversion_factor) > settings['battery_energy']:
            battery_power_terminal = -1*(settings['battery_energy'] - battery_soc)/power_to_energy_conversion_factor/batt_half_round_trip_eff
        #check empty
        elif (battery_soc - battery_power_terminal*power_to_energy_conversion_factor) < 0:
            battery_power_terminal = battery_soc/power_to_energy_conversion_factor/batt_half_round_trip_eff
        
        #enforce battery power limits
        #discharging too fast
        if battery_power_terminal > settings['battery_power']:                
            battery_power_terminal = settings['battery_power']            
        #charging too fast
        elif battery_power_terminal < -1*settings['battery_power']:
            battery_power_terminal = -1*settings['battery_power'] 
         
        #update output power after battery constraints are applied
        out_power = pv_power + battery_power_terminal
        
        #flag if a ramp rate violation has occurred - up or down - because limits of battery prevented smoothing
        violation = 0
        if abs(out_power - previous_power)>(settings['max_ramp']+0.00001):
            violation = 1
            
        
        #curtailment 
        curtail_power = 0
        
        #if curtailment is considered part of the control - don't count up-ramp violations
        if settings['curtail_as_control']:
            if (out_power - previous_power)>(settings['max_ramp']-0.00001):
                out_power = previous_power+settings['max_ramp'] #reduce output to a non-violation
                curtail_power = pv_power + battery_power_terminal - out_power #curtail the remainder
                violation = 0
        
        
        #with this setting, curtail output power upon an upramp violation - rather than sending excess power to the grid
        #curtailment still counts as a violation
        #sum total of energy output is reduced
        if settings['curtail_if_violation']:
            if (out_power - previous_power)>(settings['max_ramp']-0.00001):
                out_power = previous_power+settings['max_ramp'] #reduce output to a non-violation
                curtail_power = pv_power + battery_power_terminal - out_power #curtail the remainder
        
        #update memory variables
        if battery_power_terminal > 0:#discharging - efficiency loss increases the amount of energy drawn from the battery
            battery_soc = battery_soc - battery_power_terminal*power_to_energy_conversion_factor/batt_half_round_trip_eff
        elif battery_power_terminal < 0:#charging - efficiency loss decreases the amount of energy put into the battery
            battery_soc = battery_soc - battery_power_terminal*batt_half_round_trip_eff*power_to_energy_conversion_factor
        previous_power = out_power
        
        #update output variables
        outpower.append(out_power)
        battpower.append(battery_power_terminal)
        battsoc.append(battery_soc)
        violation_list.append(violation)
        curtail.append(curtail_power)
    
    #post-processing
    violation_count = np.sum(violation_list)
    
    
    if plotoutput:
        outpower_series = pd.Series(outpower, index=pv_input.resample(str(settings['ramp_interval'])+'t', label='right').mean().index)
        battery_soc_series = pd.Series(battsoc, index=pv_input.resample(str(settings['ramp_interval'])+'t', label='right').mean().index).multiply(1/settings['battery_energy'])
        violation_series = pd.Series(violation_list, index=pv_input.resample(str(settings['ramp_interval'])+'t', label='right').mean().index)
        battpower_series = pd.Series(battpower, index=pv_input.resample(str(settings['ramp_interval'])+'t', label='right').mean().index)
        curtail_series = pd.Series(curtail, index=pv_input.resample(str(settings['ramp_interval'])+'t', label='right').mean().index)

    
        fig, ax = plt.subplots()
        plt.plot(PV_ramp_interval, label='PV Power')
        plt.plot(outpower_series, label='Output Power')
        plt.plot(battery_soc_series, label='SOC')
        plt.plot(outpower_series[violation_series>0],'o', label='violations')
        plt.plot(battpower_series, label='Battery Power')
        plt.plot(curtail_series, label='Curtail Power')
        ax.legend(loc='upper right')
    
    total_energy = np.sum(outpower)*settings['ramp_interval']/60
    
    return violation_count, total_energy



    