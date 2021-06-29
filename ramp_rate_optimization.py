"""
Copyright (c) 2021, Electric Power Research Institute
 All rights reserved.
 Redistribution and use in source and binary forms, with or without modification,
 are permitted provided that the following conditions are met:
     * Redistributions of source code must retain the above copyright notice,
       this list of conditions and the following disclaimer.
     * Redistributions in binary form must reproduce the above copyright notice,
       this list of conditions and the following disclaimer in the documentation
       and/or other materials provided with the distribution.
     * Neither the name of DER-VET nor the names of its contributors
       may be used to endorse or promote products derived from this software
       without specific prior written permission.
 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import numpy as np
import pandas as pd
import ramp_rate_control



#sweeps battery sizes and finds # of violations for each. returns the violation count for the training set and the testing set seperately
def size_sweep(data, settings, battery_sweep_range): 
    battery_size = []
    violations_train = []
    violations_test = []  
    energy_output = []

    for battery_size_iter in battery_sweep_range:
        print("Size %.2f" % battery_size_iter)
        settings['battery_size'] = battery_size_iter
        violations_iter_train, violations_iter_test, params = optimize_params(data, settings)
        energy_output_iter = ramp_rate_control.run_smooth_controller(data.copy(), settings.copy(), 0, params[0], params[1], params[2], params[3])[1]
        violations_test.append(violations_iter_test)
        battery_size.append(battery_size_iter)
        violations_train.append(violations_iter_train)
        energy_output.append(energy_output_iter)
    return battery_size, violations_train, violations_test, energy_output

#for a given battery size and control settting - find the optimal parameters - return the parameters as well as the number of violations in the training and testing sets
def optimize_params(data, settings):
    #split the data into random, equal sized testing and training sets
    date_number_index = []
    for date_count in list(range(1,366)):
        date_number_index = np.append(date_number_index, date_count*np.ones(1440))
    date_number = pd.Series(date_number_index, data.index)
    training_days = np.random.choice(365, size=182, replace=False)
    training_set = data[date_number.isin(training_days)]
    testing_set = data[~date_number.isin(training_days)]
    training_set = training_set.reindex(index = data.index, fill_value=0)
    testing_set = testing_set.reindex(index = data.index, fill_value=0)     
    #optimize over four parameters
    cols = ['Vio','kp','ki','kf','soc_rest']
    kp_range = [0, 2]
    ki_range = [0, 2]
    #scale ki range by ramp interval - default range is for is 10 minutes
    ki_range = np.multiply(ki_range, settings['ramp_interval']/10)
    kf_range = [0, 8]
    soc_rest_range = [0.3, 0.7]
    if settings['curtail_as_control']:
        soc_rest_range = [0.99, 1]

    
    sections = 2 #2 is most efficient (binomial search)
    window_reduction_factor = 2 #between 1.001 (min) and 2 (max) for each iteration - 2 is aggressive and may move to a local minimum too fast. Too close to 1 will take long to converge
    max_iterations = 25 #maximum iterations before quitting optimization - optimum should be reached well before ~ 15 levels
    method = 2 # 1: pick best quadrant by top violation result, 2:(slightly better results found thus far) pick best quadrant by parameter average over all other variants
    continue_flag1 = 1
    continue_flag2 = 1
    violation_ave = 9999
    violation_by_level = []
    violation_by_level_test = []
    #of iterations equals sections^[(# of parameters)*(number of iterations)]
    for level in list(range(max_iterations)):
        if continue_flag2:
            print('Search level %.0f' % level)
            violations_iter = []
            violations_iter_test = []
            kp_step = (kp_range[1]-kp_range[0])/sections
            ki_step = (ki_range[1]-ki_range[0])/sections
            kf_step = (kf_range[1]-kf_range[0])/sections
            soc_rest_step = (soc_rest_range[1]-soc_rest_range[0])/sections
            for soc_rest_value in np.arange(soc_rest_range[0], soc_rest_range[0]+soc_rest_step*sections, soc_rest_step) + soc_rest_step/2:
                kf_value = 0#for kf_value in np.arange(kf_range[0], kf_range[0]+kf_step*sections, kf_step) + kf_step/2:
                if 1:
                    for ki_value in np.arange(ki_range[0], ki_range[0]+ki_step*sections, ki_step) + ki_step/2:
                        for kp_value in np.arange(kp_range[0], kp_range[0]+kp_step*sections, kp_step) + kp_step/2:
                            #train
                            violation_temp = ramp_rate_control.run_smooth_controller(training_set, settings.copy(), 0, kp=kp_value, ki=ki_value, kf=kf_value, soc_rest=soc_rest_value)[0]
                            print("Optimization Run: Violations: " + str(violation_temp))
                            violations_iter.append([violation_temp, kp_value, ki_value, kf_value, soc_rest_value])
                            violation_temp_test = ramp_rate_control.run_smooth_controller(testing_set, settings, 0, kp=kp_value, ki=ki_value, kf=kf_value, soc_rest=soc_rest_value)[0]
                            violations_iter_test.append(violation_temp_test)
            result_df = pd.DataFrame(violations_iter, columns=cols)
            if result_df['Vio'].mean() < violation_ave: #if this level is better than the previous one
#                Method 1: pick the best row - found to yield a slightly less optimum value on average than method 2 (more testing needed)
                if method == 1:
                    violation_ave = result_df['Vio'].mean()
                    best_row = result_df[result_df['Vio'] == result_df['Vio'].min()]
            
                    kp_best = best_row['kp'].values[0]
                    ki_best = best_row['ki'].values[0]
                    kf_best = best_row['kf'].values[0]
                    soc_rest_best = best_row['soc_rest'].values[0]
                    
    #           Method 2: pick each parameter high/low based on the average of all combinations for that parameter
                if method == 2:
                    violation_ave = result_df['Vio'].mean()
                    best_row = result_df[result_df['Vio'] == result_df['Vio'].min()]
                    if result_df[result_df['kp']<result_df['kp'].mean()]['Vio'].mean() < result_df[result_df['kp']>result_df['kp'].mean()]['Vio'].mean():
                        kp_best = result_df[result_df['kp']<result_df['kp'].mean()].kp.mean()
                    else:
                        kp_best = result_df[result_df['kp']>result_df['kp'].mean()].kp.mean()
                    if result_df[result_df['ki']<result_df['ki'].mean()]['Vio'].mean() < result_df[result_df['ki']>result_df['ki'].mean()]['Vio'].mean():
                        ki_best = result_df[result_df['ki']<result_df['ki'].mean()].ki.mean()
                    else:
                        ki_best = result_df[result_df['ki']>result_df['ki'].mean()].ki.mean()
                    if result_df[result_df['kf']<result_df['kf'].mean()]['Vio'].mean() < result_df[result_df['kf']>result_df['kf'].mean()]['Vio'].mean():
                        kf_best = result_df[result_df['kf']<result_df['kf'].mean()].kf.mean()
                    else:
                        kf_best = result_df[result_df['kf']>result_df['kf'].mean()].kf.mean()
                    if result_df[result_df['soc_rest']<result_df['soc_rest'].mean()]['Vio'].mean() < result_df[result_df['soc_rest']>result_df['soc_rest'].mean()]['Vio'].mean():
                        soc_rest_best = result_df[result_df['soc_rest']<result_df['soc_rest'].mean()].soc_rest.mean()
                    else:
                        soc_rest_best = result_df[result_df['soc_rest']>result_df['soc_rest'].mean()].soc_rest.mean()
            
                kp_range = [kp_best-(kp_step/window_reduction_factor), kp_best+(kp_step/window_reduction_factor)]
                ki_range = [ki_best-(ki_step/window_reduction_factor), ki_best+(ki_step/window_reduction_factor)]
                kf_range = [kf_best-(kf_step/window_reduction_factor), kf_best+(kf_step/window_reduction_factor)]
                soc_rest_range = [soc_rest_best-(soc_rest_step/window_reduction_factor), soc_rest_best+(soc_rest_step/window_reduction_factor)]
#                print('New Optimum')
#                print('Best: %.0f' % best_row['Vio'].multiply(1+settings['train/test']).values[0])
#                print('Average: %.0f' % (violation_ave*2))
                #print('Testing set average: %.0f' % np.mean(violations_iter_test))
                violation_by_level_test.append(np.mean(violations_iter_test))
                violation_by_level.append(violation_ave)
                continue_flag1 = 1
                #print([kp_best, ki_best, kf_best, soc_rest_best])
            else: #try to narrow window around the existing parameters
                if continue_flag1:
#                    print('Trying narrower window')
                    continue_flag1 = 0
                    kp_range = [kp_best-(kp_step/window_reduction_factor), kp_best+(kp_step/window_reduction_factor)]
                    ki_range = [ki_best-(ki_step/window_reduction_factor), ki_best+(ki_step/window_reduction_factor)]
                    kf_range = [kf_best-(kf_step/window_reduction_factor), kf_best+(kf_step/window_reduction_factor)]
                    soc_rest_range = [soc_rest_best-(soc_rest_step/window_reduction_factor), soc_rest_best+(soc_rest_step/window_reduction_factor)]
                else:
                    continue_flag2 = 0
#                    print('Optimization iteration level worse than previous')
#                    print('Optimal Parameters')
#                    print("level %.0f" % level)
                    print([kp_best, ki_best, kf_best, soc_rest_best, np.min(violations_iter_test)*2])

    test_min = np.min(violations_iter_test)*2
    train_min = best_row['Vio'].values[0]*2
    
    return train_min, test_min, [kp_best, ki_best, kf_best, soc_rest_best]

    
