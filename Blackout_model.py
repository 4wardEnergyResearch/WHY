# -*- coding: utf-8 -*-
"""

    Blackout model
    Copyright (C) 2023 4ward Energy Research GmbH
    office@4wardenergy.at
    The full licence is found in COPYING.txt


Created on Wed May 12 2023 16:40:35 

15 min profile in kWH

@author: 4ER Markus Baumann


"""
import pandas as pd
import numpy as np
from pandas import read_csv
from datetime import date, datetime, timedelta
import os.path
import time
import sys  # for stopping the interpreter with sys.exit()
import statistics  # for calculating average

# -------------------- SET PARAMETERS -----------------------------------------

storage_state_initial = 0                           # kWh
storage_capacity = 229.5                            # kWh 382.5 kWh mit Faktor 1.5, 229.5 kWh mit Faktor 0.9
storage_max_charge_rate = storage_capacity/8        # kWh/timestep bei 0.5 C
storage_max_discharge_rate = -storage_capacity/8    # kWh/timestep bei 0.5 C
storage_eff_charging = 0.95                         # -
storage_eff_discharging = 0.961                     # -
min_per_timestep = 15                               # min


# -------------------- DEVELOPER MODE -----------------------------------------

calculate_only_xx_blackout_scenarios = 0            # if 0: all cases are calculated
xx_blackout_scenarios = 1                           # Number of blackout cases to be calculated (recommended: 5) 480 files

export_results_to_csv = 1                           # --> 6 csv files per blackout scenario!
skip = 0                                            # skips input file reading and profile merging
create_storage_state_normal = 1                     # calculates storage state in normal mode
show_info_calc = 0                                  # prints detailed info of script

create_PV_generation_profiles = 0                   # creates profiles according to gen_list and 1kWp profile

# -------------------- FILE LOCATIONS -----------------------------------------

target_directory = os.getcwd()
export_directory = os.path.join(os.getcwd(), "out")

os.makedirs(export_directory, exist_ok=True)

# -------------------- FUNCTIONS ----------------------------------------------

# ---------------------------
# BESS storage (Battery Energy Storage System)
# ---------------------------


def calculate_soc(energy_demand, soc, max_capacity, max_charge_rate, max_discharge_rate, storage_eff_charging, storage_eff_discharging):
    """
    Simplified battery operation. Returns storage state of battery for a given
    energy input/demand as well as the resulting amount of energy which cannot
    be charged/discharged.
    All parameters must have same electrical energy unit, e.g. kWh/timestep.

    ----------
    energy_demand : float
        A positive value will result in charging attempt.
        A negative value will result in discharging attempt.
    soc : float
        Current state of charge.
    max_capacity : float
        Max. amount of energy that the battery can store.
    max_charge_rate : float
        Must be a positive value. Restricts max. amount that can be charged in timestep.
    max_discharge_rate : float
        Must be a negative value. Restricts max. amount that can be discharged in timestep.
    storage_eff_charging : float
        Must be between 0 - 1. (Stored energy in battery / Charged energy).
    storage_eff_discharging : float
        Must be between 0 - 1. (Useable energy / Discharged energy from battery).

    Returns
    -------
    out_storage_state : float
        State of charge after charging/discharging, capped by 0 or max_capacity.
    excess_energy : float
        Excess energy that cannot be charged/discharged due to constraints.

    """

    # DISCHARGING

    if (energy_demand < 0 and energy_demand >= max_discharge_rate):                # max discharging

        out_storage_state = soc + energy_demand/storage_eff_discharging
        excess_energy = 0

        if out_storage_state < 0:                                                  # empty battery check
            out_storage_state = 0
            excess_energy = energy_demand + soc*storage_eff_discharging

    elif (energy_demand < 0 and energy_demand < max_discharge_rate):               # capped discharging
        out_storage_state = soc + max_discharge_rate/storage_eff_discharging

        if out_storage_state < 0:                                                  # empty battery check
            out_storage_state = 0
            excess_energy = energy_demand - max_discharge_rate + out_storage_state 
            # max_discharge_rate + soc # old, falsch

        else:
            excess_energy = energy_demand - max_discharge_rate

    # CHARGING

    elif (energy_demand > 0 and energy_demand <= max_charge_rate):                    # full charge

        out_storage_state = soc + energy_demand*storage_eff_charging
        excess_energy = 0

        if out_storage_state > max_capacity:                                          # full battery check
            out_storage_state = max_capacity
            excess_energy = energy_demand - (max_capacity-soc)/storage_eff_charging

    elif (energy_demand > 0 and energy_demand > max_charge_rate):                     # capped charging
        out_storage_state = soc + max_charge_rate*storage_eff_charging

        if out_storage_state > max_capacity:                                          # full battery check
            out_storage_state = max_capacity
            excess_energy = energy_demand - (max_capacity-soc)/storage_eff_charging
        else:
            excess_energy = energy_demand - max_charge_rate

    else:                                                                             # all 0, no changes
        out_storage_state = soc
        excess_energy = energy_demand

    return out_storage_state, excess_energy


# ---------------------------
# Counter of consecutive events
# ---------------------------


def count_consecutive_events(series, event):
    """
    Creates a list where successive events are counted. Interruptions set counter to 0.

    Parameters
    ----------
    series : series
        Input series.
    event : int/str
        Event that increases the counter.

    Returns
    -------
    count_list : list[int]
        List of successive events.

    """

    count_list = []
    count = 0

    for index, value in enumerate(series):
        if value == event:
            count += 1
            count_list.append(count)
        else:
            count = 0
            count_list.append(count)

    return count_list


# ---------------------------
# Return average duration of sequence
# ---------------------------


def count_avg_deactivation_time(series):
    """
    Returns highest value of ascening sequences of numbers in list.
    Only checks if successive values are ascending.

    Parameters
    ----------
    series : list[int]
        Ascending sequences of numbers must have increment of 1, otherwise useless output.

    Returns
    -------
    count_list : list[int]
        List of highest number of ascending sequences.

    """

    count_list = []

    for index, value in enumerate(series):

        if index < len(series)-1:

            if value > series[index + 1]:
                count_list.append(value)
        else:
            if value > series[index - 1]:
                count_list.append(value)

    return count_list


# -----------------------------------------------
# ---- read input files
# -----------------------------------------------

timer_script_start = time.time()

if skip == 0:  # can be skipped if files are already loaded

    start_time = time.time()
    filename_import = os.path.join(target_directory, 'Blackout_INPUT_blackoutscenarios.csv')
    input_blackoutscenarios = pd.read_csv(filename_import, header=0, on_bad_lines='skip', delimiter=";", decimal=",")
    input_blackoutscenarios = input_blackoutscenarios.fillna(0)

    # read 1kWp PV Generation profile
    # then create scaled generation profile for usecase
    filename_import = os.path.join(target_directory, 'Blackout_INPUT_PVGenerationProfile1kWp.csv')
    PVgeneration_profile_1kWp = pd.read_csv(filename_import, header=0, on_bad_lines='skip', delimiter=";", decimal=",")
    PVgeneration_profile_1kWp = PVgeneration_profile_1kWp.fillna(0)

    # read load profiles
    filename_import = os.path.join(target_directory, 'Blackout_INPUT_consumer_profiles_normal.csv')
    print(target_directory)
    Cons_profiles_normal = pd.read_csv(filename_import, header=0, delimiter=";", decimal=",")

    filename_import = os.path.join(target_directory, 'Blackout_INPUT_consumer_profiles_blackout.csv')
    Cons_profiles_blackout = pd.read_csv(filename_import, header=0, delimiter=";", decimal=",")

    # read list of consumer profiles
    filename_import = os.path.join(target_directory, 'Blackout_INPUT_consumer_list_devices.csv')
    cons_list = pd.read_csv(filename_import, header=0, delimiter=";", decimal=",")

    # read list of installed generation capacities
    filename_import = os.path.join(target_directory, 'Blackout_INPUT_generation_list.csv')
    gen_list = pd.read_csv(filename_import, header=0, delimiter=";", decimal=",")

    filename_import = os.path.join(target_directory, 'Blackout_INPUT_PVgeneration_profiles.csv')
    current_usercase_generation = pd.read_csv(filename_import, header=0, delimiter=";", decimal=",")


    # convert all columns into desired type
    input_blackoutscenarios['Time'] = pd.to_datetime(input_blackoutscenarios['Time'], format='%d.%m.%Y %H:%M')

    PVgeneration_profile_1kWp['Time'] = pd.to_datetime(PVgeneration_profile_1kWp['Time'], format='%d.%m.%Y %H:%M')
    cols = PVgeneration_profile_1kWp.columns
    PVgeneration_profile_1kWp[cols[1:]] = PVgeneration_profile_1kWp[cols[1:]].apply(pd.to_numeric, errors='coerce')

    Cons_profiles_normal['Time'] = pd.to_datetime(Cons_profiles_normal['Time'], format='%d.%m.%Y %H:%M')
    cols = Cons_profiles_normal.columns
    Cons_profiles_normal[cols[1:]] = Cons_profiles_normal[cols[1:]].apply(pd.to_numeric, errors='coerce')

    Cons_profiles_blackout['Time'] = pd.to_datetime(Cons_profiles_blackout['Time'], format='%d.%m.%Y %H:%M')
    cols = Cons_profiles_blackout.columns
    Cons_profiles_blackout[cols[1:]] = Cons_profiles_blackout[cols[1:]].apply(pd.to_numeric, errors='coerce')

    current_usercase_generation['Time'] = pd.to_datetime(current_usercase_generation['Time'], format='%d.%m.%Y %H:%M')
    cols = current_usercase_generation.columns
    current_usercase_generation[cols[1:]] = current_usercase_generation[cols[1:]].apply(pd.to_numeric, errors='coerce')



# -----------------------------------------------
# ---- process parameters from input
# -----------------------------------------------

# -----------------------------------------------
# ---- Create list of blackout times from input
# -----------------------------------------------

if skip == 0:  # can be skipped if already created

    print('blackouttime creation')

    blackouttime = pd.DataFrame(columns=['blackoutstart', 'blackoutend'])

    for index, column in input_blackoutscenarios.iterrows():

        input_blackoutscenarios_duration = int(input_blackoutscenarios.loc[index, 'Duration'])
        input_blackoutscenarios_duration = timedelta(days=input_blackoutscenarios_duration)
        # print('duration =', input_blackoutscenarios_duration)

        input_blackoutscenarios_start = input_blackoutscenarios.loc[index, 'Time']
        # print('input_blackoutscenarios_start =', input_blackoutscenarios_start)
        input_blackoutscenarios_end = input_blackoutscenarios_start + input_blackoutscenarios_duration
        # print('input_blackoutscenarios_end =', input_blackoutscenarios_end)

        blackouttime.loc[index, 'blackoutstart'] = input_blackoutscenarios_start
        blackouttime.loc[index, 'blackoutend'] = input_blackoutscenarios_end

        # print('blackoutstart =', blackouttime.loc[index, 'blackoutstart'])
        # print('blackoutend =', blackouttime.loc[index, 'blackoutend'])

    blackouttime.to_csv(os.path.join(export_directory, r'blackouttimes.csv'), sep=';', decimal=",")
    print('CSV: blackouttimes.csv created.\n')

else:
    print('blackouttimes_generation off.\n')


# -----------------------------------------------
# ---- Create photovoltaics generation profiles
# -----------------------------------------------

if create_PV_generation_profiles == 1:

    # Add scaled generation profiles to GENERATION according to gen_list (generation_list)

    datetimecolumn_gen = PVgeneration_profile_1kWp.iloc[:, :1]
    current_usercase_generation = datetimecolumn_gen.copy(deep=True)

    counter_inserted_usercases_generation = 0

    for i, column in gen_list.iterrows():

        gen_name = column['name']
        gen_capacity = column['kWp']

        current_usercase_generation_transfer = PVgeneration_profile_1kWp.E1 * gen_capacity

        current_usercase_generation.insert(len(current_usercase_generation.columns),
                                           gen_name, current_usercase_generation_transfer, allow_duplicates=True)

        if show_info_calc == 1:
            print('Generation profile', gen_name, 'scaled to', gen_capacity, 'kWp and added', 'to current_usercase_generation')

        counter_inserted_usercases_generation += 1

    print(counter_inserted_usercases_generation, 'generation profile(s) added.')
    current_usercase_generation.to_csv(os.path.join(export_directory, r'current_usercase_generation.csv'),
                                       sep=';', decimal=".", index=False)
    print('CSV: current_usercase_generation.csv created.\n')


# -----------------------------------------------
# ---- Merge selected consumer profiles and photovoltaics
# -----------------------------------------------

# Create Dataframe with PV generation profile for normal and blackout scenario

datetimecolumn = Cons_profiles_normal.iloc[:, :1]
# nimmt Datetimecolumn von Profilsatz im Normalmodus - sollte gleich sein mit Profilsatz im Notmodus

current_usercase_normal = datetimecolumn.copy(deep=True)
current_usercase_blackout = datetimecolumn.copy(deep=True)

# Add consumer profiles to NORMAL scenario according to cons_list (consumer_list)

counter_inserted_usercases_normal = 0

for i, column in cons_list.iterrows():

    cons_profile = column['profile']
    cons_name = column['name']

    current_usercase_normal_transfer = Cons_profiles_normal.loc[:, cons_profile]

    if len(current_usercase_normal_transfer.index) != len(datetimecolumn.index):
        print('Number of rows of input_column not equal to used datetime_column ofcurrent_usercase_normal - please abort by pressing Ctrl+C')
        break

    current_usercase_normal.insert(len(current_usercase_normal.columns),
                                   cons_name, current_usercase_normal_transfer, allow_duplicates=True)

    if show_info_calc == 1:
        print(cons_profile, 'added as', cons_name, 'to current_usercase_normal')

    counter_inserted_usercases_normal = counter_inserted_usercases_normal + 1

print(counter_inserted_usercases_normal, 'consumer profile(s) to normal case added.')
current_usercase_normal.to_csv(os.path.join(export_directory, r'current_usercase_normal.csv'), sep=';', decimal=".", index=False)
print('CSV: current_usercase_normal.csv created.\n')

# Add consumer profiles to BLACKOUT scenario according to cons_list (consumer_list)

counter_inserted_usercases_blackout = 0

for i, column in cons_list.iterrows():

    cons_profile = column['profile']
    cons_name = column['name']

    current_usercase_blackout_transfer = Cons_profiles_blackout.loc[:, cons_name]

    if len(current_usercase_blackout_transfer.index) != len(datetimecolumn.index):
        print('Number of columns of input_conlumn not equal to used datetime_column of current_usercase_blackout - please abort by pressing Ctrl+C')
        break

    current_usercase_blackout.insert(len(current_usercase_blackout.columns), cons_name,
                                     current_usercase_blackout_transfer, allow_duplicates=True)

    if show_info_calc == 1:
        print(cons_profile, 'added as', cons_name, 'to current_usercase_blackout')

    counter_inserted_usercases_blackout = counter_inserted_usercases_blackout + 1

print(counter_inserted_usercases_blackout, 'consumer profile(s) to blackout case added.')
current_usercase_blackout.to_csv(os.path.join(export_directory, r'current_usercase_blackout.csv'), sep=';', decimal=".", index=False)
print('CSV: current_usercase_blackout.csv created.\n')


# -----------------------------------------------
# ---- Generation of SOC of storage NORMAL
# ---- Durchlauf mit cons in NORMALbetrieb um Speicherzustand zur Blackout-Startzeit zu ermitteln
# ---- if create_storage_state_normal = 1
# -----------------------------------------------


if create_storage_state_normal == 1:

    storage_state_calc = datetimecolumn.copy(deep=True)

    storage_state_normal = datetimecolumn.copy(deep=True)

    storage_state_gen = []
    res_energy_normal_gen = []

    if 'soc' in current_usercase_normal.columns:  # if skip == 1 then column already exists
        current_usercase_normal.drop(columns=['soc'])

    storage_state_calc['diff'] = current_usercase_generation.sum(
        axis=1, numeric_only=True) - current_usercase_normal.sum(axis=1, numeric_only=True)

    for i, column in storage_state_calc.iterrows():

        excess_energy_timestep = storage_state_calc.loc[i, 'diff']

        if i == 0:
            storage_state_input = storage_state_initial
            storage_state, residual_excess_energy_timestep = calculate_soc(
                excess_energy_timestep, storage_state_input, storage_capacity, storage_max_charge_rate, storage_max_discharge_rate,
                storage_eff_charging, storage_eff_discharging)
            storage_state_gen.append(storage_state)
            res_energy_normal_gen.append(residual_excess_energy_timestep)

        else:
            storage_state_input = storage_state_gen[i-1]
            storage_state, residual_excess_energy_timestep = calculate_soc(
                excess_energy_timestep, storage_state_input, storage_capacity, storage_max_charge_rate, storage_max_discharge_rate,
                storage_eff_charging, storage_eff_discharging)
            storage_state_gen.append(storage_state)
            res_energy_normal_gen.append(residual_excess_energy_timestep)

    storage_state_normal['soc'] = storage_state_gen

    print('Storage SOC profile in normal mode generated.')

    storage_state_normal.to_csv(os.path.join(export_directory, r'current_usercase_normal_soc.csv'), sep=';', decimal=".", index=False)
    print('CSV: current_usercase_normal_soc.csv created.')

else:
    print('No storage SOC profile in normal mode generated.')


# -----------------------------------------------
# ---- Work through each blackout scenario
# -----------------------------------------------


# loop over all indexes of blackoutlist

blackout_scenarios_count = 0

dynamic_blackout_range = current_usercase_blackout.copy(deep=True)
dynamic_blackout_range = dynamic_blackout_range.set_index('Time')

# Create lists for results
results_transfer_blackout_case_name = []
results_transfer_blackout_start = []
results_transfer_blackout_end = []
results_transfer_blackout_soc_init = []
results_transfer_blackout_soc_end = []

results_transfer_blackout_total_deactivations_allcons = []
results_transfer_blackout_longest_deactivation_cons = []
results_transfer_blackout_mean_deactivation_cons = []
results_transfer_blackout_share_of_supply_allcons = []

results_transfer_blackout_total_deactivations_allgen = []
results_transfer_blackout_longest_deactivation_gen = []
results_transfer_blackout_mean_deactivation_gen = []
results_transfer_blackout_share_of_supply_allgen = []

results_transfer_blackout_kWh_curtail_allcons = []
results_transfer_blackout_kWh_curtail_allgen = []
results_transfer_blackout_kWh_allcons = []
results_transfer_blackout_kWh_allgen = []

# for i, column in blackouttime.iterrows():
if calculate_only_xx_blackout_scenarios == 1:

    x = xx_blackout_scenarios       # Reduced number of blackout cases to calculate if activated

else:

    x = len(blackouttime.index)     # Number of blackout cases to calculate --> All

for i in range(x):

    timer_blackout_scen_start = time.time()

    print('----------------------------------')
    print('Current blackout case = ', blackouttime.loc[i, 'blackoutstart'], blackouttime.loc[i, 'blackoutend'])

    # Takes current blackout case according to blackout start and end time

    start = blackouttime.loc[i, 'blackoutstart']
    end = blackouttime.loc[i, 'blackoutend']

    cut_start = current_usercase_blackout['Time'] >= start
    cut_end = current_usercase_blackout['Time'] < end

    blackout_scen_gen = current_usercase_generation.loc[cut_start & cut_end]
    blackout_scen_cons = current_usercase_blackout.loc[cut_start & cut_end]
    blackout_scen_soc = storage_state_normal.loc[cut_start & cut_end]

    datacheck_datetimecolumn = pd.DataFrame()
    datacheck_datetimecolumn['gen'] = blackout_scen_gen['Time']
    datacheck_datetimecolumn['cons'] = blackout_scen_cons['Time']
    datacheck_datetimecolumn['soc'] = blackout_scen_soc['Time']

    if datacheck_datetimecolumn.eq(datacheck_datetimecolumn.iloc[:, 0], axis=0).all().all():
        print('\nDATACHECK OK: All datetime columns contain the same dates. \n')
    else:
        sys.exit('\nDATACHECK WARNING: Not all datetime columns contain the same dates. \n')

    blackout_scen_gen = blackout_scen_gen.reset_index(drop=True)
    blackout_scen_cons = blackout_scen_cons.reset_index(drop=True)
    blackout_scen_soc = blackout_scen_soc.reset_index(drop=True)

    blackout_scen_socdeact_filename = str(datetime.strftime(start, '%Y-%m-%d %H'+' Uhr'))

    # Takes storage state from starting time of blackout

    storage_state_initial_blackout = blackout_scen_soc.loc[0, 'soc']

    # For list of result
    results_transfer_blackout_case_name.append(blackout_scen_socdeact_filename)
    results_transfer_blackout_start.append(start)
    results_transfer_blackout_end.append(end)
    results_transfer_blackout_soc_init.append(storage_state_initial_blackout)

    print('Blackout storage SOC init = ', storage_state_initial_blackout, 'kWh.')

    # -------------------------------------------------
    # Calculate blackoutcase with storage and consumer deactivation
    # -------------------------------------------------

    socdeact_time = blackout_scen_cons['Time']
    socdeact_gen = blackout_scen_gen.drop(columns=['Time'])
    socdeact_cons = blackout_scen_cons.drop(columns=['Time'])

    blackout_scen_deact_cons_curtail = pd.DataFrame().reindex_like(socdeact_cons).fillna(0)
    blackout_scen_deact_cons_eval_df = pd.DataFrame().reindex_like(socdeact_cons).fillna(0)

    blackout_scen_deact_gen_curtail = pd.DataFrame().reindex_like(socdeact_gen).fillna(0)
    blackout_scen_deact_gen_eval_df = pd.DataFrame().reindex_like(socdeact_gen).fillna(0)

    socdeact_storage_state_initial_blackout = blackout_scen_soc.loc[0, 'soc']

    socdeact_soc = []

    # Counter
    counter_deact_cons = 0
    counter_deact_gen = 0

    for n, column in blackout_scen_cons.iterrows():

        current_row_gen = socdeact_gen.iloc[n]
        current_row_cons = socdeact_cons.iloc[n]

        excess_energy_timestep = np.sum(current_row_gen) - np.sum(current_row_cons)

        if n == 0:
            storage_state_initial_input = socdeact_storage_state_initial_blackout

            storage_state, residual_excess_energy_timestep = calculate_soc(
                excess_energy_timestep, storage_state_initial_input, storage_capacity, storage_max_charge_rate,
                storage_max_discharge_rate, storage_eff_charging, storage_eff_discharging)
        else:
            storage_state_initial_input = socdeact_soc[n-1]
            storage_state, residual_excess_energy_timestep = calculate_soc(
                excess_energy_timestep, storage_state_initial_input, storage_capacity, storage_max_charge_rate,
                storage_max_discharge_rate, storage_eff_charging, storage_eff_discharging)

        while residual_excess_energy_timestep != 0:
            if n == 0:
                storage_state_initial_input = socdeact_storage_state_initial_blackout

                storage_state, residual_excess_energy_timestep = calculate_soc(
                    excess_energy_timestep, storage_state_initial_input, storage_capacity, storage_max_charge_rate,
                    storage_max_discharge_rate, storage_eff_charging, storage_eff_discharging)
            else:
                storage_state_initial_input = socdeact_soc[n-1]
                storage_state, residual_excess_energy_timestep = calculate_soc(
                    excess_energy_timestep, storage_state_initial_input, storage_capacity, storage_max_charge_rate,
                    storage_max_discharge_rate, storage_eff_charging, storage_eff_discharging)

            if residual_excess_energy_timestep < 0:

                # DEACT CONSUMER

                current_row_gen = socdeact_gen.iloc[n]
                current_row_cons = socdeact_cons.iloc[n]
                current_row_index = n
                highest_consumer_index = socdeact_cons.iloc[n].idxmax()
                highest_consumer_value = socdeact_cons.iloc[n].max()

                # print('Deactivate consumer:', highest_consumer_index, ', excess =', highest_consumer_value, 'kWh in row', current_row_index)

                blackout_scen_deact_cons_curtail.at[current_row_index, highest_consumer_index] = highest_consumer_value
                blackout_scen_deact_cons_eval_df.at[current_row_index, highest_consumer_index] = 1  # write flag in evaluation DataFrame
                socdeact_cons.at[current_row_index, highest_consumer_index] = 0

                counter_deact_cons += 1

                excess_energy_timestep = np.sum(current_row_gen) - np.sum(current_row_cons)
                storage_state, residual_excess_energy_timestep = calculate_soc(
                    excess_energy_timestep, storage_state_initial_input, storage_capacity, storage_max_charge_rate,
                    storage_max_discharge_rate, storage_eff_charging, storage_eff_discharging)
                # print(counter_deact_cons, 'consumer deactivated.')

            if residual_excess_energy_timestep > 0:

                # DEACT GENERATOR

                current_row_gen = socdeact_gen.iloc[n]
                current_row_cons = socdeact_cons.iloc[n]
                current_row_index = n
                lowest_gen_index = current_row_gen[current_row_gen > 0].idxmin()
                lowest_gen_value = current_row_gen[current_row_gen > 0].min()

                # print('Deactivate Generator:', lowest_gen_index, ', excess =', lowest_gen_value, 'kWh in row', current_row_index)

                blackout_scen_deact_gen_curtail.at[current_row_index, lowest_gen_index] = lowest_gen_value
                blackout_scen_deact_gen_eval_df.at[current_row_index, lowest_gen_index] = 1  # write flag in evaluation DataFrame
                socdeact_gen.at[current_row_index, lowest_gen_index] = 0

                counter_deact_gen += 1

                excess_energy_timestep = np.sum(current_row_gen) - np.sum(current_row_cons)
                storage_state, residual_excess_energy_timestep = calculate_soc(
                    excess_energy_timestep, storage_state_initial_input, storage_capacity, storage_max_charge_rate,
                    storage_max_discharge_rate, storage_eff_charging, storage_eff_discharging)
                # print(counter_deact_gen, 'generator deactivated.')

        socdeact_soc.append(storage_state)

    print(counter_deact_cons, 'consumer deactivation(s).')
    print(counter_deact_gen, 'generator deactivation(s). \n')

    # ---------------------
    # EVALUATION
    # ---------------------

    # EACH CONSUMER

    eval_scen_cons_deact_sequences = []

    for cons_i in blackout_scen_deact_cons_eval_df:

        column = cons_i

        # cons-columns as series for function input
        series = pd.Series(blackout_scen_deact_cons_eval_df[column])
        event = 1                                   # 1 = deactivated consumer
        list_deact_cons = count_consecutive_events(series, event)
        list_deact_cons_duration = count_avg_deactivation_time(list_deact_cons)

        eval_cons_longest_deact_sequence = round(max(list_deact_cons), 0)
        eval_cons_total_deact_count = round(sum(series), 0)
        eval_cons_share_deact = round((sum(series)/len(list_deact_cons))*100, 2)
        eval_scen_cons_deact_sequences.extend(list_deact_cons_duration)
        if list_deact_cons_duration:                # If list contains values, then proceed
            eval_cons_deact_mean = round(statistics.fmean(list_deact_cons_duration), 1)
        else:
            eval_cons_deact_mean = 'empty'

        # print results
        if show_info_calc == 1:
            print('eval of', cons_i, ': deact. count =', eval_cons_total_deact_count, '| deact. share =', eval_cons_share_deact, "%")
            print('     longest sequence =', eval_cons_longest_deact_sequence, '| mean =', eval_cons_deact_mean)

    # EACH GENERATOR

    eval_scen_gen_deact_sequences = []

    for gen_i in blackout_scen_deact_gen_eval_df:

        column = gen_i

        # gen-columns as series for function input
        series = pd.Series(blackout_scen_deact_gen_eval_df[column])
        event = 1           # 1 = deactivated generator
        list_deact_gen = count_consecutive_events(series, event)
        list_deact_gen_duration = count_avg_deactivation_time(list_deact_gen)

        eval_gen_longest_deact_sequence = round(max(list_deact_gen), 0)
        eval_gen_total_deact_count = round(sum(series), 0)
        eval_gen_share_deact = round((sum(series)/len(list_deact_gen))*100, 2)
        eval_scen_gen_deact_sequences.extend(list_deact_gen_duration)
        if list_deact_gen_duration:  # If list contains values, then proceed
            eval_gen_deact_mean = round(statistics.fmean(list_deact_gen_duration), 1)
        else:
            eval_gen_deact_mean = 'empty'

        # print results
        if show_info_calc == 1:
            print('eval of', gen_i, ': deact. count =', eval_gen_total_deact_count, '| deact. share =', eval_gen_share_deact, "%")
            print('     longest sequence =', eval_gen_longest_deact_sequence, '| mean =', eval_gen_deact_mean)

    # Sum of deact
    blackout_scen_deact_eval_total_deact_gen = blackout_scen_deact_gen_eval_df.sum()                # generator total deactivations list
    blackout_scen_deact_eval_total_deact_allgen = blackout_scen_deact_eval_total_deact_gen.sum()    # total generator deactivations
    blackout_scen_deact_eval_total_deact_cons = blackout_scen_deact_cons_eval_df.sum()              # consumer total deactivations list
    blackout_scen_deact_eval_total_deact_allcons = blackout_scen_deact_eval_total_deact_cons.sum()  # total consumer deactivations

    # Sum of provided/shed loads
    blackout_scen_eval_kWh_curtail_cons = blackout_scen_deact_cons_curtail.sum()        # generator total deactivations list
    blackout_scen_eval_kWh_curtail_allcons = blackout_scen_eval_kWh_curtail_cons.sum()  # total generator deactivations

    blackout_scen_eval_kWh_curtail_gen = blackout_scen_deact_gen_curtail.sum()          # generator total deactivations list
    blackout_scen_eval_kWh_curtail_allgen = blackout_scen_eval_kWh_curtail_gen.sum()    # total generator deactivations

    blackout_scen_eval_kWh_cons = socdeact_cons.sum()                                   # total energy supplied, consumer list, in kWh
    blackout_scen_eval_kWh_allcons = blackout_scen_eval_kWh_cons.sum()                  # total energy supplied, all consumers, in kWh

    blackout_scen_eval_kWh_gen = socdeact_gen.sum()                                     # total energy produced, generator list, in kWh
    blackout_scen_eval_kWh_allgen = blackout_scen_eval_kWh_gen.sum()                    # total energy produced, all generators, in kWh

    # Overall results

    results_transfer_blackout_soc_end.append(socdeact_soc[-1])  # last element of SOC list

    results_transfer_blackout_total_deactivations_allcons.append(blackout_scen_deact_eval_total_deact_allcons)
    results_transfer_blackout_longest_deactivation_cons.append(max(blackout_scen_deact_eval_total_deact_cons)*min_per_timestep)
    results_transfer_blackout_share_of_supply_allcons.append(
        1-(blackout_scen_deact_eval_total_deact_allcons/blackout_scen_deact_cons_eval_df.size))

    if eval_scen_cons_deact_sequences:                          # If list contains values, then proceed
        results_transfer_blackout_mean_deactivation_cons.append(statistics.fmean(eval_scen_cons_deact_sequences)*min_per_timestep)
    else:
        results_transfer_blackout_mean_deactivation_cons.append(0)

    results_transfer_blackout_total_deactivations_allgen.append(blackout_scen_deact_eval_total_deact_allgen)
    results_transfer_blackout_longest_deactivation_gen.append(max(blackout_scen_deact_eval_total_deact_gen)*min_per_timestep)
    results_transfer_blackout_share_of_supply_allgen.append(
        1-(blackout_scen_deact_eval_total_deact_allgen/blackout_scen_deact_gen_eval_df.size))

    if eval_scen_gen_deact_sequences:                           # If list contains values, then proceed
        results_transfer_blackout_mean_deactivation_gen.append(statistics.fmean(eval_scen_gen_deact_sequences)*min_per_timestep)
    else:
        results_transfer_blackout_mean_deactivation_gen.append(0)

    results_transfer_blackout_kWh_curtail_allcons.append(blackout_scen_eval_kWh_curtail_allcons)
    results_transfer_blackout_kWh_curtail_allgen.append(blackout_scen_eval_kWh_curtail_allgen)
    results_transfer_blackout_kWh_allcons.append(blackout_scen_eval_kWh_allcons)
    results_transfer_blackout_kWh_allgen.append(blackout_scen_eval_kWh_allgen)

    print(i+1, '/', x, "scenarios calculated.")

    # to csv
    if export_results_to_csv == 1:

        socdeact_cons['soc'] = socdeact_soc

        socdeact_cons.insert(0, 'Time', socdeact_time)
        socdeact_cons.to_csv(os.path.join(export_directory, r'blackout_scen_socdeact ' +
                             blackout_scen_socdeact_filename + '_cons.csv'), sep=';', decimal=".", index=False)
        print('----------------------------------')
        print('CSV: blackout_scen_deact_cons', blackout_scen_socdeact_filename, '.csv created.')

        blackout_scen_deact_cons_curtail.insert(0, 'Time', socdeact_time)
        blackout_scen_deact_cons_curtail.to_csv(os.path.join(export_directory, r'blackout_scen_socdeact ' +
                                                blackout_scen_socdeact_filename + '_cons_curtailed.csv'), sep=';', decimal=".", index=False)
        print('CSV: blackout_scen_deact_cons_curtail', blackout_scen_socdeact_filename, '.csv created.')

        socdeact_gen.insert(0, 'Time', socdeact_time)
        socdeact_gen.to_csv(os.path.join(export_directory, r'blackout_scen_socdeact ' +
                            blackout_scen_socdeact_filename + '_gen.csv'), sep=';', decimal=".", index=False)
        print('----------------------------------')
        print('CSV: blackout_scen_deact_gen', blackout_scen_socdeact_filename, '.csv created.')

        blackout_scen_deact_gen_curtail.insert(0, 'Time', socdeact_time)
        blackout_scen_deact_gen_curtail.to_csv(os.path.join(export_directory, r'blackout_scen_socdeact ' +
                                               blackout_scen_socdeact_filename + '_gen_curtailed.csv'), sep=';', decimal=".", index=False)
        print('CSV: blackout_scen_deact_cons_curtail', blackout_scen_socdeact_filename, '.csv created.')

        blackout_scen_deact_gen_eval_df.insert(0, 'Time', socdeact_time)
        blackout_scen_deact_gen_eval_df.to_csv(os.path.join(export_directory, r'blackout_scen_socdeact ' +
                                               blackout_scen_socdeact_filename + '_gen_eval.csv'), sep=';', decimal=".", index=False)
        print('CSV: blackout_scen_deact_gen_eval', blackout_scen_socdeact_filename, '.csv created.')

        blackout_scen_deact_cons_eval_df.insert(0, 'Time', socdeact_time)
        blackout_scen_deact_cons_eval_df.to_csv(os.path.join(export_directory, r'blackout_scen_socdeact ' +
                                                blackout_scen_socdeact_filename + '_cons_eval.csv'), sep=';', decimal=".", index=False)
        print('CSV: blackout_scen_deact_cons_eval', blackout_scen_socdeact_filename, '.csv created.')

    # counter and timer
    blackout_scenarios_count = blackout_scenarios_count + 1
    timer_blackout_scen_end = time.time()

    print('Elapsed time for blackout scenario =',
          round(timer_blackout_scen_end - timer_blackout_scen_start, 1), 's.\n')


# merge results

results_blackout_scenarios = pd.DataFrame(data=results_transfer_blackout_case_name, columns=['Blackout_case'])
results_blackout_scenarios['Start'] = results_transfer_blackout_start
results_blackout_scenarios['End'] = results_transfer_blackout_end
results_blackout_scenarios['SOC init [kWh]'] = results_transfer_blackout_soc_init
results_blackout_scenarios['SOC end [kWh]'] = results_transfer_blackout_soc_end

results_blackout_scenarios['Cons: Number of deactivations [-]'] = results_transfer_blackout_total_deactivations_allcons
results_blackout_scenarios['Cons: Longest deactivation [min]'] = results_transfer_blackout_longest_deactivation_cons
results_blackout_scenarios['Cons: Mean duration of deactivations [min]'] = results_transfer_blackout_mean_deactivation_cons
results_blackout_scenarios['Cons: Share of supply [-]'] = results_transfer_blackout_share_of_supply_allcons
results_blackout_scenarios['Cons: kWh supplied [kWh]'] = results_transfer_blackout_kWh_allcons
results_blackout_scenarios['Cons: kWh shed [kWh]'] = results_transfer_blackout_kWh_curtail_allcons

results_blackout_scenarios['Gen: Number of deactivations [-]'] = results_transfer_blackout_total_deactivations_allgen
results_blackout_scenarios['Gen: Longest deactivation [min]'] = results_transfer_blackout_longest_deactivation_gen
results_blackout_scenarios['Gen: Mean duration of deactivations [min]'] = results_transfer_blackout_mean_deactivation_gen
results_blackout_scenarios['Gen: Share of supply [-]'] = results_transfer_blackout_share_of_supply_allgen
results_blackout_scenarios['Gen: kWh produced [kWh]'] = results_transfer_blackout_kWh_allgen
results_blackout_scenarios['Gen: kWh curtailed [kWh]'] = results_transfer_blackout_kWh_curtail_allgen

print('----------------------------------')
print(blackout_scenarios_count, 'blackout case(s) calculated.')

# write result file to csv
info_storage_capacity = "storage_capacity = " + str(storage_capacity)
results_blackout_scenarios[info_storage_capacity] = ""

export_timestamp = datetime.now()
export_str_date_time = export_timestamp.strftime("%Y-%m-%d %H-%M-%S")
export_time = str('export: ' + export_str_date_time)
results_blackout_scenarios[export_time] = ""

results_blackout_scenarios.to_csv(os.path.join(export_directory, r'results_blackout_scenarios ' +
                                  export_str_date_time + '.csv'), sep=';', decimal=".", index=False)
print('results_blackout_scenarios', export_str_date_time, '.csv created.')


timer_script_end = time.time()
print('Finished script in', round(timer_script_end - timer_script_start, 0), 's.')
print('Export folder: ', export_directory)
