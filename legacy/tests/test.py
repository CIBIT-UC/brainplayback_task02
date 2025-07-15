# %%
import pandas as pd

events_file = '/users3/uccibit/alexsayal/BIDS-BRAINPLAYBACK-TASK2/sub-01/ses-01/func/sub-01_ses-01_task-02a_run-1_events.tsv'

events = pd.read_csv(events_file, sep='\t')

# round onset and duration to integer
events['onset'] = events['onset'].round(0).astype(int)
events['duration'] = events['duration'].round(0).astype(int)

# # Identify all Noise trials which duration is 6 seconds and remove them
intersong_trials = events.query("trial_type == 'Noise' and duration > 5.5 and duration < 6.5")

# # rename noise_trials to 'intersong'
events.loc[intersong_trials.index, "trial_type"] = "Intersong"

# # remove all 'intersong' trials
events = events[events.trial_type != 'Intersong']

events.reset_index(drop=True, inplace=True)

print(events)

trial_type_counter = {trial_type: 0 for trial_type in events['trial_type'].unique() if trial_type != 'Noise'}

for ii in range(len(events)):
    cname = events['trial_type'][ii]
    if cname != 'Noise':
        # update counter
        trial_type_counter[cname] += 1
        events['trial_type'][ii] = cname + str(trial_type_counter[cname]).zfill(2)

print(events)

# %%
trialwise_conditions = events["trial_type"].unique()

# remove Noise from trialwise_conditions
trialwise_conditions = [x for x in trialwise_conditions if "Noise" not in x]

print(trialwise_conditions)
# %%
for contrast in trialwise_conditions:
    print(f'{contrast} - Noise')



# %%
