import os
import numpy as np
import json
import matplotlib.pyplot as plt
from nussl import nussl as nussl_dev
import csv
from copy import deepcopy
from scipy.optimize import minimize


class SharedResourceManager:
    def __init__(self):
        self.reference_waveform_dict = {}
        self.estimate_filepaths = {}
        self.reference_waveform_cache = {}
        self.estimate_waveform_cache = {}
        self.estimate_key_cache = {}
        self.median_subjective_scores = {}
        self.sdr_json_cache = {}

    def load_references(self):
        assert self.median_subjective_scores
        if not self.reference_waveform_dict:
            self.reference_waveform_dict = load_all_reference_waveforms(self.median_subjective_scores)
        return self.reference_waveform_dict

    def load_subjective_data(self):
        if not self.median_subjective_scores:
            subjective_row_dicts = load_subjective_data()
            subjective_scores, self.median_subjective_scores = calculate_median_score_by_song_target_stimulus(
                subjective_row_dicts)
        return self.median_subjective_scores

    def get_estimate_match(self, track_name, target, estimates_filepaths_dict):
        cache_key = (track_name, target)
        if cache_key not in self.estimate_key_cache:
            self.estimate_key_cache[cache_key] = filter_dict_by_track_target(track_name, target, estimates_filepaths_dict)
        return self.estimate_key_cache[cache_key]

    def get_estimate_audio(self, filepath):
        if filepath not in self.estimate_waveform_cache:
            self.estimate_waveform_cache[filepath] = nussl_dev.AudioSignal(filepath)
        return self.estimate_waveform_cache[filepath]

    def get_reference_waveform(self, track_name, target, sample_offset, split_filepath='30s_previews.csv'):
        cache_key = (track_name, target, sample_offset)
        if cache_key in self.reference_waveform_cache:
            return self.reference_waveform_cache[cache_key]
        else:
            waveforms = self.reference_waveform_dict[(track_name, target, 'reference')]
            new_waveforms, track_start_time, track_stop_time = trim_waveforms(waveforms, track_name,
                                                                              split_filepath=split_filepath,
                                                                              sample_offset=sample_offset)
            return_tuple = (new_waveforms, track_start_time, track_stop_time)
            self.reference_waveform_cache[cache_key] = return_tuple
            return return_tuple

    def get_submission_filepaths(self, median_subjective_scores):
        if not self.estimate_filepaths:
            self.estimate_filepaths = get_submission_filepaths(median_subjective_scores)
        return self.estimate_filepaths

    def get_get_sdr_from_json_submission(self, path_to_json, estimate_method,
                                         track_name, target, track_start_time, track_stop_time):
        key = (path_to_json, estimate_method,
               track_name, target, track_start_time, track_stop_time)
        if key not in self.sdr_json_cache:
            self.sdr_json_cache[key] = get_sdr_from_json_submission('sigsepresults/sigsep-mus-2018/submissions/',
                                                                    estimate_method, track_name, target,
                                                                    track_start_time, track_stop_time)
        return self.sdr_json_cache[key]


the_resources = SharedResourceManager()


def _report_wsdr(approach_name, scores, aggregator=np.nanmedian, sample_start=None, sample_end=None):
    SDR = {}
    WSDR = {}
    print(approach_name)
    print(''.join(['-' for i in range(len(approach_name))]))

    if sample_start is not None or sample_end is not None:
        for j, key in enumerate(scores):
            if key not in ['combination', 'permutation']:
                SDR[key] = aggregator(scores[key]['SDR'][sample_start:sample_end])
                WSDR[key] = aggregator(scores[key]['ISR'][sample_start:sample_end])
                print(f'{key} SDR: {SDR[key]:.2f} dB')
                print(f'{key} WSDR: {WSDR[key]:.2f} dB')
                print()
    else:
        for j, key in enumerate(scores):
            if key not in ['combination', 'permutation']:
                SDR[key] = aggregator(scores[key]['SDR'])
                WSDR[key] = aggregator(scores[key]['ISR'])
                print(f'{key} SDR: {SDR[key]:.2f} dB')
                print(f'{key} WSDR: {WSDR[key]:.2f} dB')
    print('')
    return SDR, WSDR


def _report_sdr(approach_name, scores, aggregator=np.nanmedian, sample_start=None, sample_end=None):
    SDR = {}
    SIR = {}
    SAR = {}
    print(approach_name)
    print(''.join(['-' for i in range(len(approach_name))]))

    if sample_start is not None or sample_end is not None:
        for j, key in enumerate(scores):
            if key not in ['combination', 'permutation']:
                SDR[key] = aggregator(scores[key]['SDR'][sample_start:sample_end])
                SIR[key] = aggregator(scores[key]['SIR'][sample_start:sample_end])
                SAR[key] = aggregator(scores[key]['SAR'][sample_start:sample_end])
                print(f'{key} SDR: {SDR[key]:.2f} dB')
                print(f'{key} SIR: {SIR[key]:.2f} dB')
                print(f'{key} SAR: {SAR[key]:.2f} dB')
                print()
        print()
    else:
        for j, key in enumerate(scores):
            if key not in ['combination', 'permutation']:
                SDR[key] = aggregator(scores[key]['SDR'])
                SIR[key] = aggregator(scores[key]['SIR'])
                SAR[key] = aggregator(scores[key]['SAR'])
                print(f'{key} SDR: {SDR[key]:.2f} dB')
                print(f'{key} SIR: {SIR[key]:.2f} dB')
                print(f'{key} SAR: {SAR[key]:.2f} dB')
    print()
    return SDR, SIR, SAR


def trim_waveforms(reference_waveforms, track_name, split_filepath='30s_previews.csv', sample_offset=0):
    # forgive me.
    sample_rate = 44100
    with open(split_filepath, 'r') as csv_fp:
        reader = csv.reader(csv_fp)
        for row in reader:
            if row[0] == track_name:
                start_time = int(row[1]) + sample_offset
                end_time = int(row[2]) + sample_offset
                new_audio_files = []
                for audio_file in reference_waveforms:
                    new_audio_file = deepcopy(audio_file)
                    new_audio_file.set_active_region(start_time, end_time)
                    new_audio_files.append(new_audio_file)
                print('Start sample: {}\t Stop sample: {}'.format(start_time, end_time))
                return new_audio_files, int(row[3]), int(row[4])  # start time s, stop time s

        print('No match for {}'.format(track_name))
        raise FileNotFoundError


def visualize_and_embed(sources):  # useful for time alignment debug
    plt.figure(figsize=(10, 7))
    nussl_dev.utils.visualize_sources_as_waveform(
        sources, show_legend=True)
    plt.show()


def get_sdr_from_json_submission(json_filepath_root, method, track_name, target, start_time_s, stop_time_s):
    desired_json_filepath = os.path.join(json_filepath_root, method, 'test', '{}.json'.format(track_name))
    with open(desired_json_filepath, 'r') as fp:
        ft2d_result_from_submission_json = json.load(fp)

    sdr_dict = {}
    for target_results in ft2d_result_from_submission_json['targets']:
        if target_results['name'] == target:
            sdr_dict[target] = []
            for frame in target_results['frames']:
                if start_time_s <= frame['time'] < stop_time_s:
                    sdr_dict[target].append(frame['metrics']['SDR'])
        # print('{} SDR: {:.2f} dB'.format(name, np.nanmedian(sdr_dict[name])))
    return np.nanmedian(sdr_dict[target])


def parse_split_list(path_to_split_csv):
    with open(path_to_split_csv, 'r') as csv_fp:
        reader = csv.reader(csv_fp)

        cut_dict = {}
        for row in reader:
            # A Classic Education - NightOwl,2910600,4233600,66,96
            cut_dict[row[0]] = (int(row[3]), int(row[4]))  # start/stop sec
    return cut_dict


def get_method_for_stimulus(trial, condition):  # decoder ring for mushra data
    lookup_dict = {
        ('trial1', 'C1'): '2DFT', ('trial1', 'C2'): 'HPLP', ('trial1', 'C3'): 'IRM2', ('trial1', 'C4'): 'REP2',
        ('trial1', 'C5'): 'TAU1',
        ('trial2', 'C1'): '2DFT', ('trial2', 'C2'): 'HPLP', ('trial2', 'C3'): 'IRM2', ('trial2', 'C4'): 'REP2',
        ('trial2', 'C5'): 'TAU1',
        ('trial3', 'C1'): '2DFT', ('trial3', 'C2'): 'HPLP', ('trial3', 'C3'): 'IRM2', ('trial3', 'C4'): 'REP2',
        ('trial3', 'C5'): 'TAU1',
        ('trial4', 'C1'): 'IRM2', ('trial4', 'C2'): 'MWF', ('trial4', 'C3'): 'TAU1', ('trial4', 'C4'): 'UHL3',
        ('trial5', 'C1'): 'IRM2', ('trial5', 'C2'): 'MWF', ('trial5', 'C3'): 'TAU1', ('trial5', 'C4'): 'UHL3',
    }
    if (trial, condition) in lookup_dict:
        return lookup_dict[(trial, condition)]
    elif condition in ['anchor35', 'anchor70', 'reference']:
        return condition
    else:
        raise KeyError


def get_track_name_for_trial_id(trial_id):
    lookup_dict = {
        'trial1': 'Sambasevam Shanmugam - Kaathaadi', 'trial2': 'Sambasevam Shanmugam - Kaathaadi',
        'trial3': 'Arise - Run Run Run', 'trial4': 'The Doppler Shift - Atrophy',
        'trial5': 'Mu - Too Bright'
    }
    return lookup_dict[trial_id]


def get_target_for_trial_id(trial_id):
    lookup_dict = {
        'trial1': 'vocals', 'trial2': 'accompaniment',
        'trial3': 'accompaniment', 'trial4': 'other',
        'trial5': 'drums'
    }
    return lookup_dict[trial_id]


def load_subjective_data(remove_anchor_rows=True):
    subjective_data_path = 'mushra_backup_6_10-2022.csv'
    valid_data_rows = []
    with open(subjective_data_path, 'r') as fp:
        dict_reader = csv.DictReader(fp)
        headers = dict_reader.fieldnames
        for row in dict_reader:
            if row['Reference Exclude'] != 'y':
                row['Stimulus Type'] = get_method_for_stimulus(row['trial_id'], row['rating_stimulus'])
                if remove_anchor_rows and 'anchor' in row['Stimulus Type']:
                    continue

                row['Track Name'] = get_track_name_for_trial_id(row['trial_id'])
                row['Target Type'] = get_target_for_trial_id(row['trial_id'])
                valid_data_rows.append(row)
    return valid_data_rows


def calculate_median_score_by_song_target_stimulus(subjective_data_row_dicts):
    subjective_scores = {}
    for row in subjective_data_row_dicts:
        data_key = (row['Track Name'], row['Target Type'], row['Stimulus Type'])
        if data_key not in subjective_scores:
            subjective_scores[data_key] = []
        subjective_scores[data_key].append(int(row['rating_score']))

    median_subjective_scores = {key: np.nanmedian(x) for key, x in subjective_scores.items()}
    return subjective_scores, median_subjective_scores


def get_submission_filepath(track_name, stimulus_method, target):
    if stimulus_method == 'reference':
        raise NotImplementedError

    submission_root_dir = 'SiSEC18-MUS-30-WAV'
    filepath = os.path.join(submission_root_dir, stimulus_method, 'test', track_name, '{}.wav'.format(target))
    assert os.path.exists(filepath)
    return filepath


def get_submission_filepaths(median_subjective_scores):
    filepaths = {}
    for key in median_subjective_scores.keys():
        track_name, target, stimulus_method = key
        if stimulus_method == 'reference' or 'anchor' in stimulus_method:
            filepaths[key] = None
        else:
            filepaths[key] = get_submission_filepath(track_name, stimulus_method, target)
    return filepaths


def get_reference_waveforms(track_name, target):
    other_items = {
        'vocals': ['drums', 'bass', 'other'],
        'drums': ['vocals', 'bass', 'other'],
        'other': ['vocals', 'bass', 'drums'],
        'bass': ['vocals', 'other', 'drums'],
        'accompaniment': ['drums', 'bass', 'other'],
    }

    other_items_label = '+'.join(other_items[target])
    mus_db_tfm = nussl_dev.datasets.MUSDB18(folder='musdb18', subsets=['test'],
                                            transform=nussl_dev.datasets.transforms.SumSources([other_items[target]]))
    track_index = mus_db_tfm.musdb.get_track_indices_by_names([track_name])[0]
    musdb_item = mus_db_tfm[track_index]
    # if it's accompaniment, swap the order of the waveforms?
    if target == 'accompaniment':
        return musdb_item['sources'][other_items_label], musdb_item['sources']['vocals']
    else:
        return musdb_item['sources'][target], musdb_item['sources'][other_items_label]


def load_all_reference_waveforms(median_subjective_scores):
    return_dictionary = {}
    for key in median_subjective_scores.keys():
        track_name, target, stimulus_method = key
        if stimulus_method == 'reference':
            return_dictionary[key] = get_reference_waveforms(track_name, target)
    return return_dictionary


def filter_dict_by_track_target(track_name, target, complete_dict):
    # returns all stimulus type keys for a given track_name and target:
    filter_keys = []
    for key in complete_dict.keys():
        this_track_name, this_target, method = key
        if method not in ['reference'] and track_name == this_track_name and this_target == target:
            filter_keys.append(key)
    return {x: y for x, y in complete_dict.items() if x in filter_keys}


def format_2dec(float_value):
    return '{:.2f}'.format(float_value)


def calculate_sum_distance(results_headers, result_rows):
    wsdr_index = results_headers.index('WSDR (dB)')
    subj_score_index = results_headers.index('Subjective Score (0-100)')
    distance = 0
    for row in result_rows:
        distance += abs(float(row[subj_score_index]) - float(row[wsdr_index])) ** 2
    return distance


def abs_difference(x, y):
    return abs(float(x) - float(y))


def get_sum_distance_for_cache_csv(csv_filepath):
    with open(csv_filepath, 'r') as fp:
        reader = csv.reader(fp)
        headers = next(fp).split(',')
        rows = [x for x in reader]
    return calculate_sum_distance(headers, rows)


def append_coeff_summary_to_csv(coefs, error_value):
    weighting_study_results_dir = 'sdr_weighting_study'
    if not os.path.exists(weighting_study_results_dir):
        os.mkdir(weighting_study_results_dir)
    path_to_csv = os.path.join(weighting_study_results_dir, 'coefs_error_log.csv')
    if not os.path.exists(path_to_csv):
        with open(path_to_csv, 'w') as fp:
            writer = csv.writer(fp)
            writer.writerow(['e_spat', 'e_interf', 'e_artif', 'error sum'])
            writer.writerow(coefs + [error_value])
    else:
        with open(path_to_csv, 'a') as fp:
            writer = csv.writer(fp)
            writer.writerow(coefs + [error_value])


def test_sdr_weights(weights):
    weights = list(weights)
    for weight in weights:
        if weight < 0:
            return 70000
    study_filepath = os.path.join('sdr_weighting_study', 'cache', 'wsdr_study_{}_{}_{}.csv'.format(weights[0], weights[1], weights[2]))
    if os.path.exists(study_filepath):
        error_result = get_sum_distance_for_cache_csv(study_filepath)
        print('using cache result')
        print('Error Distance: {}'.format(error_result))
        return error_result

    print('Testing SDR weights {}'.format(str(weights)))
    # subjective data drives this experiment. Let's load the conditions:
    the_resources.load_subjective_data()
    median_subjective_scores = the_resources.load_subjective_data()
    estimates_filepaths_dict = the_resources.get_submission_filepaths(median_subjective_scores)
    reference_waveforms_dict = the_resources.load_references()
    sample_offset_by_estimate_method = {  # shift reference samples by n to align for SDR
        'IRM2': 0, 'TAU1': 0, 'UHL3': 0, 'MWF': 0, 'HPLP': -2048, '2DFT': -2048, 'REP2': -2048,
    }
    sample_offsets = sorted(list({y for x, y in sample_offset_by_estimate_method.items()}))  # [0, -2048]

    # e_spat_scale, e_interf_scale, e_artif_scale
    results_headers = ['Track Name', 'Target', 'Method', 'Subjective Score (0-100)',
                       'WSDR (dB)', 'SDR (dB)', 'SIR (dB)', 'SAR (dB)',
                       'JSON SDR (dB)', 'Sample offset', 'WSDR Weights', 'Subjective Score - WSDR (abs)']

    results_rows = []
    for sample_offset in sample_offsets:
        for reference_key in reference_waveforms_dict.keys():
            track_name, target, reference_method = reference_key
            matching_estimates = the_resources.get_estimate_match(track_name, target, estimates_filepaths_dict)

            reference_waveforms, track_start_time, track_stop_time = the_resources.get_reference_waveform(
                track_name, target, sample_offset, '30s_previews.csv')

            # SiSEC2018 submission .wav files
            for estimate_key, estimate_filepath in matching_estimates.items():
                this_track_name, this_target, estimate_method = estimate_key
                if sample_offset != sample_offset_by_estimate_method[estimate_method]:
                    continue

                print('\n\n\n')
                print('{} {}'.format(track_name, target))
                print('________________________________')
                estimate_audio = the_resources.get_estimate_audio(estimate_filepath)
                new_evaluation = nussl_dev.evaluation.BSSEvalV4(
                    list(reference_waveforms),
                    [estimate_audio, estimate_audio],
                    source_labels=[target, 'Rest'], window=44100, hop=44100, mss_sdr_weights=weights
                )
                new_evaluation.evaluate()
                sdr_median, wsdr_median = _report_wsdr(estimate_method, new_evaluation.scores)
                old_evaluation = nussl_dev.evaluation.BSSEvalV4(
                    list(reference_waveforms),
                    [estimate_audio, estimate_audio],
                    source_labels=[target, 'Rest'], window=44100, hop=44100
                )
                old_evaluation.evaluate()
                sdr, sir, sar = _report_sdr('Original museval', old_evaluation.scores)
                # Check against the submitted results. See readme for download URL
                submitted_sdr = get_sdr_from_json_submission(os.path.join('sigsep-mus-2018', 'submissions'),
                                                             estimate_method,
                                                             track_name, target, track_start_time, track_stop_time)
                if abs(submitted_sdr - sdr[target]) > 0.1:
                    print('Warning! SDR is {:.2f} but reported is {:.2f}'.format(sdr[target], submitted_sdr))

                subjective_score = median_subjective_scores[estimate_key]
                results_rows.append([track_name, target, estimate_method, format_2dec(subjective_score),
                                     format_2dec(wsdr_median[target]),
                                     format_2dec(sdr[target]), format_2dec(sir[target]), format_2dec(sar[target]),
                                     format_2dec(submitted_sdr), sample_offset, weights,
                                     abs(subjective_score - wsdr_median[target])])
    error_distance = calculate_sum_distance(results_headers, results_rows)
    print('Error Distance: {}'.format(error_distance))

    if not os.path.exists('sdr_weighting_study'):
        os.mkdir('sdr_weighting_study')
    cache_dir = os.path.join('sdr_weighting_study', 'cache')
    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)
    study_filepath = os.path.join(cache_dir, 'wsdr_study_{}_{}_{}.csv'.format(weights[0], weights[1], weights[2]))
    with open(study_filepath, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(results_headers)
        writer.writerows(results_rows)

    append_coeff_summary_to_csv(weights, error_distance)

    return error_distance


def main():  # Table 1
    x0 = np.array([0.0630, 0.03762, 0.0002])  # Initial starting point
    res = minimize(test_sdr_weights, x0, method='COBYLA', options={'verbose': 1, 'maxiter': 300, 'rhobeg': 0.001},
                   tol=0.00001)
    print(res.x)
    return


def test_single_weight_example():
    test_sdr_weights([1.0, 1.0, 0.05])
    return


if __name__ == '__main__':
    main()
