from curriculumlearning.specialist.interface import Specialist

def batch_specialists(start_generation, batch_sizes, generation_trials, fit_historical=True):
    configs = {}
    for batch_size in batch_sizes:
        configs[f'sp{batch_size}'] = dict(
            start_generation=start_generation,
            fit_batch_size=batch_size,
            score_batch_size=batch_size,
            generation_trials=generation_trials,
            fit_historical_data=fit_historical
        )
    return configs
