import fire
import os


CONFIGS = {
    'arc': {
        'tasks': 'arc_challenge',
        'n_shots': 25,
        'metric_name': 'acc_norm'
    },
    'hellaswag': {
        'tasks': 'hellaswag',
        'n_shots': 10,
        'metric_name': 'acc_norm'
    },
    'truthfulqa': {
        'tasks': 'truthfulqa_mc2',
        'n_shots': 0,
        'metric_name': 'acc'
    },
    'mmlu': {
        'tasks': 'mmlu_abstract_algebra,mmlu_anatomy,mmlu_astronomy,mmlu_business_ethics,mmlu_clinical_knowledge,mmlu_college_biology,mmlu_college_chemistry,mmlu_college_computer_science,mmlu_college_mathematics,mmlu_college_medicine,mmlu_college_physics,mmlu_computer_security,mmlu_conceptual_physics,mmlu_econometrics,mmlu_electrical_engineering,mmlu_elementary_mathematics,mmlu_formal_logic,mmlu_global_facts,mmlu_high_school_biology,mmlu_high_school_chemistry,mmlu_high_school_computer_science,mmlu_high_school_european_history,mmlu_high_school_geography,mmlu_high_school_government_and_politics,mmlu_high_school_macroeconomics,mmlu_high_school_mathematics,mmlu_high_school_microeconomics,mmlu_high_school_physics,mmlu_high_school_psychology,mmlu_high_school_statistics,mmlu_high_school_us_history,mmlu_high_school_world_history,mmlu_human_aging,mmlu_human_sexuality,mmlu_international_law,mmlu_jurisprudence,mmlu_logical_fallacies,mmlu_machine_learning,mmlu_management,mmlu_marketing,mmlu_medical_genetics,mmlu_miscellaneous,mmlu_moral_disputes,mmlu_moral_scenarios,mmlu_nutrition,mmlu_philosophy,mmlu_prehistory,mmlu_professional_accounting,mmlu_professional_law,mmlu_professional_medicine,mmlu_professional_psychology,mmlu_public_relations,mmlu_security_studies,mmlu_sociology,mmlu_us_foreign_policy,mmlu_virology,mmlu_world_religions',
        'n_shots': 5,
        'metric_name': 'acc'
    },
    'winogrande': {
        'tasks': 'winogrande',
        'n_shots': 5,
        'metric_name': 'acc'
    },
    'gsm8k': {
        'tasks': 'gsm8k',
        'n_shots': 5,
        'metric_name': 'acc'
    },
}


def evaluate(config_name, model_name_or_path, output_path):
    config = CONFIGS[config_name]
    tasks, n_shots, metric_name = \
        config['tasks'], config['n_shots'], config['metric_name']

    command = f'/data/home/beidic/anaconda3/envs/peihao/bin/python -m lm_eval --model hf '\
              f'--model_args pretrained={model_name_or_path},parallelize=True '\
              f'--tasks {tasks} '\
              f'--num_fewshot {n_shots} '\
              f'--batch_size auto '\
              f'--output_path {output_path}'

    print(f'COMMAND: {command}')

    if os.system(command) != 0:
        print(f'COMMAND FAILED: {command}')


def main(model_name_or_path):
    for config_name in CONFIGS.keys():
        if os.path.exists(model_name_or_path):
            output_path = f'{model_name_or_path}/eval_{config_name}.json'
        else:
            model_name = model_name_or_path.split('/')[-1]
            output_path = f'eval_{model_name}_{config_name}.json'

        if (os.path.exists(output_path) or
                os.path.exists(f'{output_path}.running')):
            print(f'{output_path} exists, skipped.')
            continue
        else:
            print('', file=open(f'{output_path}.running', 'w'), flush=True)
            evaluate(
                config_name=config_name,
                model_name_or_path=model_name_or_path,
                output_path=output_path)
            os.remove(f'{output_path}.running')


if __name__ == '__main__':
    fire.Fire(main)
