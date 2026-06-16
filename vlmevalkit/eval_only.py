import json
import os
import argparse
import pandas as pd
from vlmeval.smp import *
from vlmeval.config import supported_VLM
from vlmeval.dataset import build_dataset
from vlmeval.utils.result_transfer import MMMU_result_transfer, MMTBench_result_transfer
from vlmeval.dataset.video_dataset_config import supported_video_datasets
from tabulate import tabulate
import copy as cp

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation Only Script based on run.py")
    parser.add_argument('--data', type=str, nargs='+', help='Names of Datasets')
    parser.add_argument('--model', type=str, nargs='+', help='Names of Models')
    parser.add_argument('--config', type=str, help='Path to the Config Json File')
    parser.add_argument('--work-dir', type=str, default='./outputs', help='directory where inference results are saved')
    parser.add_argument('--api-nproc', type=int, default=4, help='Parallel API calling for evaluation')
    parser.add_argument('--retry', type=int, default=100, help='retry numbers for API VLMs')
    parser.add_argument('--judge-args', type=str, default=None, help='Judge arguments in JSON format')
    parser.add_argument('--judge', type=str, default=None, help='Explicitly set the judge model')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--use-verifier', action='store_true', help='use verifier to evaluate')
    parser.add_argument('--use-vllm', action='store_true', help='flag kept for config compatibility, not used in eval')

    args = parser.parse_args()
    return args

def build_dataset_from_config(cfg, dataset_name):
    import vlmeval.dataset
    import inspect
    config = cp.deepcopy(cfg[dataset_name])
    if config == {}:
        return supported_video_datasets[dataset_name]()
    assert 'class' in config
    cls_name = config.pop('class')
    if hasattr(vlmeval.dataset, cls_name):
        cls = getattr(vlmeval.dataset, cls_name)
        sig = inspect.signature(cls.__init__)
        valid_params = {k: v for k, v in config.items() if k in sig.parameters}
        if cls.MODALITY == 'VIDEO':
            if valid_params.get('fps', 0) > 0 and valid_params.get('nframe', 0) > 0:
                raise ValueError('fps and nframe should not be set at the same time')
            if valid_params.get('fps', 0) <= 0 and valid_params.get('nframe', 0) <= 0:
                raise ValueError('fps and nframe should be set at least one valid value')
        return cls(**valid_params)
    else:
        raise ValueError(f'Class {cls_name} is not supported in `vlmeval.dataset`')

def main():
    logger = get_logger('EVAL_ONLY')
    args = parse_args()

    use_config, cfg = False, None
    if args.config is not None:
        assert args.data is None and args.model is None, '--data and --model should not be set when using --config'
        use_config, cfg = True, load(args.config)
        args.model = list(cfg['model'].keys())
        args.data = list(cfg['data'].keys())
    else:
        assert len(args.data), '--data should be a list of data files'

    if 'MMEVAL_ROOT' in os.environ:
        args.work_dir = os.environ['MMEVAL_ROOT']

    for _, model_name in enumerate(args.model):
        pred_root_meta = osp.join(args.work_dir, model_name)
        
        if not osp.exists(pred_root_meta):
            logger.error(f"Directory {pred_root_meta} does not exist. Did you run inference?")
            continue

        for _, dataset_name in enumerate(args.data):
            try:
                if use_config:
                    dataset = build_dataset_from_config(cfg['data'], dataset_name)
                else:
                    dataset_kwargs = {}
                    if dataset_name in ['MMLongBench_DOC', 'DUDE', 'DUDE_MINI', 'SLIDEVQA', 'SLIDEVQA_MINI']:
                        dataset_kwargs['model'] = model_name
                    dataset = build_dataset(dataset_name, **dataset_kwargs)

                if dataset is None:
                    logger.error(f'Dataset {dataset_name} is not valid, skipping.')
                    continue

                pred_format = get_pred_file_format() 
                result_filename = f'{model_name}_{dataset_name}.{pred_format}'
                result_file = osp.join(pred_root_meta, result_filename)

                if not osp.exists(result_file):
                    logger.warning(f"File {result_file} not found via standard path. Searching recursively...")
                    candidates = find_file(pred_root_meta, result_filename)
                    if candidates:
                        result_file = candidates
                        logger.info(f"Found file at: {result_file}")
                    else:
                        logger.error(f"Cannot find inference result file for {model_name} on {dataset_name}. Skipping.")
                        continue
                else:
                    logger.info(f"Processing result file: {result_file}")

                judge_kwargs = {
                    'nproc': args.api_nproc,
                    'verbose': args.verbose,
                    'retry': args.retry if args.retry is not None else 3,
                    **(json.loads(args.judge_args) if args.judge_args else {}),
                }

                if args.retry is not None:
                    judge_kwargs['retry'] = args.retry
                if args.judge is not None:
                    judge_kwargs['model'] = args.judge
                else:
                    if dataset.TYPE in ['MCQ', 'Y/N', 'MCQ_MMMU_Pro'] or listinstr(
                        ['moviechat1k', 'mme-reasoning'], dataset_name.lower()
                    ):
                        if listinstr(['WeMath', 'MME-Reasoning'], dataset_name):
                            judge_kwargs['model'] = 'qwen2.5-vl-72b-instruct'
                        elif listinstr(['VisualPuzzles'], dataset_name):
                            judge_kwargs['model'] = 'exact_matching'
                        elif listinstr(['VisuLogic'], dataset_name):
                            judge_kwargs['model'] = 'exact_matching'
                        else:
                            judge_kwargs['model'] = 'qwen2.5-vl-72b-instruct'
                    elif listinstr(['MMVet', 'LLaVABench', 'MMBench_Video'], dataset_name):
                        if listinstr(['LLaVABench_KO'], dataset_name):
                            judge_kwargs['model'] = 'qwen2.5-vl-72b-instruct'
                        else:
                            judge_kwargs['model'] = 'qwen2.5-vl-72b-instruct'
                    elif listinstr(['VGRPBench'], dataset_name):
                        judge_kwargs['model'] = 'qwen2.5-vl-72b-instruct'
                    elif listinstr(['MathVista', 'MathVerse', 'MathVision', 'DynaMath', 'VL-RewardBench', 'LogicVista', 'MOAT', 'OCR_Reasoning', 'VSP_maze_task_main_original'], dataset_name):
                        judge_kwargs['model'] = 'qwen2.5-vl-72b-instruct'
                    elif listinstr(['OlympiadBench'], dataset_name):
                        use_api_judger = judge_kwargs.get("olympiad_use_api_judger", False)
                        if use_api_judger:
                            judge_kwargs['model'] = 'qwen2.5-vl-72b-instruct'
                    elif listinstr(['MMLongBench', 'MMDU', 'DUDE', 'SLIDEVQA', 'MIA-Bench', 'WildVision', 'MMAlignBench', 'MM-IFEval'], dataset_name):
                        judge_kwargs['model'] = 'qwen2.5-vl-72b-instruct'
                    elif listinstr(['ChartMimic'], dataset_name):
                        judge_kwargs['model'] = 'qwen2.5-vl-72b-instruct'
                    elif listinstr(['VDC'], dataset_name):
                        judge_kwargs['model'] = 'qwen2.5-vl-72b-instruct'
                    elif listinstr(['Video_MMLU_QA', 'Video_MMLU_CAP'], dataset_name):
                        judge_kwargs['model'] = 'qwen2.5-vl-72b-instruct'
                    elif listinstr(['MMVMBench'], dataset_name):
                        judge_kwargs['model'] = 'qwen2.5-vl-72b-instruct'
                    elif listinstr(['CVQA_EN', 'CVQA_LOC'], dataset_name):
                        judge_kwargs['model'] = 'qwen2.5-vl-72b-instruct'
                    elif listinstr(['M4Bench'], dataset_name):
                        judge_kwargs['model'] = 'qwen2.5-vl-72b-instruct'
                    elif listinstr(['AyaVisionBench'], dataset_name):
                        judge_kwargs['model'] = 'qwen2.5-vl-72b-instruct'
                    elif listinstr(['MathCanvas'], dataset_name):
                        judge_kwargs['model'] = 'qwen2.5-vl-72b-instruct'
                    elif listinstr(['MMReason', 'EMMA'], dataset_name):
                        judge_kwargs['model'] = 'qwen2.5-vl-72b-instruct'
                
                if args.use_verifier:
                    judge_kwargs['use_verifier'] = True
                if args.use_vllm:
                    judge_kwargs['use_vllm'] = True

                logger.info(f"Judge Kwargs: {judge_kwargs}")

                if dataset_name in ['MMMU_TEST']:
                    result_json = MMMU_result_transfer(result_file)
                    logger.info(f'Transfer MMMU_TEST result to json for official evaluation, json file saved in {result_json}')
                    continue
                elif 'MMT-Bench_ALL' in dataset_name:
                    submission_file = MMTBench_result_transfer(result_file, **judge_kwargs)
                    logger.info(f'Submission file saved in {submission_file}')
                    continue
                elif 'MLLMGuard_DS' in dataset_name:
                    logger.info('The evaluation of MLLMGuard_DS is not supported yet. ')
                    continue
                elif 'AesBench_TEST' == dataset_name:
                    logger.info(f'The results are saved in {result_file}. Please send it to the AesBench Team.')
                    continue
                elif dataset_name in ['DocVQA_TEST', 'InfoVQA_TEST', 'Q-Bench1_TEST', 'A-Bench_TEST']:
                    logger.info(f'{dataset_name} is a test split without ground-truth. Only inference is supported.')
                    continue
                elif dataset_name in ['MMBench_TEST_CN', 'MMBench_TEST_EN', 'MMBench', 'MMBench_CN', 'MMBench_TEST_CN_V11', 'MMBench_TEST_EN_V11', 'MMBench_V11', 'MMBench_CN_V11'] and not MMBenchOfficialServer(dataset_name):
                     logger.error(f'Can not evaluate {dataset_name} on non-official servers.')
                     continue

                eval_proxy = os.environ.get('EVAL_PROXY', None)
                old_proxy = os.environ.get('HTTP_PROXY', '')
                if eval_proxy is not None:
                    proxy_set(eval_proxy)

                eval_results = dataset.evaluate(result_file, **judge_kwargs)

                if eval_results is not None:
                    assert isinstance(eval_results, dict) or isinstance(eval_results, pd.DataFrame)
                    logger.info(f'The evaluation of model {model_name} x dataset {dataset_name} has finished! ')
                    logger.info('Evaluation Results:')
                    if isinstance(eval_results, dict):
                        logger.info('\n' + json.dumps(eval_results, indent=4))
                    elif isinstance(eval_results, pd.DataFrame):
                        if len(eval_results) < len(eval_results.columns):
                            eval_results = eval_results.T
                        logger.info('\n' + tabulate(eval_results))

                if eval_proxy is not None:
                    proxy_set(old_proxy)

            except Exception as e:
                logger.exception(f'Evaluation for Model {model_name} x Dataset {dataset_name} failed: {e}')
                continue

    logger.info("All evaluations completed.")

def find_file(directory, filename):
    for root, dirs, files in os.walk(directory):
        if filename in files:
            return os.path.join(root, filename)
    return None

if __name__ == '__main__':
    load_env()
    main()