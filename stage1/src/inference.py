"""Core inference pipeline for self-correction experiments.

Supports both vLLM (fast batched) and HuggingFace transformers (fallback).
"""

import json
import os
import yaml
import torch
from pathlib import Path

# vLLM disabled — causes torch corruption on Grace cluster. Use HF transformers.
VLLM_AVAILABLE = False

from transformers import AutoModelForCausalLM, AutoTokenizer


# ─── Model Loading ───────────────────────────────────────────────────────────

MODEL_CONFIGS = {
    'llama': {
        'name': 'meta-llama/Llama-3.1-8B-Instruct',
        'max_new_tokens': 1024,
        'chat_template': True,
    },
    'mistral': {
        'name': 'mistralai/Mistral-7B-Instruct-v0.3',
        'max_new_tokens': 1024,
        'chat_template': True,
    },
    'qwen': {
        'name': '/scratch/user/ashish_molakalapalli/datasets/qwen2.5-7b',
        'max_new_tokens': 1024,
        'chat_template': True,
    },
}


class ModelWrapper:
    """Unified interface for vLLM and HF transformers inference."""

    def __init__(self, model_key, use_vllm=None, cache_dir=None):
        self.model_key = model_key
        self.config = MODEL_CONFIGS[model_key]
        self.model_name = self.config['name']
        self.max_new_tokens = self.config['max_new_tokens']

        if cache_dir is None:
            cache_dir = os.environ.get(
                'TRANSFORMERS_CACHE',
                '/scratch/user/ashish_molakalapalli/nlp_project/data/models'
            )

        if use_vllm is None:
            use_vllm = VLLM_AVAILABLE

        self.use_vllm = use_vllm

        if self.use_vllm:
            self._load_vllm(cache_dir)
        else:
            self._load_hf(cache_dir)

    def _load_vllm(self, cache_dir):
        self.llm = LLM(
            model=self.model_name,
            download_dir=cache_dir,
            dtype='float16',
            max_model_len=4096,
            gpu_memory_utilization=0.90,
        )
        self.tokenizer = self.llm.get_tokenizer()

    def _load_hf(self, cache_dir):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, cache_dir=cache_dir
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            cache_dir=cache_dir,
            torch_dtype=torch.float16,
            device_map='auto',
        )
        self.model.eval()

    def generate(self, prompts, temperature=0.0, max_new_tokens=None, n=1):
        """Generate completions for a list of prompts.

        Args:
            prompts: list of prompt strings (already formatted with chat template)
            temperature: sampling temperature (0 = greedy)
            max_new_tokens: override default max tokens
            n: number of completions per prompt

        Returns:
            list of lists of strings (one list of n completions per prompt)
        """
        max_tokens = max_new_tokens or self.max_new_tokens

        if self.use_vllm:
            return self._generate_vllm(prompts, temperature, max_tokens, n)
        else:
            return self._generate_hf(prompts, temperature, max_tokens, n)

    def _generate_vllm(self, prompts, temperature, max_tokens, n):
        params = SamplingParams(
            temperature=max(temperature, 0.01) if temperature > 0 else 0,
            top_p=0.95 if temperature > 0 else 1.0,
            max_tokens=max_tokens,
            n=n,
        )
        outputs = self.llm.generate(prompts, params)
        results = []
        for output in outputs:
            completions = [o.text.strip() for o in output.outputs]
            results.append(completions)
        return results

    def _generate_hf(self, prompts, temperature, max_tokens, n):
        results = []
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True, max_length=3072)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            gen_kwargs = {
                'max_new_tokens': max_tokens,
                'do_sample': temperature > 0,
                'pad_token_id': self.tokenizer.pad_token_id,
            }
            if temperature > 0:
                gen_kwargs['temperature'] = temperature
                gen_kwargs['top_p'] = 0.95
                gen_kwargs['num_return_sequences'] = n
            else:
                gen_kwargs['num_return_sequences'] = 1

            with torch.no_grad():
                output_ids = self.model.generate(**inputs, **gen_kwargs)

            # Decode only the new tokens
            input_len = inputs['input_ids'].shape[1]
            completions = []
            for seq in output_ids:
                text = self.tokenizer.decode(seq[input_len:], skip_special_tokens=True).strip()
                completions.append(text)

            # If n > 1 but greedy, duplicate (shouldn't happen in practice)
            while len(completions) < n:
                completions.append(completions[0])

            results.append(completions)
        return results

    def format_chat(self, messages):
        """Format messages using the model's chat template.

        Args:
            messages: list of dicts with 'role' and 'content'

        Returns:
            formatted prompt string
        """
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


# ─── Prompt Building ────────────────────────────────────────────────────────

_prompts_cache = {}

def load_prompts():
    """Load prompt templates from YAML config files."""
    if _prompts_cache:
        return _prompts_cache

    config_dir = Path(__file__).parent.parent / 'configs' / 'prompts'

    with open(config_dir / 'initial.yaml') as f:
        _prompts_cache['initial'] = yaml.safe_load(f)
    with open(config_dir / 'correction.yaml') as f:
        _prompts_cache['correction'] = yaml.safe_load(f)

    return _prompts_cache


def build_initial_prompt(dataset_name, question, cot=False):
    """Build the initial generation prompt for a question.

    Args:
        dataset_name: one of 'gsm8k', 'triviaqa', 'strategyqa', 'humaneval'
        question: the question text (or prompt for HumanEval)
        cot: if True, prepend chain-of-thought prefix

    Returns:
        list of message dicts for chat template
    """
    prompts = load_prompts()
    template = prompts['initial'][dataset_name]

    if dataset_name == 'humaneval':
        content = template.format(prompt=question)
    else:
        content = template.format(question=question)

    if cot:
        cot_prefix = prompts['correction'].get('cot_prefix', "Let's think step by step.\n\n")
        content = cot_prefix + content

    return [{'role': 'user', 'content': content}]


def build_correction_prompt(strategy, dataset_name, question, response, explanation=None):
    """Build a correction prompt for the given strategy.

    Args:
        strategy: one of 's1', 's2', 's3', 's4_confidence', 's5_explain', 's5_verify'
        dataset_name: dataset name for task-specific prompts
        question: original question
        response: model's previous response (y0)
        explanation: for s5_verify, the explanation from s5_explain step

    Returns:
        list of message dicts for chat template
    """
    prompts = load_prompts()
    correction = prompts['correction']

    if strategy == 's1' or strategy == 's3':
        content = correction['s1_simple'].format(response=response)

    elif strategy == 's2':
        task_map = {
            'gsm8k': 's2_math',
            'triviaqa': 's2_factual',
            'strategyqa': 's2_commonsense',
            'humaneval': 's2_code',
        }
        template_key = task_map[dataset_name]
        content = correction[template_key].format(
            question=question, response=response
        )

    elif strategy == 's4_confidence':
        content = correction['s4_confidence'].format(
            question=question, response=response
        )

    elif strategy == 's5_explain':
        content = correction['s5_explain'].format(
            question=question, response=response
        )

    elif strategy == 's5_verify':
        content = correction['s5_verify'].format(
            question=question, response=response, explanation=explanation or ''
        )

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return [{'role': 'user', 'content': content}]
