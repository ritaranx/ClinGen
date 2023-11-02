import openai
import asyncio
from typing import List, Dict, Any
import argparse
from collections import defaultdict
from tqdm import tqdm
import re
import time
import json
import random
import nltk
import os


def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),.!?\"\']", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()


parser = argparse.ArgumentParser("")
parser.add_argument("--temperature", default=1, type=float, help="which seed to use")
parser.add_argument("--top_p", default=1.0, type=float, help="top_p for sampling")
parser.add_argument("--n_sample", default=10, type=int, help="number of examples to be generated")
parser.add_argument("--dataset", default='', type=str, help="which model to use")

parser.add_argument("--model_name", default='gpt-3.5-turbo', type=str, help="which model to use")
parser.add_argument("--max_tokens", default=512, type=int, help="which seed to use")
parser.add_argument("--keyword_type", default='.', type=str, help="kg or llm")

args = parser.parse_args()

args.api_key = YOUR_API_KEY

if args.dataset in ['litcovid']:
    args.domain = 'COVID-19 Literature'
    args.n_label = 7
elif args.dataset in ['hoc']:
    args.domain = 'Cancer Document'
    args.n_label = 10
else:
    raise NotImplementedError


async def dispatch_openai_requests(
        messages_list: List[List[Dict[str, Any]]],
        model: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
) -> List[str]:
    """Dispatches requests to OpenAI API asynchronously.
    
    Args:
        messages_list: List of messages to be sent to OpenAI ChatCompletion API.
        model: OpenAI model to use.
        temperature: Temperature to use for the model.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use for the model.
    Returns:
        List of responses from OpenAI API.
    """
    async_responses = [
        openai.ChatCompletion.acreate(
            model=model,
            messages=x,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
        for x in messages_list
    ]
    return await asyncio.gather(*async_responses)


def load_demo(args):
    """
    Loading the few-shot demonstration for synthetic data generation
    """
    class_example = defaultdict(list)
    with open(f"../data/{args.dataset}/train_few.jsonl", 'r') as f:
        for lines in f:
            x = json.loads(lines)
            for i, y in enumerate(x['_id']):
                if y == 1:
                    class_example[i].append(x['text'])
    return class_example


def load_keywords(args, name):
    example = []
    with open(f"../data/{args.dataset}/{args.keyword_type}/{name}.txt", 'r') as f:
        for lines in f:
            text = lines.replace("\n", "")
            text = text.lstrip('-').lstrip('0123456789.').strip("\"\',()[]").strip().lower()
            if text == "":
                continue
            example.append(text)
    return example


def gen_one_prompt(args, keyword_dict, i, class_name, few_shot_demo, demo_num=3):
    with open(f"../data/{args.dataset}/styles.txt", 'r') as f:
        styles = [x.lower().strip('\n') for x in f.readlines()]
    style = random.sample(styles, 1)[0]
    prompt_init = re.sub("_", " ", f"""
                            Suppose you need to create a dataset for {args.domain}. Your task is to:\n
                            1. generate a sentence about {args.domain}.
                            """).strip()
    topic_i = random.sample(keyword_dict[class_name], 1)[0]
    prompt_init += f"\n2. the sentence should mimic the style of {style}," \
                   f"\n3. the sentence should be relevant to the subtopic of {topic_i} for {class_name}."
    demo = f" Some examples for {class_name} are: \n"
    random.shuffle(few_shot_demo[i])
    for data in few_shot_demo[i][:demo_num]:
        sentences = nltk.sent_tokenize(data)
        first_three_sentences = " ".join(sentences[:3])
        demo += f'Label: {class_name}\n'
        demo += f'Text: {first_three_sentences}\n'
    demo += f'Label: {class_name}\n'
    demo += f'Text:'
    prompt = prompt_init + demo

    return prompt, topic_i


def main(args):
    with open(f"../data/{args.dataset}/label.txt", 'r') as f:
        label_names = [x.lower().strip('\n') for x in f.readlines()]
    keyword_dict = {}
    for label_name in label_names:
        keywords = load_keywords(args, label_name.replace(" ", "_"))
        keyword_dict[label_name] = keywords

    few_shot_demo = load_demo(args)

    openai.api_key = args.api_key

    for i, class_name in tqdm(enumerate(label_names)):
        example_cnt = 0
        j = 0
        while example_cnt < args.n_sample:
            prompts = []
            keywords = []
            for _ in range(20):
                prompt, keyword = gen_one_prompt(args, keyword_dict, i, class_name, few_shot_demo, label_names)

                print('============== Input Prompt: =============')
                print(prompt)
                print('============== End of Prompt =============')
                prompts.append(
                    [{"role": "user", "content": prompt}]
                )
                keywords.append(keyword)
            try:
                os.makedirs(f"../data/{args.dataset}/{args.keyword_type}/{class_name}/", exist_ok=True)
                f = open(f"../data/{args.dataset}/{args.keyword_type}/{class_name}/train_{j}.jsonl", 'w')

                response = asyncio.run(
                    dispatch_openai_requests(
                        messages_list=prompts,
                        model=args.model_name,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                        top_p=args.top_p,
                    )
                )
                # parse the output from LLM
                ans = [x['choices'][0]['message']['content'] for x in response]

                # write the generated text to the target folder
                for text, keyword in zip(ans, keywords):
                    example_cnt += 1
                    lbl = [0 for _ in range(args.n_label)]
                    lbl[i] = 1
                    data = {"_id": lbl, "keyword": keyword, "text": text}
                    f.write(json.dumps(data) + '\n')
                print("=========================")
                print(f"# Examples: {example_cnt}")
                if ans[0]:
                    print(f"Example: {ans[0]}")
                j += 1
            except openai.error.RateLimitError:
                print(f"RateLimitError for class {i}.")
                time.sleep(20)
                continue
            except openai.error.APIError:
                print(f"APIError for class {i}.")
                time.sleep(10)
                continue
            except openai.error.InvalidRequestError:
                print("InvalidRequestError!")
                time.sleep(10)
                continue
            except openai.error.ServiceUnavailableError:
                print("ServiceUnavailableError")
                time.sleep(10)
                continue


if __name__ == '__main__':
    main(args)
