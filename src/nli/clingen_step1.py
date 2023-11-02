import openai
import asyncio
from typing import List, Dict, Any
import argparse
import os
import re
import time
import json
import random


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

if args.dataset in ['mediqa-nli', 'mqp']:
    args.content = 'claim'
elif args.dataset in ['mediqa-rqe']:
    args.content = 'consumer health queries'
elif args.dataset in ['pubhealth', 'healthver']:
    args.content = 'health news'
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
    class_example = []
    with open(f"../data/{args.dataset}/train_few.jsonl", 'r') as f:
        for lines in f:
            example = json.loads(lines)
            class_example.append(example["text_a"])
    return class_example


def load_keywords(args):
    example = []
    dirs = os.listdir(f"../data/{args.dataset}/{args.keyword_type}/")
    for dir in dirs:
        with open(f"../data/{args.dataset}/{args.keyword_type}/{dir}", 'r') as f:
            for lines in f:
                text = lines.replace("\n", "")
                text = text.lstrip('-').strip("\"\',()[]").strip().lower()
                if text == "" or len(text) > 30:
                    continue
                example.append(text)
    return example


def gen_one_prompt(args, keywords, demo_num=3):
    with open(f"../data/{args.dataset}/styles.txt", 'r') as f:
        styles = [x.lower().strip('\n') for x in f.readlines()]
    style = random.sample(styles, 1)[0]
    prompt_init = re.sub("_", " ", f"""
                                Suppose you need to create a set of {args.content}. Your task is to:\n1. generate one 
                                sentence for a {args.content}. 
                                """).strip()
    topic_i = random.sample(keywords, 1)[0]
    prompt_init += f"\n2. the {args.content} should be relevant to '{topic_i}'."
    prompt_init += f"\n3. the {args.content} should mimic the style of {style}."
    demo = f"\nSome examples are: \n"
    few_shot_demo = load_demo(args)
    random.shuffle(few_shot_demo)
    for data in few_shot_demo[:demo_num]:
        demo += "--------------------\n"
        demo += f"{data}\n"
    demo += "--------------------\n"
    prompt = prompt_init + demo
    return prompt, topic_i, style


def main(args):
    augment_entities = load_keywords(args)

    openai.api_key = args.api_key

    example_cnt = 0
    j = 0
    while example_cnt < args.n_sample:
        prompts = []
        keywords = []
        styles = []
        for _ in range(15):
            prompt, keyword, style = gen_one_prompt(args, augment_entities)

            print('============== Input Prompt: =============')
            print(prompt)
            print('============== End of Prompt =============')
            prompts.append([{"role": "user", "content": prompt}])
            keywords.append(keyword)
            styles.append(style)
        try:
            os.makedirs(f"../data/{args.dataset}/{args.keyword_type}/step1/", exist_ok=True)
            f = open(f"../data/{args.dataset}/{args.keyword_type}/step1/train_{j}.jsonl", 'w')

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
            for text, style, keyword in zip(ans, styles, keywords):
                parsed_texts = text.strip("\n").strip().split("--------------------")
                for parsed_text in parsed_texts:
                    parsed_text = clean_str(parsed_text.strip("\n").strip())

                    try:
                        text_json = {"text": parsed_text, "style": style, "keyword": keyword}
                        f.write(json.dumps(text_json) + '\n')
                        example_cnt += 1
                    except:
                        print("Decode Error!", parsed_text)
                        pass

            print("=========================")
            print(f"# Examples / Total: {example_cnt} / {args.n_sample}")
            if ans[0]:
                print(f"Example: {ans[0]}")
            j += 1
            time.sleep(5)
            exit()

        except openai.error.RateLimitError:
            print(f"RateLimitError.")
            time.sleep(20)
            continue
        except openai.error.APIError:
            print(f"APIError.")
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
        except openai.error.Timeout:
            print("TimeoutError")
            time.sleep(10)
            continue


if __name__ == '__main__':
    main(args)
