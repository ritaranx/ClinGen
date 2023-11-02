import openai
import asyncio
from typing import List, Dict, Any
import argparse
from collections import defaultdict
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
parser.add_argument("--max_tokens", default=256, type=int, help="which seed to use")
parser.add_argument("--keyword_type", default='.', type=str, help="kg or llm")

args = parser.parse_args()

args.api_key = YOUR_API_KEY

if args.dataset in ['mediqa-nli']:
    args.content_first = 'premise'
    args.content = 'hypothesis'
    args.domain = 'Natural Language Entailment'
elif args.dataset in ['mediqa_rqe']:
    args.content_first = 'question A'
    args.content = 'question B'
    args.domain = 'Question Entailment'
elif args.dataset in ['pubhealth', 'healthver']:
    args.content_first = 'claim'
    args.content = 'evidence'
    args.domain = 'Fact Verification'
elif args.dataset in ['mqp']:
    args.content_first = 'question'
    args.content = 'question'
    args.domain = 'Sentence Similarity Calculation'
else:
    raise NotImplementedError

if args.dataset == 'mediqa-rqe':
    class_desc = {
        "not_entail": "the answer to the question B is not a answer to the question A.",
        "entail": "every answer to the question B is also a complete or partial answer to the question A."
    }
elif args.dataset == 'mediqa-nli':
    class_desc = {
        "entail": "the hypothesis is definitely a true description of the patient given the premise.",
        "contradict": "the hypothesis is definitely a false description of the patient given the premise.",
        "neutral": "the hypothesis might or might not be a true description of the patient given the premise.",
    }
elif args.dataset == 'mqp':
    class_desc = {
        "not_equivalent": "two sentences are syntactically dissimilar but contextually similar.",
        "equivalent": "two sentences may look similar syntactically but contextually dissimilar."
    }
elif args.dataset == 'pubhealth':
    class_desc = {
        "refute": "the evidence contradict the claim.",
        "unproven": "the evidence is not relevant to the claim and does not provide enough information to support or contradict the claim.",
        "support": "the evidence support or justify the correctness of the claim.",
        "mixture": "some information from the evidence support the claim and some information from the evidence contradict the claim."
    }


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
    with open(f"../../data/{args.dataset}/train_few.jsonl", 'r') as f:
        for lines in f:
            x = json.loads(lines)
            idx = x["_id"]
            if "original_text" in x:
                del x["original_text"]
            class_example[idx].append(json.dumps(x))
    return class_example


def load_sentence(args):
    class_example = []
    dirs = os.listdir(f"../data/{args.dataset}/{args.keyword_type}/step1/")
    for dir in dirs:
        with open(f"../data/{args.dataset}/{args.keyword_type}/step1/{dir}", 'r') as f:
            for lines in f:
                x = json.loads(lines)
                class_example.append([x["text"], x["keyword"]])
    return class_example


def gen_one_prompt(args, idx, keywords, label_name, demos, demo_num=3):
    sentence, keyword = random.sample(keywords, 1)[0]
    prompt_init = re.sub("_", " ",
                         f"""Suppose you need to create a pair of sentence for the {args.domain} task with the label 
                         '{label_name}'. Given the {args.content_first}: '{sentence}', your task is to:\n
                         """).strip()
    prompt_init += f"1. generate one short {args.content} about {keyword} so that {class_desc[label_name]}.\n"
    prompt_init += f"2. the {args.content} should mimic the style of the first sentence.\n"

    demo = f"\nSome examples are: \n"
    random.shuffle(demos)
    for data in demos[:demo_num]:
        demo += "------\n"
        demo += f"{data}\n"
    if args.dataset == 'mqp':
        demo += "------\n"
    else:
        demo += "------\nComplete the following example by filling the text for 'sent_b' in the dictionary. Please " \
                "return the whole dictionary:\n "
        demo += json.dumps({"_id": idx, "label": label_name, "sent_a": sentence, "sent_b": ""})
    prompt = prompt_init + demo
    return prompt, sentence, keyword


def main(args):
    with open(f"../../data/{args.dataset}/label.txt", 'r') as f:
        label_names = [x.lower().strip('\n') for x in f.readlines()]
    few_shot_demo = load_demo(args)
    sentences_a = load_sentence(args)

    openai.api_key = args.api_key
    for i, label_name in enumerate(label_names):
        example_cnt = 0
        j = 0
        while example_cnt < (args.n_sample // len(label_names)):
            prompts = []
            sentences = []
            keywords = []
            for _ in range(10):
                prompt, sentence, keyword = gen_one_prompt(args, i, sentences_a, label_name,
                                                           demos=few_shot_demo[i])

                print('============== Input Prompt: =============')
                print(prompt)
                print('============== End of Prompt =============')
                prompts.append(
                    [{"role": "user", "content": prompt}]
                )

                sentences.append(sentence)
                keywords.append(keyword)
            try:
                os.makedirs(f"../data/{args.dataset}/{args.keyword_type}/{label_name}/", exist_ok=True)
                f = open(f"../data/{args.dataset}/{args.keyword_type}/{label_name}/train_{j}.jsonl", 'w')

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
                for text, sentence, keyword in zip(ans, sentences, keywords):
                    parsed_texts = text.strip("\n").strip().split("--------------------")
                    for parsed_text in parsed_texts:
                        try:
                            parsed_text = json.loads(parsed_text)
                            parsed_text["keyword"] = keyword
                            f.write(json.dumps(parsed_text) + '\n')
                            example_cnt += 1
                        except:
                            print("Decode Error!", parsed_text)
                            pass

                print("=========================")
                print(f"# Examples / Total: {example_cnt} / {args.n_sample}")
                if ans[0]:
                    print(f"Example: {ans[0]}")
                j += 1
                time.sleep(60)

            except openai.error.RateLimitError:
                print(f"RateLimitError.")
                time.sleep(60)
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
