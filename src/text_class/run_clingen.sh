model_name=gpt-3.5-turbo
temperature=1.0
n_sample=5000
top_p=1.0


keyword_type=kg  # or llm
dataset=litcovid  # or hoc

python clingen.py --temperature=${temperature} --top_p=${top_p} --n_sample=${n_sample} --dataset=${dataset} \
                  --model_name=${model_name} --keyword_type=${keyword_type}