# ClinGen

This is the code for our paper [Knowledge-Infused Prompting: Assessing and Advancing Clinical Text Data Generation with Large Language Models](https://arxiv.org/abs/2311.00287).

# Model Framework
![ClinGen](figure/clingen.png)

## Dataset
### Generated Datasets
The original train/validation/test data, and the generated synthetic training data **will be uploaded** in Huggingface Dataset Hub soon:

| Corpus | # Train | # Test | # Class | Task | Link | 
| ------  | ------- | ----- | ----------- | ----------- | ----------- |
| LitCovid | 24960 | 6238 | 7 | Text Classification | [litcovid]()
| HOC | 3091 | 898 | 10 | Text Classification |  [hoc]()
| GAD | 4750 | 350 | 1 | Relation Extraction | [gad]()
| CDR | 8431 | 2522 | 1 | Relation Extraction | [cdr]()
| ChemProt | 8793 | 10807 | 5 | Relation Extraction | [chemprot]()
| MedNLI | 11232 | 1422 | 3 | Natural Language Inference | [mednli]()
| MEDIQA-NLI | - | 405 | 3 | Natural Language Inference |  [mediqa-nli]()
| MEDIQA-RQE | 8588 | 302 | 2 | Natural Language Inference | [mediqa-rqe]()
| PUBHEALTH | 9804 | 1231 | 4 | Fact Verification | [pubhealth]()
| HealthVer | 10591 | 1824 | 3 | Fact Verification | [healthver]()
| MQP | 10 | 3033 | 2 | Sentence Similarity | [mqp]()
| BC5CDR-Disease | 4882 | 5085 | 1 | Named Entity Recognition |  [bc5cdr-disease]()
| BC5CDR-Chemical | 4882 | 5085 | 1 | Named Entity Recognition | [bc5cdr-chemical]()
| NCBI-Disease | 5336 | 921 | 1 | Named Entity Recognition | [ncbi-disease]()
| CHEMDNER | 14522 | 12430 | 1 | Named Entity Recognition | [chemdner]()
| CASI | 5 | 100 | 6 | Attribute Extraction | [casi]()

## Training Data Generation
First of all, please apply an OpenAI API key [here](https://openai.com/blog/openai-api), if you don't have one yet.
Then, replace the `YOUR_API_KEY` in `clingen.py` with your own API key.
Finally, run `bash run_clingen.sh` with your specified dataset name and keyword type.

## Questions?
Feel free to contact `ran.xu at emory.edu` for any questions regarding this repo. Please try to specify the problem with details so we can help you better and quicker!

## Citation
If you find this repository helpful, please kindly consider citing the corresponding paper. Thanks in advance!

```
@misc{xu2023knowledgeinfused,
    title={Knowledge-Infused Prompting: Assessing and Advancing Clinical Text Data Generation with Large Language Models},
    author={Ran Xu and Hejie Cui and Yue Yu and Xuan Kan and Wenqi Shi and Yuchen Zhuang and Wei Jin and Joyce Ho and Carl Yang},
    year={2023},
    eprint={2311.00287},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
