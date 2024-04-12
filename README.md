# DUE-Benchmark

This repository is forked from the original [due-benchmark/baselines](https://github.com/due-benchmark/baselines).

This repository only supports the data loader scripts, not the training scripts. The training scripts are available in the original repository.

## Usage

### Install

```bash
$ pip install git+https://github.com/retrieva/DUE-Benchmark.git
```

### How to use

```python
from benchmarker.data.reader import Corpus

corpus = Corpus()
corpus.read_benchmark_challenge(directory="/path/to/dataset", ocr="ocr name")

# Get the data
train_subset = getattr(corpus, "train")
for train_instance in train_subset:
  print(train_instance)
```

See `tests/test_corpus.py` for an example of how to use the data loader.

---

Following is the original README.md from the original repository.

This is the repository that provide tools to download data, reproduce the baseline results and evaluation.



## What can you achieve with this guide
Based on this repository, you may be able to:

1. download data for benchmark in a unified format.
2. run all the baselines.
3. evaluate already trained baseline models.

## Install benchmark-related repositories
Start the container:
```bash
sudo userdocker run nvcr.io/nvidia/pytorch:20.12-py3
```

Clone the repo with:
```bash
git clone git@github.com:due-benchmark/baselines.git
git clone git@github.com:due-benchmark/evaluator.git

```

Install the requirements:
```bash
cd evaluator && pip install -e .
cd ../baselines && pip install -e .
```

# 1. Download datasets and the base model
The datasets are re-hosted on the https://duebenchmark.com/data and can be downloaded from there.
Moreover, since the baselines are finetuned based on the T5 model, you need to download the original model.
Again it is re-hosted at https://duebenchmark.com/data.
Please place datasets into the `DATASETS_ROOT` directory after downloading.
Make sure the path to the model directory is set as `INIT_LARGE_MODEL_PATH`.\
The structure of the directory for each of the dataset is like in the following example:
```bash
└── KleisterCharity
    ├── dev
    │   ├── document.jsonl
    │   └── documents_content.jsonl
    ├── document.jsonl
    ├── documents.json
    ├── documents.jsonl
    ├── documents_content.jsonl
    ├── test
    │   ├── document.jsonl
    │   └── documents_content.jsonl
    └── train
        ├── document.jsonl
        └── documents_content.jsonl
```


## 1.1 Download pdf files (optional)
The pdf files are also stored on the website. Please download and unpack them in case you need to train you own models.
Please note that for our baselines and some simple advancements all the neccessary OCR information is already provided in the previously downloaded `document_content.jsonl`.

# 2. Run baseline trainings

## 2.1 Process datasets into memmaps (binarization)

Instead of processing input data batch by batch, epoch after epoch, we assume some intermediate step that consists of tokenization, input tensors' preparation, and storing such data in a binarized form.

To process datasets into the said 'memmaps,' two variables in the `create_memmaps.sh` files must be set: `DATASETS_ROOT`, and `TOKENIZER`. The former provides the path to the directory where all of the datasets are downloaded and unpacked. The latter is a path to model dump used to perform the tokenization process. Every of our baseline uses the T5 model as a backbone. Thus it can be a patch to arbitrary model downloaded from our [Datasets and Baselines](http://duebenchmark.com/data) page, e.g., the `T5-large`. Similarly, the model has to be unpacked before performing the binarization.

By default, the `create_memmaps.sh`, assumes that we are about to process every dataset from the DUE benchmark, using all available OCR layers and limit on input sequence length equivalent to those from our experiments.

Long story short, set `DATASETS_ROOT` and `TOKENIZER`, then run `./create_memmaps.sh`.

## 2.2 Run training script

Single training can be started with the following command, assuming `OUT_DIR` is set as an output for the trained model's checkpoints and generated outputs.
Additionally, set `DATASET` to any of the previously generated datasets (e.g., to `DeepForm`).
Choose and export the OCR of your choice into the `OCR_ENGINE` variable (choose from `microsoft_cv, tesseract, djvu` )

To reproduce our model's training from the original due benchmark see the following example script(in `baselines` repo):
```bash
# modify paths below
export INIT_LARGE_MODEL_PATH=path/to/the/model/directory
export DATASETS_ROOT=path/to/the/data/root
export OUT_DIR=path/where/you/want/your/results
# choose from available options
export DATASET=DeepForm
export OCR_ENGINE=microsoft_cv
# start the script to train 1D model
./examples/due/no_pretrain/train_1d_${DATASET}.sh
# or, start the script to train 2D model
./examples/due/no_pretrain/train_2d_${DATASET}.sh

```
The models presented in the paper differs only in two places. The first is the choice of `--relative_bias_args`.
T5 uses	`[{'type': '1d'}]` whereas `+2D` use `[{'type': '1d'}, {'type': 'horizontal'}, {'type': 'vertical'}]`

Note that in scripts some parameters regarding number of GPUS/workers are set up - feel free to modify them for you own use.
# 3. Evaluate
## 3.1 Evaluate on dev set for your own use
### 3.1.1 Postprocess outputs
In order to compare two files (generated by the model with the provided library and the gold-truth answers), one has to convert the generated output into a format that can be directly compared with `documents.jsonl`.
Please use:
```bash
python postprocessors/converter.py \
--test_generation ${OUT_DIR}/val_generations.txt \
--reference_path  ${DATASETS_ROOT}/${DATASET}/dev/document.jsonl \
--outpath ${OUT_DIR}/converted_val_generations.txt
```
For PWC dataset use a different postprocessor (available in the path `postprocessors/converter_pwc.py`)

### 3.1.2 Call evaluator
Finally outputs can be evaluated using the provided evaluator.
Assuming the evaluator was previously installed, its documentation can be accessed by using the `deval --help`.
To generate scores run:
```bash
deval --out-files ${OUT_DIR}/converted_test_generations.txt \
 --reference ${DATASETS_ROOT}/${DATASET}/test/document.jsonl \
  -m ${METRIC} -i
```
where `METRIC` should match the right metric for the chosen dataset (see evaluator's README.md).

The expected output should be formatted roughly like the following table:
```bash
       Label       F1  Precision   Recall
  advertiser 0.512909   0.513793 0.512027
contract_num 0.778761   0.780142 0.777385
 flight_from 0.794376   0.795775 0.792982
   flight_to 0.804921   0.806338 0.803509
gross_amount 0.355476   0.356115 0.354839
         ALL 0.649771   0.650917 0.648630
```

## 3.2 Prepare test set submission
In order to upload the answers to DUE benchmark website (https://duebenchmark.com/), one has to postprocess them like here:
```bash
python postprocessors/converter.py \
--test_generation ${OUT_DIR}/test_generations.txt \
--reference_path  ${DATASETS_ROOT}/${DATASET}/test/document.jsonl \
--outpath ${OUT_DIR}/converted_test_generations.txt
```
For PWC dataset use a different postprocessor (available in the path `postprocessors/converter_pwc.py`)

Then, just simply upload the `${OUT_DIR}/converted_test_generations.txt` to our website.
