# Enzyme-GLM   
Enzyme-GLM is a model for enzyme-substrate compatibility judgment based on a pre-trained bilingual large model fine-tuned,    
which comes in cls and blank versions. With Enzyme-GLM, you can achieve the following objectives:

# Preparation   

Before commencing the use of Enzyme-GLM, you are required to undertake the preparatory work as follows:   

1. Glone this repo   
```shell
git clone https://github.com//YinhuiQiao/Enzyme-GLM   
cd Enzyme-GLM
```

2. Please first install dependencies by `pip install -r requirements.txt`

3. You should get checkpoints data from here(https:xxx),   
then ensure they are placed under the checkpoint file: './checkpoint/'.

# Usage

## 1. Use cls for enzyme-substrate compatibility judgment:
Modify the data loading section in the `cls_predict.py` file from "line 108 test_dataset = dataset('../data/price/price.csv',MAXK_LENGTH,tokenizer=tokenizer)" to the test data path.
Modify the output path in "line 145 with open('../data/result/cls0808_result_price.csv', 'w', newline='') as f:".
Run the command:
```shell
python cls_predict.py
```

## 2. Use blank for enzyme-substrate compatibility judgment:
Modify the data loading section in the `blank_predict.py` file from "line 114 test_dataset = dataset('../data/enzyme_substrate_pair_test.csv',MAXK_LENGTH,tokenizer=tokenizer)" to the test data path.
Modify the output path in "line 152 with open('../data/result/blank_result_seed_all_test.csv', 'w', newline='') as f:".
Run the command:
```shell
python blank_predict.py
```

## 3. Use cls to assign substrates to enzymes
Write the target enzyme into 'find_substrate_example.csv', and choose an appropriate top value   
(default is 10, meaning the top ten molecules with the highest scores are considered as potential substrates for the enzyme and output).
For each enzyme to be screened, the code will match all SMILES in the molecule library to predict compatibility scores.   
You can see all the molecules in the file “all_SMI_plusplus_normalized_TFYG.csv” and add required additional molecules in the established format in this file.
Run the command:
```shell
python find_substrate.py
```

## 4. Use cls to assign enzymes to molecules
Write the target molecule into 'find_enzyme_example.csv', and choose an appropriate top value   
(default is 10, meaning the top 10 enzymes with the highest scores are considered as potential enzymes for the molecule and output).
For each molecule to be screened, the code will match all amino acid sequences in the enzyme library and predict compatibility scores.   
You can see all the enzymes in the file “all_proteins_sequence.json” and add required additional enzymes in the established format in this file.
Run the command:
```shell
python find_enzyme.py
```

## 5. Use the bilingual model to get the Embedding of sequences
Write the target molecule into 'get_embedding.csv', and specify the sequence type (proteins or molecule).
Run the command:
```shell
python Bilingula_GLM.py
```