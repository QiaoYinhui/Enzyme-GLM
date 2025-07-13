# Enzyme-GLM-gen

Enzyme-GLM-gen is an enzyme generation model trained on GLM (Genrral Language Model). It can perfrom  
unconditional production of amino acid sequences and conditional enzyme generation based on molecular   
prompts.

Before using Enzyme-GLM-gen, please ensure you meet the following requirements:

- 1.Have a computer running a Linux-based operating system
- 2.Install the dependencies. Please refer to the link: (https://github.com/THUDM/GLM)
bash scripts/generate_Pro_condition.sh
- 3.Download the model weights(Enzyme-GLM-304M-s2eV2_2-8_10e01-22-13-02) from link https://zenodo.org/records/15868721 and modify the model weights path in the .sh file.
(generate_Pro_uncondition.sh and generate_Pro_condition.sh)


### unconditional production of amino acid sequences
```shell
bash scripts/generate_Pro_uncondition.sh
```

### conditional enzyme generation based on molecular prompts
```shell
bash scripts/generate_Pro_condition.sh
```
