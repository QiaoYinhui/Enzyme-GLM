# Enzyme-GLM-gen

Enzyme-GLM-gen is an enzyme generation model trained on GLM (Genrral Language Model). It can perfrom  
unconditional production of amino acid sequences and conditional enzyme generation based on molecular   
prompts.

Before using Enzyme-GLM-gen, please ensure you meet the following requirements:

- 1.Have a computer running a Linux-based operating system
- 2.Install the dependencies. Please refer to the link: (https://github.com/THUDM/GLM)


### unconditional production of amino acid sequences
```shell
bash scripts/generate_Pro_uncondition.sh
```

### conditional enzyme generation based on molecular prompts
```shell
bash scripts/generate_Pro_condition.sh
```
