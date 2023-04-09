# P1_P2_detection

Since it is not possible to publicly share intracranial pressure signals from real patients, a synthetic toy dataset is provided as an example to run the analysis pipeline. Therefore, the results obtained with it have no scientific significance.

**To run the pipeline:**

With conda activated:

either:  
$ conda env create -f workflow/envs/venv.yaml  
$ conda activate venv  
$ cd workflow  
$ snakemake -s Snakefile --cores 8  

or, if snakemake is already installed in your current environment:  
$ snakemake -s Snakefile --cores 8 --use-conda
