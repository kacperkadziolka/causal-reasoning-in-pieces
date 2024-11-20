# How to contribute to the University Labs

To simply excel the development, there is an option to connect your locally running jupyter notebook to the kernel server
hosted on the University's lab computer. This way you can use the computational power of the lab computer to run your experiments.

## Environment

To avoid the quota exceeded error, instead of spinning up the jupyter server under the `/home` directory, create a conda 
virtual environment in the `/local` directory. This way you can install all the necessary packages without worrying about
the space constraints.

1. Create a conda environment in the `/local` directory
```bash
conda create -p /local/$USER/conda_envs/my_conda_env
```
2. Activate the environment
```bash
conda activate /local/$USER/conda_envs/my_conda_env
```
3. Install the necessary package to run the jupyter notebook:
```bash
conda install jupyter
```

## Running the jupyter server

1. First generate the jupyter configuration file:
```bash
jupyter notebook --generate-config
```
2. Generate the password for authentication:
```bash
jupyter notebook password
```
3. Start the jupyter server:
```bash
jupyter notebook --no-browser --port=8888 --ip=0.0.0.0
```