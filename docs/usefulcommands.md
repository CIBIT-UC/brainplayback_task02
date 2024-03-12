lfs quota -uh alexsayal ~

squeue -u alexsayal

sbatch fmriprep-batch.sh

sacct -s r -X --format=JobID,JobName%12,Priority,Elapsed,NCPU,CPUTime,ExitCode,State

salloc -p hpc --job-name nilearntest



# jupyter
fonte: https://adam-streck.medium.com/creating-persistent-jupyter-notebooks-on-a-cluster-using-vs-code-slurm-and-conda-140b922a97a8

iniciar sessao interativa

screen

ativar conda env

jupyter-notebook --no-browser --ip=0.0.0.0 --port 8888

copiar url 

no vs code conectado por ssh usar esse python environment

