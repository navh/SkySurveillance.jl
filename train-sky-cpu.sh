#!/bin/bash
#SBATCH --account=def-rsadve
#SBATCH --mail-user=<a.hebb@mail.utoronto.ca>
#SBATCH --mail-type=ALL

#SBATCH --nodes 1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8

#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=t-stdenv23-julia19-%N-%j.out

module purge
module load StdEnv/2023
module load julia

julia --project src/SkySurveillance.jl params-test.toml