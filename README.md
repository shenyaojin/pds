# Pressure Diffusion Simulator

## Introduction

Python library to simulate the wellbore pressure diffusion in Mariner. Should combine with DSS analyzer.

**Keywords**:
- 1D simulator (I'll migrate to MOOSE while facing 2D)
- FD method
- Using Mariner's PG data as a source / synthetic data (future work)
- Time sampling optimizer included

## Modules

- Core
  - Define the core classes and functions
- Optimizer
  - Time sampling optimizer
- Solver
  - FD(currently) solver
- Test
  - Test the core functions

## Usage


I'm using RCP's Midland and my personal Linux workstation, Haynesville, to develop/debug the library simultaneously; the commit would be quite often.

For now, this code is for personal research use only and only accepts the source term file format from Bakken Mariner. I can't promise it can run in your local environment.


## Future plan

I feel it's starting to be a complicated tool and hard to maintain multiple repos at the same time. 
So, I decided to integrate the data analysis and simulation part into a new repository.
