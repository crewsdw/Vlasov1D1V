# [Vlasov1D1V](https://github.com/crewsdw/Vlasov1D1V/)

This collection of Python codes solves the Vlasov-Poisson system in 1D+2V with high-order accuracy using the Runge-Kutta Discontinuous Galerkin finite element method and CUDA-accelerated libraries for tensor products, namely [CuPy](https://github.com/cupy/cupy).

## About:
To read about the theory behind this project and some theoretical studies using it, check out Chapter 3 (Numerical Methods) of this document:
[Crews PhD General Exam](https://students.washington.edu/dcrews/documents/GenExamR2.pdf)

## Use
You'll need a CUDA-compatible GPU and an install of the CuPy library.

To use the scipts: download the files, make a data folder, and adjust the run parameters (resolutions, domain length, etc.) at the beginning of the "main.py" file.
To play around with different initial conditions, adjust the eigenvalue parameter "om" (for frequency, omega) during initialization of the distribution function.
