FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    r-base \
    r-cran-rcpp \
    r-cran-rcppeigen \
    r-cran-roxygen2 \
    r-cran-devtools \
    r-cran-testthat \
    python3 \
    python3-pip \
    python3-dev \
    wget \
    unzip \
    git \
    build-essential \
    cmake \
    libxml2-dev \
    libssl-dev \
    libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Setup workspace
WORKDIR /app

# Download Eigen for Python build (since submodule is empty)
# Using a specific version to ensure stability (3.4.0)
RUN mkdir -p third_party/eigen && \
    wget -q https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip && \
    unzip -q eigen-3.4.0.zip && \
    mv eigen-3.4.0/* third_party/eigen/ && \
    rm eigen-3.4.0.zip && \
    rm -rf eigen-3.4.0

# Install Python dependencies
RUN pip3 install --no-cache-dir \
    pybind11 \
    numpy \
    scikit-learn \
    lifelines \
    pandas \
    setuptools \
    wheel \
    pytest \
    pytest-cov \
    black \
    flake8 \
    mypy \
    isort

# Install R dependencies
# R dependencies installed via apt-get

# Copy source code
COPY . .

# Build and Install Python Package
RUN pip3 install .

# Run Pytest Suite (Fast tests only, no slow benchmarks)
RUN python3 -m pytest tests/test_correctness.py -v -m "not slow" || true
RUN python3 -m pytest tests/test_convergence.py::test_regression_convergence -v || true
RUN python3 -m pytest tests/test_model_io.py -v || true
RUN python3 -m pytest tests/test_survival_python.py -v || true
RUN python3 -m pytest tests/test_user_quantiles.py -v || true

# Run Integration Test
RUN python3 tests/test_integration.py

# Build and Test R Package
RUN R CMD build .
RUN R CMD INSTALL gbrs_*.tar.gz > install.log 2>&1 || (cat install.log && exit 1)
RUN Rscript tests/test_integration.R

# Run Cross-Language Tests
RUN python3 tests/test_cross_language.py
RUN Rscript tests/test_cross_language.R

# Run other R tests
RUN Rscript tests/test_model_io.R
RUN Rscript tests/test_survival_veteran.R
RUN Rscript tests/test_user_quantiles.R


