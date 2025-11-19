FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    r-base \
    python3 \
    python3-pip \
    python3-dev \
    wget \
    unzip \
    git \
    build-essential \
    cmake \
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
    setuptools \
    wheel

# Install R dependencies
RUN Rscript -e "install.packages(c('Rcpp', 'RcppEigen'), repos='https://cloud.r-project.org')"

# Copy source code
COPY . .

# Build and Test R Package
RUN R CMD build .
RUN R CMD INSTALL gbrs_0.0.0.9000.tar.gz
RUN Rscript tests/test_integration.R

# Build and Test Python Package
RUN pip3 install .
RUN python3 tests/test_integration.py
