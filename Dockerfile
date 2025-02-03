# Step 1: Use the official TensorFlow 2.11 image as the base
FROM tensorflow/tensorflow:2.11.0

# Step 2: Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Step 3: Upgrade pip to the latest version
RUN python3 -m pip install --upgrade pip

# Step 4: Copy your requirements file into the container
COPY requirements_GenNet.txt /tmp/requirements_GenNet.txt

# Step 5: Install Python packages
RUN pip install --no-cache-dir -r /tmp/requirements_GenNet.txt

# Step 6: Set the working directory
WORKDIR /app

# Step 7: Copy your project files into the container
COPY . /app

# Step 8: Set environment variables (optional)
ENV RESULT_PATH="/app/results"
ENV DATA_PATH="/app/examples"

# Step 9: Define the entrypoint to simplify CLI usage
ENTRYPOINT ["python", "GenNet.py"]
