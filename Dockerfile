# Step 1: Use the official TensorFlow 2.11 image as the base
FROM tensorflow/tensorflow:2.11.0

# Step 2: Install any system dependencies you need
# (optional: only if your Python packages or code require system libs)
RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Step 3: Copy your requirements file into the container
COPY requirements_GenNet.txt /tmp/requirements_GenNet.txt

# Step 4: Install your pinned Python packages via pip
# Use --no-cache-dir to keep the image smaller
RUN pip install --no-cache-dir -r /tmp/requirements_GenNet.txt

# Step 5: Set a working directory (example)
WORKDIR /app

# Step 6: Copy your Python command-line tool code into /app
COPY . /app

# Step 7: Define entrypoint or command to run your CLI tool by default
# e.g., if your main Python file is "mycli.py"
ENTRYPOINT ["python", "GenNet.py"]
