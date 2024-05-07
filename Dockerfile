FROM pytorch/pytorch:2.3.0-cuda11.8-cudnn8-runtime

# Set the working directory
WORKDIR .

# Install requirements
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the files
COPY . .
# CMD ["python", "src/main.py"]