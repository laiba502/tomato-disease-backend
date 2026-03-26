# Use Python 3.10
FROM python:3.10

# Set the working directory
WORKDIR /code

# Copy the requirements first to cache them
COPY ./requirements.txt /code/requirements.txt

# Install the libraries
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy all your project files (including models)
COPY . .

# Start the FastAPI app on port 7860 (Hugging Face's favorite port)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]