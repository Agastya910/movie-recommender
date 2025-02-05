# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file first to leverage Docker caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose port (if you are using FastAPI or Flask)
EXPOSE 8000

# Command to run the app
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
