FROM python:3.14-slim

WORKDIR /app

COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

COPY . .

EXPOSE 8000
CMD ["python", "api.py"]
