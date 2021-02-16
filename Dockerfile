# Dockerfile
FROM python:3.7.5
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . /app
WORKDIR /app
EXPOSE 8050
CMD python main_v2.py
