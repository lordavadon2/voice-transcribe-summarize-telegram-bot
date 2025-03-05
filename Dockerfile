FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

VOLUME /app/data
ENV CONFIG_PATH=/app/data/config.json
ENV USERS_PATH=/app/data/authorized_users.json

CMD ["python", "bot.py"]