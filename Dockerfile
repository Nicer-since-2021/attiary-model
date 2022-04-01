FROM python:3.8.5

WORKDIR /app
COPY . .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 5001

CMD python ./app.py
# CMD ["python3", "-m", "flask", "run", "--host=0.0.0.0"]