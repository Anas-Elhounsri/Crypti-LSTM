FROM python:3.9

WORKDIR /lambda_lstm

COPY lambda_function.py /lambda_lstm/lambda_function.py

RUN pip install pandas numpy tensorflow scikit-learn

CMD ["python","lambda_function.py"]


