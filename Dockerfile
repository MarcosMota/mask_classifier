FROM python:3.9

RUN apt-get update && apt-get install -y python3-opencv
RUN pip install opencv-python

WORKDIR /app

COPY requirements.txt requirements.txt


RUN pip install -r requirements.txt


EXPOSE 8501

COPY . .

ENTRYPOINT [ "streamlit", "run" ]

CMD [  "app.py" ]



