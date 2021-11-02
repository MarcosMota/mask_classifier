FROM jjanzic/docker-python3-opencv

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt
RUN pip install opencv-python
RUN apt-get install libgl

EXPOSE 8501

COPY . .

ENTRYPOINT [ "streamlit", "run" ]

CMD [  "app.py" ]



