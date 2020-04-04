FROM nvcr.io/nvidia/pytorch:20.03-py3

RUN apt-get update
# We copy just the requirements.txt first to leverage Docker cache
COPY ./requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip install -r requirements.txt

RUN pip install dlib

RUN apt-get install ffmpeg -y

COPY . /app

ENTRYPOINT [ "python" ]

CMD [ "server.py" ]