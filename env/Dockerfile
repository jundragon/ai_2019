FROM ufoym/deepo:pytorch-py36-cu100

ENV PYTHONUNBUFFERED 0

RUN apt-get update && apt-get -y upgrade && apt-get -y install wget
RUN sh -c 'echo "deb http://apt.postgresql.org/pub/repos/apt/ `lsb_release -cs`-pgdg main" >> /etc/apt/sources.list.d/pgdg.list'
RUN wget -q https://www.postgresql.org/media/keys/ACCC4CF8.asc -O - | apt-key add -
RUN apt-get update && apt-get -y upgrade && apt-get -y install libpq-dev
RUN apt-get install -y libsm6 libxext6 libxrender-dev

WORKDIR /app

ADD	./requirements.txt	/app/
ADD ./env/wait-for-it.sh    /
RUN	pip install -r requirements.txt

# 소스 추가 부분
ADD ./src          /app/