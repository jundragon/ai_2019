# 2019 인공지능 챌린지

## 개요

2019 인공지능 R&D 챌린지 Track 1 솔루션

트랙1 시각지능 (상황인지) : 동영상 내에서 특정 인물, 사물을 찾아라



### 미션

500개의 video 클립 (30fps, 10초, video 클립당 이미지 300장)에서 등장하는 인물과 사물을 제한시간 5시간 안에 카운트 하기

단, video 클립에 한번 등장한 객체는 카운트 하지 않는다. 



#### output 예시

`t1_gt.json`

```json
{
  "track1_GT": [
    {
      "id": 1,
      "objects": [
          12,
          0,
          0,
          32,
          3,
          0
      ]
    }
  ]
}
```



objects 는 각각 순서대로 

사람, 소화기, 소화전, 자동차, 자전거, 오토바이의 수를 의미한다.



## 개발 환경

- postgreSQL(Django ORM)
- python 3.6
- docker 개발환경 (`./env/docker-compose.yml`)



개발 환경 설정을 docker 컨테이너 기반으로 자동화 하였다. 이유는 디바이스 환경마다 설치해야 하는 라이브러리의 버전이 달라지고 종속성이 꼬였기 때문...

가상환경 설정에 대한 정보는 `/env` 내부에 파일로 작성 되어 있다.

- docker-compose.yml : docker container 실행 정보

```yaml
version: '3.5'

services:
  db:
    image: postgres
    volumes:
      - ../pgdb:/var/lib/postgresql/data
    environment:
      - POSTGRES_DB=ai2019
      - POSTGRES_USER=userid
      - POSTGRES_PASSWORD=userpw
      - POSTGRES_INITDB_ARGS=--encoding=UTF-8
    ports:
      - "5432:5432"

  src:
    build:
      context: "../"
      shm_size: '8gb'
      dockerfile: /env/Dockerfile
    volumes:
      - ../src:/app
      - $HOME/data:/data
      - $HOME/models:/models
      - $HOME/results:/results
    links:
      - db
    environment:
      - DJANGO_DEBUG=True
      - DJANGO_DB_HOST=db
      - DJANGO_DB_PORT=5432
      - DJANGO_DB_NAME=ai2019
      - DJANGO_DB_USERNAME=userid
      - DJANGO_DB_PASSWORD=userpw
      - DJANGO_SECRET_KEY=dev_secret_key
    ports:
      - "8000:8000"
      - "6006:6006"
    command:
      - df -k /dev/shm
      - bash
      - -c
      - |
        /wait-for-it.sh db:5432 -t 10
        python manage.py makemigrations
        python manage.py migrate
        python run.py --input "/data/input/t1_video" --output "/results" --count 500 --step 10
```



- Dockerfile : docker container 빌드 정보

```dockerfile
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
```



## 라이브러리 (오픈소스)

- yolo https://github.com/eriklindernoren/PyTorch-YOLOv3.git
- siammask https://github.com/foolwood/SiamMask.git
- yolo data convert https://github.com/ssaru/convert2Yolo.git



## 준비
track1 data를 `$HOME/data` 에 받는다.

docker mount volume 정보를 변경하면 다른 경로에 지정할 수 있다.

`./env/docker-compose.yml`
```yaml
volumes:
  - $HOME/data:/data:/data
```



## RUN (by CLI)

in `docker container`

`run.py`

| param       | example                | description                                                  |
| ----------- | ---------------------- | ------------------------------------------------------------ |
| --input     | "/data/input/t1_video" | 비디오 클립이 위치하는 경로<br />디렉토리 하위에 t1_video_00001 비디오 번호가 필요하고<br />t1_video_00001_00001.jpg 이미지에도 비디오와 프레임 넘버가 필요하다 |
| --output    | "/data/output"         | 결과를 확인할 폴더를 지정하면 됨                             |
| --xml       | True or False          | yolo 분석 결과로 오토 라벨링을 위한 xml 출력 유무 지정<br />voc format |
| --txt       | True or False          | yolo 분석 결과로 오토 라벨링을 위한 txt 출력 유무 지정<br />yolo format |
| --start     | 10                     | 분석 비디오 시작위치 <br />만약 10으로 설정되어 있다면 t1_video_00010 부터 시작 |
| --step      | 2                      | 분석 프레임 step<br />만약 2로 설정 되어 있다면 2프레임마다 한번씩 분석 |
| --visual    | True or False          | output 경로에 분석 결과 정보를 이미지에 표시하여 저장한다. 단, 속도 느려짐 |
| --count     | 500                    | 시간 내 분석이 안될 경우도 있기에 dummy answer 를 만들기 위한 문제 문항수 설정 파라미터<br />500으로 설정되어 있다면 분석 결과에서 분석 중인 파일도 결과를 0으로 출력 |
| --sftp      | 127.0.0.1              | 분산 처리를 위한 sftp 통신 연결 정보 연결을 위한 ip 정보를 적음 |
| --sftp_home | "/home"       | sftp 목적지의 홈 경로                                        |
| --sftp_port | 22                     | 기본 22 포트                                                 |
| --sftp_id   | "id"             | sftp 접속 아이디                                             |
| --sftp_pw   | "password"             | sftp 접속 패스워드                                           |



### run

10 프레임 단위로 기본 설정으로 `/data/input/t1_video` 를 입력받아 분석하여 `/data/ouput` 에 출력 

```bash
$ python /src/run.py --input "/data/input/t1_video" --output "/data/output" --step 10
```


### ssh 연결해서 파일 중간 저장하고 로드하기

sftp 로 2대의 컴퓨터를 연동하여 중간 저장 파일을 확인하면서 분석하기

```bash
$ python /src/run.py --input "/data/input/t1_video" --output "/results" --count 500 --step 10 --sftp 192.168.3.51
```



### Auto Labeling

voc format 으로 yolo 분석 결과를 출력 (라벨링 파일)

```bash
$ python /src/run.py --xml True --input "/data/track1/t1_video" --output "/data/output"
```

라벨링 결과 파일은 이미지와 함께 https://github.com/tzutalin/labelImg 이 프로그램으로 편집이 가능하다.



### result (대회 결과 출력하기)

분석이 안된 비디오도 우선 count에 지정한 개수로 더미 데이터를 출력 (채점용)

```bash
$ python /src/result.py --input "data/output" --output "data/result" --count 500
```



### 비디오 파일을 프레임으로 나누기

입력 소스가 비디오 파일이라면 지정한 프레임으로 나누어 이미지로 저장 

```bash
$ python /src/video2frame.py --input "/data/video" --output "/data/video/output"
```



### yolo 전이학습 시키기

```bash
$ python /src/train.py
```
