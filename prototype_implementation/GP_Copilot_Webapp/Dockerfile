FROM python:3.10-slim


RUN apt-get -y update && \
    apt-get -y install wget git bzip2 libcurl4-gnutls-dev gcc nano

COPY ./requirements.txt /src/requirements.txt
RUN python3 -m venv /opt/venv
RUN python3 -m pip install --upgrade pip
RUN /bin/bash -c "source /opt/venv/bin/activate && \
     pip3 install -r /src/requirements.txt"


WORKDIR /srv/copilot
COPY copilot/ /srv/copilot/copilot
COPY GP_Copilot/ /srv/copilot/GP_Copilot
COPY static/ /srv/copilot/static
COPY ./manage.py /srv/copilot
COPY ./run-manage-py.sh /srv/copilot
RUN mkdir -p /chroma
COPY chroma/ /chroma
RUN mkdir -p uploads
RUN mkdir -p validation_files
COPY validation_files/validation_results_full.csv validation_files/validation_results_full.csv


RUN /bin/bash -c "source /opt/venv/bin/activate && \
	/srv/copilot/manage.py collectstatic --noinput"


RUN mkdir -p /srv/logs/

# COPY get_model.sh /srv/copilot
# RUN chmod 777 /srv/copilot/get_model.sh
# RUN /srv/copilot/get_model.sh


EXPOSE 8000

# CMD ['source', '/opt/venv/bin/activate', '&&', 'python3', 'manage.py', 'runserver', '0.0.0.0:8000']

# https://betterstack.com/community/guides/scaling-python/dockerize-django/ 
