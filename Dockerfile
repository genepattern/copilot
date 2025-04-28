FROM python:3.11-slim

# Install system dependencies
RUN apt-get -y update && \
    apt-get -y install wget git bzip2 libcurl4-gnutls-dev gcc nano

# Install GenePattern Copilot
RUN git clone https://github.com/genepattern/copilot.git /srv/copilot
WORKDIR /srv/copilot
RUN pip install -r requirements.txt

# Build the chroma database
RUN cd vectorstore &&  \
    python build_database.py

# Set up and run the server
RUN python manage.py makemigrations && \
    python manage.py migrate && \
    python manage.py collectstatic --noinput
EXPOSE 8000
CMD python manage.py runserver 0.0.0.0:8050