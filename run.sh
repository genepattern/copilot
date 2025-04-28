#!/bin/bash

docker run --rm -p 8050:8050 \
  -v $PWD/.env:/srv/copilot/.env \
  -v $PWD/gcp.json:/srv/copilot/gcp.json \
  -v $PWD/db.sqlite3:/srv/copilot/db.sqlite3 genepattern/copilot:0.2