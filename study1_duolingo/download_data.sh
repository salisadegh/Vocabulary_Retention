#!/usr/bin/env bash
# Downloads the Duolingo learning-traces dataset (~361 MB gz, ~12.85M rows)
# Settles & Meeder (2016), ACL. See the Dataverse record for licence terms.
set -e
mkdir -p data/duolingo_raw
URL="https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/N8XJME/0OTHXW"
echo "Fetching dataset from Harvard Dataverse (doi:10.7910/DVN/N8XJME)..."
curl -L "$URL" -o data/duolingo_raw/settles.acl16.learning_traces.13m.csv.gz
ls -lh data/duolingo_raw/
echo "Next: python study1_duolingo/make_sample.py"
