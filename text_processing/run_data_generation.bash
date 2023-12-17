#!/usr/bin/env bash
# -*- coding:utf-8 -*-

# for data_format in entity relation event absa
for data_format in entity
do
    python uie_convert.py -format spotasoc -config data_config/${data_format}/twitter2015.yaml -output ${data_format}
done

python scripts/data_statistics.py -data converted_data/text2spotasoc/
