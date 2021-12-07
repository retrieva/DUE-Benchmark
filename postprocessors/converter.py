#!/usr/bin/env python3
"""Scripts that convert T5-model outputs to format that can be directly compared with `documents.jsonl`"""

from glob import glob
import json
import os
from collections import defaultdict
import fire


def main(test_generation, reference_path, outpath):
    data = defaultdict(list)

    with open(test_generation) as raw_out:
        for line in raw_out:
            line = json.loads(line)
            doc_id = line['doc_id'].split('__')[0]
            if not (line['label_name'], line['preds']) in data[doc_id]:
                data[doc_id].append((line['label_name'], line['preds']))

    with open(reference_path) as expected, open(outpath, 'w+') as output:
        for line in expected:
            line = json.loads(line)
            ans = []
            for key, val in data[line['name']]:
                key = key.rstrip('=')
                vals = [v.strip() for v in val.split(' | ')]
                ans.append({'key': key, 'values': [{'value': val} for val in vals]})

            ans_doc = {'name': line['name'], 'annotations': ans}
            output.write(json.dumps(ans_doc) + '\n')


if __name__ == "__main__":
    fire.Fire(main)

