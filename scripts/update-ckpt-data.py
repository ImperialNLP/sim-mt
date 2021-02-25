#!/usr/bin/env python
import os
import sys
import torch


from nmtpytorch.utils.misc import load_pt_file


if __name__ == '__main__':

    for fname in sys.argv[1:]:
        model_dict = load_pt_file(fname)

        # copy data src
        src = dict(model_dict['opts']['data']['test_2017_flickr_set'])

        for new_set in ('test_2017_mscoco', 'test_2018_flickr'):
            d = {}
            for key, value in src.items():
                # new path
                value = value.parent / value.name.replace('test_2017_flickr', new_set)
                if os.path.exists(value):
                    d[key] = value
                else:
                    print('missing ', value)

            model_dict['opts']['data'][f'{new_set}_set'] = d

        torch.save(model_dict, fname)
