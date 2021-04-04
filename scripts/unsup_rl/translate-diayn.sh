#!/usr/bin/env bash

mkdir result
#ckpt=/data/jive/multi30k-ozan-models/en-de/en_de-cgru-nmt-bidir/nmt-r34885-val030.best.bleu_40.310.ckpt
ckpt=/data/jive/multi30k-ozan-models/en-de/en_de-cgru-nmt-bidir/nmt-r34885-val034.best.loss_1.751.ckpt
mode=xe
home_dir=/data/jive/sim2/simmt
data_dir=/data/jive/sim2/simmt/data/multi30k/en-de
multeval_dir=/data/jive/multeval-0.5.1

CUDA_VISIBLE_DEVICES=0 nmtpy translate -S src:mltd-mscoco/src.txt -s mscoco-words -o ${mode} ${ckpt}
sed -r 's/(@@ )|(@@ ?$)//g' ${mode}.mscoco-words.beam6 > result/${mode}.mscoco-words.beam6
CUDA_VISIBLE_DEVICES=0 nmtpy translate -S src:mltd-flickr/src.txt -s flickr-words -o ${mode} ${ckpt}
sed -r 's/(@@ )|(@@ ?$)//g' ${mode}.flickr-words.beam6 > result/${mode}.flickr-words.beam6
CUDA_VISIBLE_DEVICES=0 nmtpy translate -S src:mltd-flickr6/src.txt -s flickr6-words -o ${mode} ${ckpt}
sed -r 's/(@@ )|(@@ ?$)//g' ${mode}.flickr6-words.beam6 > result/${mode}.flickr6-words.beam6
CUDA_VISIBLE_DEVICES=0 nmtpy translate -S src:../../multi30k-dataset/data/task1/tok/test_2017_mscoco.lc.norm.tok.en -s mscoco -o ${mode} ${ckpt}
sed -r 's/(@@ )|(@@ ?$)//g' ${mode}.mscoco.beam6 > result/${mode}.mscoco.beam6
CUDA_VISIBLE_DEVICES=0 nmtpy translate -s test_2017_flickr -o ${mode} ${ckpt}
sed -r 's/(@@ )|(@@ ?$)//g' ${mode}.test_2017_flickr.beam6 > result/${mode}.test_2017_flickr.beam6
CUDA_VISIBLE_DEVICES=0 nmtpy translate -s test_2016_flickr -o ${mode} ${ckpt}
sed -r 's/(@@ )|(@@ ?$)//g' ${mode}.test_2016_flickr.beam6 > result/${mode}.test_2016_flickr.beam6

#ckpt=/data/jive/multi30k-ozan-models/de-diyan3/en_de-cgru-nmt-bidir-diyan/nmt-r7e75d-val022.best.bleu_39.750.ckpt
#ckpt=/data/jive/multi30k-ozan-models/de-diyan3/en_de-cgru-nmt-bidir-diyan/nmt-r10a58-val044.best.bleu_40.310.ckpt
# alpha 1
#ckpt=/data/jive/multi30k-ozan-models/de-diyan3/en_de-cgru-nmt-bidir-diyan/nmt-r819b0-val026.best.bleu_40.310.ckpt
# alpha 001 no reg
#ckpt=/data/jive/multi30k-ozan-models/de-diyan3/en_de-cgru-nmt-bidir-diyan/nmt-ra13f3-val021.best.bleu_40.310.ckpt
#ckpt=/data/jive/multi30k-ozan-models/de-diyan3/en_de-cgru-nmt-bidir-diyan/nmt-re9390-val046.best.bleu_40.310.ckpt
#ckpt=/data/jive/multi30k-ozan-models/de-diyan3/en_de-cgru-nmt-bidir-diyan2lossdecay/nmt-r19c0f-val015.best.bleu_40.070.ckpt
ckpt=/data/jive/multi30k-ozan-models/de-diyan3/en_de-cgru-nmt-bidir-diyan2lossdecay/nmt-r0226c-val009.best.bleu_40.000.ckpt
mode=diayn

CUDA_VISIBLE_DEVICES=0 nmtpy translate -S src:mltd-mscoco/src.txt -s mscoco-words -o ${mode} ${ckpt}
sed -r 's/(@@ )|(@@ ?$)//g' ${mode}.mscoco-words.beam6 > result/${mode}.mscoco-words.beam6
CUDA_VISIBLE_DEVICES=0 nmtpy translate -S src:mltd-flickr/src.txt -s flickr-words -o ${mode} ${ckpt}
sed -r 's/(@@ )|(@@ ?$)//g' ${mode}.flickr-words.beam6 > result/${mode}.flickr-words.beam6
CUDA_VISIBLE_DEVICES=0 nmtpy translate -S src:mltd-flickr6/src.txt -s flickr6-words -o ${mode} ${ckpt}
sed -r 's/(@@ )|(@@ ?$)//g' ${mode}.flickr6-words.beam6 > result/${mode}.flickr6-words.beam6
CUDA_VISIBLE_DEVICES=0 nmtpy translate -S src:../../multi30k-dataset/data/task1/tok/test_2017_mscoco.lc.norm.tok.en -s mscoco -o ${mode} ${ckpt}
sed -r 's/(@@ )|(@@ ?$)//g' ${mode}.mscoco.beam6 > result/${mode}.mscoco.beam6
CUDA_VISIBLE_DEVICES=0 nmtpy translate -s test_2017_flickr -o ${mode} ${ckpt}
sed -r 's/(@@ )|(@@ ?$)//g' ${mode}.test_2017_flickr.beam6 > result/${mode}.test_2017_flickr.beam6
CUDA_VISIBLE_DEVICES=0 nmtpy translate -s test_2016_flickr -o ${mode} ${ckpt}
sed -r 's/(@@ )|(@@ ?$)//g' ${mode}.test_2016_flickr.beam6 > result/${mode}.test_2016_flickr.beam6

cd $multeval_dir

./multeval.sh eval --refs ${data_dir}/test_2017_flickr.lc.norm.tok.de --hyps-baseline ${home_dir}/result/xe.test_2017_flickr.beam6 --hyps-sys1 ${home_dir}/result/diayn.test_2017_flickr.beam6  --meteor.language de --latex table.tex --rankDir rank
mv table.tex ${home_dir}/result/table.tex.test_2017_flickr
./multeval.sh eval --refs ${data_dir}/test_2016_flickr.lc.norm.tok.de --hyps-baseline ${home_dir}/result/xe.test_2016_flickr.beam6 --hyps-sys1 ${home_dir}/result/diayn.test_2016_flickr.beam6  --meteor.language de --latex table.tex --rankDir rank
mv table.tex ${home_dir}/result/table.tex.test_2016_flickr
./multeval.sh eval --refs ${data_dir}/test_2016_flickr.lc.norm.tok.de /data/jive/multi30k/en-de/test_2016.lc.norm.tok.1.de /data/jive/multi30k/en-de/test_2016.lc.norm.tok.2.de /data/jive/multi30k/en-de/test_2016.lc.norm.tok.3.de /data/jive/multi30k/en-de/test_2016.lc.norm.tok.4.de /data/jive/multi30k/en-de/test_2016.lc.norm.tok.5.de --hyps-baseline ${home_dir}/result/xe.test_2016_flickr.beam6 --hyps-sys1 ${home_dir}/result/diayn.test_2016_flickr.beam6  --meteor.language de --latex table.tex --rankDir rank
mv table.tex ${home_dir}/result/table.tex.test_2016_flickr_multi
./multeval.sh eval --refs /data/jive/multi30k-dataset/data/task1/tok/test_2017_mscoco.lc.norm.tok.de --hyps-baseline ${home_dir}/result/xe.mscoco.beam6 --hyps-sys1 ${home_dir}/result/diayn.mscoco.beam6  --meteor.language de --latex table.tex --rankDir rank
mv table.tex ${home_dir}/result/table.tex.mscoco
