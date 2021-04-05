#!/usr/bin/env bash

mkdir result
ckpt=/data/jive/multi30k-ozan-models/en-fr/en_fr-cgru-nmt-bidir/nmt-r40f92-val022.best.bleu_57.960.ckpt
mode=xe
home_dir=/data/jive/sim2/simmt
data_dir=/data/jive/sim2/simmt/data/multi30k/en-fr
multeval_dir=/data/jive/multeval-0.5.1

CUDA_VISIBLE_DEVICES=0 nmtpy translate -S src:mltd-mscoco-fr/src.txt -s mscoco-words -o ${mode} ${ckpt}
sed -r 's/(@@ )|(@@ ?$)//g' ${mode}.mscoco-words.beam6 > result/${mode}.mscoco-words.beam6
CUDA_VISIBLE_DEVICES=0 nmtpy translate -S src:mltd-flickr-fr/src.txt -s flickr-words -o ${mode} ${ckpt}
sed -r 's/(@@ )|(@@ ?$)//g' ${mode}.flickr-words.beam6 > result/${mode}.flickr-words.beam6
CUDA_VISIBLE_DEVICES=0 nmtpy translate -S src:mltd-flickr6-fr/src.txt -s flickr6-words -o ${mode} ${ckpt}
sed -r 's/(@@ )|(@@ ?$)//g' ${mode}.flickr6-words.beam6 > result/${mode}.flickr6-words.beam6
CUDA_VISIBLE_DEVICES=0 nmtpy translate -S src:../../multi30k-dataset/data/task1/tok/test_2017_mscoco.lc.norm.tok.en -s mscoco -o ${mode} ${ckpt}
sed -r 's/(@@ )|(@@ ?$)//g' ${mode}.mscoco.beam6 > result/${mode}.mscoco.beam6
CUDA_VISIBLE_DEVICES=0 nmtpy translate -s test_2017_flickr -o ${mode} ${ckpt}
sed -r 's/(@@ )|(@@ ?$)//g' ${mode}.test_2017_flickr.beam6 > result/${mode}.test_2017_flickr.beam6
CUDA_VISIBLE_DEVICES=0 nmtpy translate -s test_2016_flickr -o ${mode} ${ckpt}
sed -r 's/(@@ )|(@@ ?$)//g' ${mode}.test_2016_flickr.beam6 > result/${mode}.test_2016_flickr.beam6

#ckpt=/data/jive/multi30k-ozan-models/fr-diyan3/en_fr-cgru-nmt-bidir/nmt-rf64e9-val021.best.bleu_57.440.ckpt
#ckpt=/data/jive/multi30k-ozan-models/fr-diyan3/en_fr-cgru-nmt-bidir/nmt-r727b5-val041.best.bleu_58.420.ckpt
ckpt=/data/jive/multi30k-ozan-models/fr-diyan4/en_fr-cgru-nmt-bidir/nmt-r175f0-val026.best.bleu_57.960.ckpt
mode=diayn

CUDA_VISIBLE_DEVICES=0 nmtpy translate -S src:mltd-mscoco-fr/src.txt -s mscoco-words -o ${mode} ${ckpt}
sed -r 's/(@@ )|(@@ ?$)//g' ${mode}.mscoco-words.beam6 > result/${mode}.mscoco-words.beam6
CUDA_VISIBLE_DEVICES=0 nmtpy translate -S src:mltd-flickr-fr/src.txt -s flickr-words -o ${mode} ${ckpt}
sed -r 's/(@@ )|(@@ ?$)//g' ${mode}.flickr-words.beam6 > result/${mode}.flickr-words.beam6
CUDA_VISIBLE_DEVICES=0 nmtpy translate -S src:mltd-flickr6-fr/src.txt -s flickr6-words -o ${mode} ${ckpt}
sed -r 's/(@@ )|(@@ ?$)//g' ${mode}.flickr6-words.beam6 > result/${mode}.flickr6-words.beam6
CUDA_VISIBLE_DEVICES=0 nmtpy translate -S src:../../multi30k-dataset/data/task1/tok/test_2017_mscoco.lc.norm.tok.en -s mscoco -o ${mode} ${ckpt}
sed -r 's/(@@ )|(@@ ?$)//g' ${mode}.mscoco.beam6 > result/${mode}.mscoco.beam6
CUDA_VISIBLE_DEVICES=0 nmtpy translate -s test_2017_flickr -o ${mode} ${ckpt}
sed -r 's/(@@ )|(@@ ?$)//g' ${mode}.test_2017_flickr.beam6 > result/${mode}.test_2017_flickr.beam6
CUDA_VISIBLE_DEVICES=0 nmtpy translate -s test_2016_flickr -o ${mode} ${ckpt}
sed -r 's/(@@ )|(@@ ?$)//g' ${mode}.test_2016_flickr.beam6 > result/${mode}.test_2016_flickr.beam6

cd $multeval_dir

./multeval.sh eval --refs ${data_dir}/test_2017_flickr.lc.norm.tok.fr --hyps-baseline ${home_dir}/result/xe.test_2017_flickr.beam6 --hyps-sys1 ${home_dir}/result/diayn.test_2017_flickr.beam6  --meteor.language fr --latex table.tex --rankDir rank
mv table.tex ${home_dir}/result/table.tex.test_2017_flickr
./multeval.sh eval --refs ${data_dir}/test_2016_flickr.lc.norm.tok.fr --hyps-baseline ${home_dir}/result/xe.test_2016_flickr.beam6 --hyps-sys1 ${home_dir}/result/diayn.test_2016_flickr.beam6  --meteor.language fr --latex table.tex --rankDir rank
mv table.tex ${home_dir}/result/table.tex.test_2016_flickr
./multeval.sh eval --refs /data/jive/multi30k-dataset/data/task1/tok/test_2017_mscoco.lc.norm.tok.fr --hyps-baseline ${home_dir}/result/xe.mscoco.beam6 --hyps-sys1 ${home_dir}/result/diayn.mscoco.beam6  --meteor.language fr --latex table.tex --rankDir rank
mv table.tex ${home_dir}/result/table.tex.mscoco
