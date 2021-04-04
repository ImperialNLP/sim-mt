#!/usr/bin/env bash
in_file=test_2016_flickr
mkdir result
ckpt=/data/jive/ozanmodels/iwslt-prep/nmt-re11d4-val041.best.loss_2.192.ckpt
#ckpt=/data/jive/ozanmodels/iwslt-prep/nmt-re11d4-val049.best.bleu_29.740.ckpt
#ckpt=/data/jive/ozanmodels2/iwslt-init/nmt-r8eb1a-val020.best.bleu_27.410.ckpt
mode=xe
home_dir=/data/jive/sim2/simmt
data_dir=/data/jive/simmt/data/iwslt
multeval_dir=/data/jive/multeval-0.5.1

#CUDA_VISIBLE_DEVICES=1 nmtpy translate -s test_2017_flickr -o ${mode} ${ckpt}
#cp ${mode}.test_2017_flickr.beam6 result/${mode}.test

CUDA_VISIBLE_DEVICES=1 nmtpy translate -S src:/data/jive/simmt/data/multi30k/en-de/${in_file}.lc.norm.tok.de -s mscoco -o ${mode} ${ckpt}
cp ${mode}.mscoco.beam6 result/

ckpt=/data/jive/ozanmodels-iwslt/iwslt3/nmt-r4470a-val012.best.bleu_29.580.ckpt
#ckpt=/data/jive/ozanmodels-iwslt/iwslt/nmt-r6716c-val010.best.bleu_29.740.ckpt
#ckpt=/data/jive/ozanmodels-iwslt/iwslt2/nmt-r35118-val007.best.bleu_27.540.ckpt
#ckpt=/data/jive/ozanmodels-iwslt/iwslt2/nmt-r527c3-val011.best.bleu_29.030.ckpt
#ckpt=/data/jive/ozanmodels-iwslt/iwslt/nmt-r22747-val013.best.bleu_29.740.ckpt
mode=sac

#CUDA_VISIBLE_DEVICES=1 nmtpy translate -s test_2017_flickr -o ${mode} ${ckpt}
#cp ${mode}.test_2017_flickr.beam6 result/${mode}.test
CUDA_VISIBLE_DEVICES=1 nmtpy translate -S src:/data/jive/simmt/data/multi30k/en-de/${in_file}.lc.norm.tok.de -s mscoco -o ${mode} ${ckpt}
cp ${mode}.mscoco.beam6 result/

cd result
python ../compare_unks.py xe.mscoco.beam6 sac.mscoco.beam6 /data/jive/simmt/data/multi30k/en-de/${in_file}.lc.norm.tok.en > nounks.test
#python compare_unks.py xe.test sac.test ${data_dir}/iwslt14-test.tgt.txt > nounks.test
cut -f1 -d$'\t' nounks.test > xe-unk.test
cut -f2 -d$'\t' nounks.test > sac-unk.test
cut -f3 -d$'\t' nounks.test > ref-unk.test

cd $multeval_dir

./multeval.sh eval --refs /data/jive/simmt/data/multi30k/en-de/${in_file}.lc.norm.tok.en --hyps-baseline ${home_dir}/result/xe.mscoco.beam6 --hyps-sys1 ${home_dir}/result/sac.mscoco.beam6  --meteor.language en --latex table.tex --rankDir rank
mv table.tex ${home_dir}/result/table-big.tex
./multeval.sh eval --refs ${home_dir}/result/ref-unk.test --hyps-baseline ${home_dir}/result/xe-unk.test --hyps-sys1 ${home_dir}/result/sac-unk.test --meteor.language en --latex table.tex --rankDir rank
mv table.tex ${home_dir}/result/table.tex
#./multeval.sh eval --refs /data/jive/simmt/wmt16_en_de/newstest2009.low.en --hyps-baseline ${home_dir}/result/xe.mscoco.beam6 --hyps-sys1 ${home_dir}/result/sac.mscoco.beam6  --meteor.language en --latex table.tex --rankDir rank
#mv table.tex ${home_dir}/result/table.tex
