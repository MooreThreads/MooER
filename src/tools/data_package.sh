#!/bin/bash
set -x  # for better debug view
export PATH=$PWD:$PATH
export PATH=$PWD/../:$PATH
export PYTHONIOENCODING=UTF-8
export PYTHONPATH=../:$PYTHONPATH
export LC_ALL=C

THIS_DIR="$( cd "$( dirname "$0" )" && pwd )"

echo "##### config #####"
wav_scp='data/wav.scp'
text='data/text'
write_dir='data/pkg/path'
write_num=1
write_prefix='data.#.list'
text_norm=true
shuffle=true
data_type=raw
num_threads=32
# raw or shard

# parse config
. ${THIS_DIR}/parse_options.sh || exit 1;

mkdir -p $write_dir

process_root=${THIS_DIR}/process_tmp
mkdir -p $process_root
cp $text ${process_root}/text.org
cp $wav_scp ${process_root}/wav.scp.org
# do text normlization
if [ $text_norm = true ]; then
  echo "do text normlization"
  paste -d " " <(cut -f 1 -d" " ${process_root}/text.org) \
      <(cut -f 2- -d" " ${process_root}/text.org \
      | tr 'a-z' 'A-Z' | sed 's/\([A-Z]\) \([A-Z]\)/\1▁\2/g' \
      | sed 's/\([A-Z]\) \([A-Z]\)/\1▁\2/g' | tr -d " ") \
      > ${process_root}/text.process
  sed -i 's/\xEF\xBB\xBF//' ${process_root}/text.process
else
  cp ${process_root}/text.org ${process_root}/text.process
fi

if [ $data_type = shard ]; then
  python3 ${THIS_DIR}/make_shard_list.py --resample 16000 --num_utts_per_shard 100000 \
    --num_threads $num_threads ${process_root}/wav.scp.org ${process_root}/text.process $write_dir \
    ${process_root}/data.list
else
  echo "data_type only support shard, but got $data_type" && exit 1
fi

# shuffle
if [ $shuffle = true ]; then
  shuf ${process_root}/data.list -o ${process_root}/data.list.shuffle
else
  cp ${process_root}/data.list ${process_root}/data.list.shuffle
fi

# split and rename
lines_num=`cat ${process_root}/data.list.shuffle | wc -l`
lines_each=`echo $((lines_num / write_num)) | bc -l`
echo "All samples: $lines_num ; Write for $write_num file ; Each has samples: $lines_each"
mkdir ${process_root}/split
split -l $lines_each -d ${process_root}/data.list.shuffle ${process_root}/split/$write_prefix
i=0
for path in `ls ${process_root}/split | grep $write_prefix`; do
  write_file=`echo $write_prefix | sed "s|#|${i}|g"`
  cp ${process_root}/split/$path ${write_dir}/$write_file
  i=$((i + 1))
done

# remove process dir
rm -r ${THIS_DIR}/process_tmp
