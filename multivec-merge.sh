export merged="output/bg-cs-da-de-el-en-es-fi-fr-hu-it-sv.multiskip.iter_10+window_1+min_count_5+negative_5+size_40+threads_32"

rm $merged
for lang in bg cs da de el en es fi fr hu it sv; do python ~/wammar-utils/prefix_lines.py -i $merged.$lang -o $merged.$lang.prefixed -p $lang: && cat $merged.$lang.prefixed >> $merged; done
