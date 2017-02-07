for f in *.mpg
do
 ffmpeg -i $f -ab 1k -ac 1 -ar 44100 -vn ${f/.mpg/.wav}
done