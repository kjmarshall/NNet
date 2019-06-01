b_data="bessel_1.data"
b_noisy="bessel_1_noisy.data"
b_pred=ARG1
b_output=ARG2

set terminal pdf enhanced color size 6in,6in font 'Verdana,16'
# set terminal png

set xrange [0:20]
set yrange [-1:1]

set xlabel "x"
set ylabel "y"
set style fill transparent solid 0.5 noborder

set title "Bessel Function Regression w/\n Feed Forward Neural Network \n \eta = 0.01, epochs = 1000, batchSize = 10"
set output b_output
plot b_noisy u 1:2 w p pt 6 lc rgb "#6749ff" t "b0n", \
	 b_pred u 1:2 w p pt 7 ps .5 lc rgb "#ffb049" t "b0p", \
	 b_data u 1:2 w l lw 2 lc rgb "red" t "b0a"
