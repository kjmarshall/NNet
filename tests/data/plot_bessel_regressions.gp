b0_data="bessel_0.data"
b0_noisy="bessel_0_noisy.data"
b0_pred="bessel_0.prediction"
b1_data="bessel_1.data"
b1_noisy="bessel_1_noisy.data"
b1_pred="bessel_1.prediction"
b2_data="bessel_2.data"
b2_noisy="bessel_2_noisy.data"
b2_pred="bessel_2.prediction"
b3_data="bessel_3.data"
b3_noisy="bessel_3_noisy.data"
b3_pred="bessel_3.prediction"

set terminal pdf enhanced color size 6in,6in font 'Verdana,16'
# set terminal png

set xrange [0:20]
set yrange [-1:1]

set xlabel "x"
set ylabel "y"
set style fill transparent solid 0.5 noborder

set title "Bessel Function Regression w/\n Feed Forward Neural Network \n \eta = 0.01, epochs = 1000, batchSize = 10"
set output "b0.pdf"
plot b0_noisy u 1:2 w p pt 6 lc rgb "#6749ff" t "b0n", \
	 b0_pred u 1:2 w p pt 7 ps .5 lc rgb "#ffb049" t "b0p", \
	 b0_data u 1:2 w l lw 2 lc rgb "red" t "b0a"

set output "b1.pdf"
plot b1_noisy u 1:2 w p pt 6 lc rgb "#6749ff" t "b1n", \
	 b1_pred u 1:2 w p pt 7 ps .5 lc rgb "#ffb049" t "b1p", \
	 b1_data u 1:2 w l lw 2 lc rgb "red" t "b1a"

set output "b2.pdf"
plot b2_noisy u 1:2 w p pt 6 lc rgb "#6749ff" t "b2n", \
	 b2_pred u 1:2 w p pt 7 ps .5 lc rgb "#ffb049" t "b2p", \
	 b2_data u 1:2 w l lw 2 lc rgb "red" t "b2a"

set output "b3.pdf"
plot b3_noisy u 1:2 w p pt 6 lc rgb "#6749ff" t "b3n" , \
	 b3_pred u 1:2 w p pt 7 ps .5 lc rgb "#ffb049" t "b3p", \
	 b3_data u 1:2 w l lw 2 lc rgb "red" t "b3a"

