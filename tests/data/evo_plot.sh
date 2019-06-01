#!/bin/bash

SCRIPT="evo_plot.gp"
ARG1=$1
ARG2=$2
echo -e ${ARG1}
echo -e ${ARG2}

rename 's/\d+/sprintf("%03d", $&)/e' evo/*.data

for file in	$( find evo/*.data -type f ); do
	echo -e $file
	output_pdf=${file/evo/evo_plots}
	output_pdf=${output_pdf/data/pdf}
	output_png=${output_pdf/pdf/png}
	output_gif=${output_pdf/pdf/gif}
	echo -e $file $output_pdf $output_png $output_gif
	# gnuplot -c ${SCRIPT} ${file} ${output_pdf}
	convert -density 600 $output_pdf ${output_gif}
done
# cd evo_plots
# convert -layers OptimizePlus -delay 75 pred_[0123][0-9][0-9].png -loop 0 out.gif
# cd ..
