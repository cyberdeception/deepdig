reset
set terminal pdfcairo monochrome enhanced dashed font "Times-Roman,20" linewidth 2
set output "f2_exp2_all_zday.pdf"
set key samplen 2 top center horizontal spacing .75 width -1 font ",18"
set xlabel "number of attack classes" offset 0,0.5
set ylabel "%" offset 1,0
unset grid
set xtics 1
set ytics 20
set yrange [-10:119]
set style data lines
plot "f2" u 1:2 title "Bi-Di" w lines, "f2" u 1:3 title "N-Gram" w lines, "f2" u 1:4 title "ens-SVM" w lines
set terminal windows
set output
unset terminal
replot
replot
