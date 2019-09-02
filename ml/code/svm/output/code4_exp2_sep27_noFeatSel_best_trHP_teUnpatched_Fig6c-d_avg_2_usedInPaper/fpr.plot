reset
set terminal pdfcairo monochrome enhanced dashed font "Times-Roman,20" linewidth 2
set output "fpr_expcd_all_zday.pdf"
set key samplen 2 top center horizontal spacing .75 width -1 font ",18"
set xlabel "number of attack classes" offset 0,0.5
set ylabel "%" offset 1,0
unset grid
set xtics 1
set ytics 5
#set yrange [-3:25]
set yrange [-3:5]
set style data lines
plot "fpr" u 1:2 title "Bi-Di" w lines, "fpr" u 1:3 title "N-Gram" w lines, "fpr" u 1:4 title "ens-SVM" w lines
set terminal windows
set output
unset terminal
replot
replot
