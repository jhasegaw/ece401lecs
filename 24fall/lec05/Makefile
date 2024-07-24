target = lec05
byproducts = *~ exp/* *.{aux,bbl,blg,dvi,log,out,toc,nav,snm,synctex.gz} 

$(target).pdf: exp/orthogonality_for_spectrum.png exp/squarewave.png exp/Sine_one_period.png
	pdflatex $(target)
	pdflatex $(target)

exp/orthogonality_for_spectrum.png:
	mkdir -p exp
	python make_figs.py

exp/squarewave.png:
	mkdir -p exp
	wget -O exp/squarewave.png https://upload.wikimedia.org/wikipedia/commons/thumb/2/2c/Fourier_Series.svg/432px-Fourier_Series.svg.png

exp/Sine_one_period.png:
	mkdir -p exp
	wget -O exp/Sine_one_period.png https://upload.wikimedia.org/wikipedia/commons/thumb/d/d2/Sine_one_period.svg/1024px-Sine_one_period.svg.png

clean:
	-rm $(byproducts)
	-rmdir exp

superclean:
	-rm $(byproducts) $(target).pdf
	-rmdir exp


