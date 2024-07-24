target = lec17
byproducts = *~ exp/* *.{aux,bbl,blg,dvi,log,out,toc,nav,snm,synctex.gz} 

$(target).pdf:	$(target).tex exp/centeredaveragingfilter.png exp/squarewave.png exp/squarewave_alone.png
	pdflatex $(target)
	pdflatex $(target)

exp/centeredaveragingfilter.png: makefigs.py
	mkdir -p exp
	python makefigs.py

exp/squarewave.png:
	mkdir -p exp
	wget -O exp/squarewave.png https://upload.wikimedia.org/wikipedia/commons/thumb/2/2c/Fourier_Series.svg/432px-Fourier_Series.svg.png


exp/squarewave_alone.png:
	mkdir -p exp
	python makefigs.py

clean:
	-rm $(byproducts)
	-rmdir exp

superclean:
	-rm $(byproducts) $(target).pdf
	-rmdir exp



