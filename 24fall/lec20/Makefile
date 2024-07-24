target = lec20
figs = 
byproducts = *~ exp/* *.{aux,bbl,blg,dvi,log,out,toc,nav,snm,synctex.gz} 

all:	$(target).pdf

$(target).pdf:	$(target).tex exp/ideal_lpf.png
	pdflatex $(target)
	pdflatex $(target)

exp/ideal_lpf.png: makefigs.py
	mkdir -p exp
	python makefigs.py

clean:
	-rm $(byproducts)
	-rmdir exp

superclean:
	-rm $(byproducts) $(target).pdf
	-rmdir exp



