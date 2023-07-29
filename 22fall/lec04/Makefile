target = lec04
byproducts = *~ exp/* *.{aux,bbl,blg,dvi,log,out,toc,nav,snm,synctex.gz} 

all:	figs pdf

pdf:
	pdflatex $(target)
	pdflatex $(target)

figs:
	mkdir -p exp
	python slides05.py

clean:
	-rm $(byproducts)
	-rmdir exp

superclean:
	-rm $(byproducts) $(target).pdf
	-rmdir exp


