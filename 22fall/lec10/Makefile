target = lec10
byproducts = *~ exp/* *.{aux,bbl,blg,dvi,log,out,toc,nav,snm,synctex.gz} 

all:	figs pdf

pdf:
	pdflatex $(target)
	pdflatex $(target)

figs:
	mkdir -p exp
	convert -coalesce convolution.gif exp/convolution.png

clean:
	-rm $(byproducts)
	-rmdir exp

superclean:
	-rm $(byproducts) $(target).pdf
	-rmdir exp



