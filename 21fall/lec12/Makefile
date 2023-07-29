target = lec12
figs = exp/convolutionproof.png
byproducts = *~ exp/* *.{aux,bbl,blg,dvi,log,out,toc,nav,snm,synctex.gz} 

all:	$(figs) pdf

pdf:	$(target).tex 
	pdflatex $(target)
	pdflatex $(target)

$(figs):
	python makefigs.py

clean:
	-rm $(byproducts)
	-rmdir exp

superclean:
	-rm $(byproducts) $(target).pdf
	-rmdir exp



