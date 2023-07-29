target = lec24
figs = exp/numsum.png exp/densum.png
byproducts = *~ exp/* *.{aux,bbl,blg,dvi,log,out,toc,nav,snm,synctex.gz} 

all:	$(target).pdf

$(target).pdf:	$(target).tex $(figs)
	pdflatex $(target)
	pdflatex $(target)

$(figs):
	mkdir -p exp
	python makefigs.py

clean:
	-rm $(byproducts) $(target).pdf
	-rmdir exp



