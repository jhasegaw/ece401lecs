target = lec20
figs = exp/dft_of_cosine.png
byproducts = *~ exp/* *.{aux,bbl,blg,dvi,log,out,toc,nav,snm,synctex.gz} 

all:	$(target).pdf

$(target).pdf:	$(target).tex $(figs)
	pdflatex $(target)
	pdflatex $(target)

$(figs):
	python makefigs.py

clean:
	-rm $(byproducts) $(target).pdf
	-rmdir exp



