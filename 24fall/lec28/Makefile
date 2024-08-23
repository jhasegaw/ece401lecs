target = lec28
figs = exp/impulses.gif exp/magresponse.gif exp/toneresponse.gif
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



