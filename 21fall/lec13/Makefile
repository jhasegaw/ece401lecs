target = lec13
figs = exp/firstdiffonly_magnitude.png exp/firstdiff_magnitude.png exp/firstdiff_phase.png exp/firstdiff_tonesweep10.png exp/firstdiff_tonesweep75.png exp/firstdiff_tonesweep150.png exp/differenced_unitstep.png
byproducts = *~ exp/* *.{aux,bbl,blg,dvi,log,out,toc,nav,snm,synctex.gz} 

all:	$(target).pdf

$(target).pdf:	$(target).tex $(figs)
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



