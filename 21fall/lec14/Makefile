target = lec14
figs = exp/squarewave_real.png exp/puredelay.png exp/squarewave.png exp/delayedsquarewave.png exp/differenced_squarewave.png exp/delayed_differenced_squarewave.png exp/differenced_delayed_squarewave.png exp/localaveragefilters.png exp/dirichletform.png exp/centeredaveragingfilter.png


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



