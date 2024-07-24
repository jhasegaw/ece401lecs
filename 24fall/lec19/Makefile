target = lec19
byproducts = *~ exp/* *.{aux,bbl,blg,dvi,log,out,toc,nav,snm,synctex.gz} 

$(target).pdf:	$(target).tex exp/centeredaveragingfilter.png
	pdflatex $(target)
	pdflatex $(target)

exp/centeredaveragingfilter.png: makefigs.py
	mkdir -p exp
	python makefigs.py

clean:
	-rm $(byproducts)
	-rmdir exp

superclean:
	-rm $(byproducts) $(target).pdf
	-rmdir exp



