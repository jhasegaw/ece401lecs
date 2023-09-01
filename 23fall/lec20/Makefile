target = lec20
byproducts = *~ exp/* *.{aux,bbl,blg,dvi,log,out,toc,nav,snm,synctex.gz} 

$(target).pdf:	$(target).tex exp/hamming3piece.png
	pdflatex $(target)
	pdflatex $(target)

exp/hamming3piece.png: makefigs.py
	mkdir -p exp
	python makefigs.py

clean:
	-rm $(byproducts)
	-rmdir exp

superclean:
	-rm $(byproducts) $(target).pdf
	-rmdir exp



