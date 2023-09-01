target = lec11
byproducts = *~ exp/* *.{aux,bbl,blg,dvi,log,out,toc,nav,snm,synctex.gz} 

$(target).pdf:	$(target).tex exp/convolutionproof.png
	pdflatex $(target)
	pdflatex $(target)

exp/convolutionproof.png: makefigs.py
	mkdir -p exp
	python makefigs.py

clean:
	-rm $(byproducts)
	-rmdir exp

superclean:
	-rm $(byproducts) $(target).pdf
	-rmdir exp



