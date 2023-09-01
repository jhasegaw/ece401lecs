target = lec08
byproducts = *~ exp/* *.{aux,bbl,blg,dvi,log,out,toc,nav,snm,synctex.gz} 

$(target).pdf: $(target).tex exp/dt_quadrature_undersampled.png
	pdflatex $(target)
	pdflatex $(target)

exp/dt_quadrature_undersampled.png: makefigs.py
	mkdir -p exp
	python makefigs.py

clean:
	-rm $(byproducts)
	-rmdir exp

superclean:
	-rm $(byproducts) $(target).pdf
	-rmdir exp
