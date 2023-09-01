lec = lec06
byproducts = *~ exp/* *.{aux,bbl,blg,dvi,log,out,toc,nav,snm,synctex.gz} 

$(lec).pdf: $(lec).tex exp/dt_sine44.png
	pdflatex $(lec)
	pdflatex $(lec)

exp/dt_sine44.png:
	mkdir -p exp
	python makefigs.py

clean:
	-rm $(byproducts)
	-rmdir exp

superclean:
	-rm $(byproducts) $(target).pdf
	-rmdir exp
