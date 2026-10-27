target = lec27
byproducts = *~ exp/* *.{aux,bbl,blg,dvi,log,out,toc,nav,snm,synctex.gz} 

$(target).pdf:
	pdflatex $(target)
	pdflatex $(target)

clean:
	-rm $(byproducts)

superclean:
	-rm $(byproducts) $(target).pdf

