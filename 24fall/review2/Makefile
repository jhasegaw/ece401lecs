target = review2
byproducts = *~ exp/* *.{aux,bbl,blg,dvi,log,out,toc,nav,snm,synctex.gz} 

all:	$(target).pdf

$(target).pdf:	$(target).tex 
	pdflatex $(target)
	pdflatex $(target)

clean:
	-rm $(byproducts)
	-rmdir exp

superclean:
	-rm $(byproducts) $(target).pdf
	-rmdir exp



