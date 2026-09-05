target = lec08
byproducts = *~ exp/* *.{aux,bbl,blg,dvi,log,out,toc,nav,snm,synctex.gz} 

$(target).pdf: exp/orthogonality_for_spectrum.png
	pdflatex $(target)
	pdflatex $(target)

exp/orthogonality_for_spectrum.png:
	mkdir -p exp
	python make_figs.py

clean:
	-rm $(byproducts)
	-rmdir exp

superclean:
	-rm $(byproducts) $(target).pdf
	-rmdir exp


