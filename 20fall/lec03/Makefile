target = lec03
byproducts = *~ exp/* *.{aux,bbl,blg,dvi,log,out,toc,nav,snm,synctex.gz} 

all:	figs pdf

pdf:
	pdflatex $(target)
	pdflatex $(target)

figs:
	mkdir -p exp
	convert -coalesce Unfasor.gif exp/Unfasor-%d.png
	python slides03.py
	convert -coalesce WaveInterference.gif exp/WaveInterference-%d.png
	convert -coalesce Sumafasores.gif exp/Sumafasores-%d.png

clean:
	-rm $(byproducts)
	-rmdir exp

superclean:
	-rm $(byproducts) $(target).pdf
	-rmdir exp
