target = lec10

$(target).pdf: $(target).tex exp/differencer_highfreq0.png
	pdflatex $(target)
	pdflatex $(target)

exp/differencer_highfreq0.png:
	mkdir -p exp
	python makefigs.py

clean:
	-rm *~ exp/* *.{aux,bbl,blg,dvi,log,out,toc,nav,snm,synctex.gz} 
	-rmdir exp




