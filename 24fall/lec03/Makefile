target = lec03
byproducts = *~ exp/* *.{aux,bbl,blg,dvi,log,out,toc,nav,snm,synctex.gz} 

$(target).pdf: exp/WaveInterference-0.png exp/triplebeat.png
	pdflatex $(target)
	pdflatex $(target)

exp/WaveInterference-0.png:
	mkdir -p exp
	wget -O exp/WaveInterference.gif https://upload.wikimedia.org/wikipedia/commons/5/5d/Waventerference.gif
	convert -coalesce exp/WaveInterference.gif exp/WaveInterference-%d.png

exp/triplebeat.png:
	mkdir -p exp
	python make_figs.py

clean:
	-rm $(byproducts)
	-rmdir exp

superclean:
	-rm $(byproducts) $(target).pdf
	-rmdir exp

