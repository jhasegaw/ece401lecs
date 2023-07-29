target = lec03
byproducts = *~ exp/* *.{aux,bbl,blg,dvi,log,out,toc,nav,snm,synctex.gz} 

all:	figs pdf

pdf:
	pdflatex $(target)
	pdflatex $(target)

figs:
	mkdir -p exp
	convert -coalesce squarewave_animation.gif exp/squarewave_animation.png
	convert -coalesce sawtooth_anim1.gif exp/sawtooth_anim1.png
	convert -coalesce sawtooth_anim2.gif exp/sawtooth_anim2.png
	convert -coalesce Fourier_Convergence.gif exp/Fourier_Convergence.png
	convert -coalesce WaveInterference.gif exp/WaveInterference.png
	python slides04.py

clean:
	-rm $(byproducts)
	-rmdir exp

superclean:
	-rm $(byproducts) $(target).pdf
	-rmdir exp

