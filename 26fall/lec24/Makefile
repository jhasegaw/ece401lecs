target = lec24
byproducts = *~ exp/* *.{aux,bbl,blg,dvi,log,out,toc,nav,snm,synctex.gz} 

all:	$(target).pdf Acf_new.svg.png

$(target).pdf:	$(target).tex Acf_new.svg.png Comparison_convolution_correlation.svg.png Pink_noise_spectrum.svg.png exp/speech_and_autocorrelation.png
	pdflatex $(target)
	pdflatex $(target)

clean:
	-rm $(byproducts)
	-rmdir exp

superclean:
	-rm $(byproducts) $(target).pdf
	-rmdir exp

Acf_new.svg.png:
	wget https://thumb.wikimedia.org/wikipedia/commons/thumb/e/ed/Acf_new.svg/500px-Acf_new.svg.png
	mv 500px-Acf_new.svg.png Acf_new.svg.png

exp/speech_and_autocorrelation.png:
	python makefigs.py
