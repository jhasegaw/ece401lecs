target = lec23
byproducts = *~ exp/* *.{aux,bbl,blg,dvi,log,out,toc,nav,snm,synctex.gz} 

all:	$(target).pdf

$(target).pdf:	$(target).tex Acf_new.svg.png Comparison_convolution_correlation.svg.png Pink_noise_spectrum.svg.png
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

Comparison_convolution_correlation.svg.png:
	wget https://thumb.wikimedia.org/wikipedia/commons/thumb/2/21/Comparison_convolution_correlation.svg/960px-Comparison_convolution_correlation.svg.png
	mv 960px-Comparison_convolution_correlation.svg.png Comparison_convolution_correlation.svg.png

Pink_noise_spectrum.svg.png:
	wget https://thumb.wikimedia.org/wikipedia/commons/thumb/7/77/Pink_noise_spectrum.svg/960px-Pink_noise_spectrum.svg.png
	mv 960px-Pink_noise_spectrum.svg.png Pink_noise_spectrum.svg.png

Anscombe_quartet_3.svg.png:
	wget https://thumb.wikimedia.org/wikipedia/commons/thumb/e/ec/Anscombe%27s_quartet_3.svg/960px-Anscombe%27s_quartet_3.svg.png
	mv "960px-Anscombe's_quartet_3.svg.png" Anscombe_quartet_3.svg.png
