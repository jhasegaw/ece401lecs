target = lec26
byproducts = *~ exp/* *.{aux,bbl,blg,dvi,log,out,toc,nav,snm,synctex.gz} 

image1 = https://upload.wikimedia.org/wikipedia/commons/thumb/4/48/Dirac_distribution_PDF.svg/640px-Dirac_distribution_PDF.svg.png
img1 = Dirac_distribution.png
image2 =https://upload.wikimedia.org/wikipedia/commons/thumb/0/0e/Convolution_of_two_pulses_with_impulse_response.svg/640px-Convolution_of_two_pulses_with_impulse_response.svg.png
img2 = two_pulses_with_impulse_response.png

movie1 = https://upload.wikimedia.org/wikipedia/commons/b/b4/Dirac_function_approximation.gif
mov1 = Dirac_function_approximation
figs = exp/$(mov1)-0.png

all:	$(target).pdf

$(target).pdf:	$(target).tex $(figs)
	pdflatex $(target)
	pdflatex $(target)

$(figs):
	python makefigs.py
	test -s exp/$(img1) || wget -O exp/$(img1) $(image1)
	test -s exp/$(img2) || wget -O exp/$(img2) $(image2)

	test -s exp/$(mov1).gif || wget -O exp/$(mov1).gif exp $(movie1)
	test -s exp/$(mov1)-0.png || convert -coalesce exp/$(mov1).gif exp/$(mov1)-%d.png

clean:
	-rm $(byproducts)
	-rmdir exp

superclean:
	-rm $(byproducts) $(target).pdf
	-rmdir exp



