target = lec18
byproducts = *~ exp/* *.{aux,bbl,blg,dvi,log,out,toc,nav,snm,synctex.gz} 

movie1 = https://upload.wikimedia.org/wikipedia/commons/b/b4/Dirac_function_approximation.gif
mov1 = Dirac_function_approximation
figs = exp/$(mov1)-0.png

all:	$(target).pdf

$(target).pdf:	$(target).tex two_pulses_with_impulse_response.png exp/Dirac_function_approximation-0.png exp/Dirac_distribution.png
	pdflatex $(target)
	pdflatex $(target)

exp/Dirac_distribution.png:
	python makefigs.py

Convolution_of_two_pulses_with_impulse_response.svg.png:
	wget https://thumb.wikimedia.org/wikipedia/commons/thumb/0/0e/Convolution_of_two_pulses_with_impulse_response.svg/Convolution_of_two_pulses_with_impulse_response.svg.png

two_pulses_with_impulse_response.png:
	mv Convolution_of_two_pulses_with_impulse_response.svg.png two_pulses_with_impulse_response.png

Dirac_function_approximation.gif:
	wget https://upload.wikimedia.org/wikipedia/commons/b/b4/Dirac_function_approximation.gif

exp/Dirac_function_approximation-0.png:
	convert -coalesce Dirac_function_approximation.gif exp/$(mov1)-%d.png

clean:
	-rm $(byproducts)
	-rmdir exp

superclean:
	-rm $(byproducts) $(target).pdf
	-rmdir exp



