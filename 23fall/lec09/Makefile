target = lec09
byproducts = *~ exp/* *.{aux,bbl,blg,dvi,log,out,toc,nav,snm,synctex.gz} 

$(target).pdf: $(target).tex exp/convolution-94.png exp/gradient.png
	pdflatex $(target)
	pdflatex $(target)

exp/convolution-94.png:
	mkdir -p exp
	wget -O exp/convolution.gif https://upload.wikimedia.org/wikipedia/commons/b/b9/Convolution_of_spiky_function_with_box2.gif
	convert -coalesce exp/convolution.gif exp/convolution.png

exp/gradient.png:
	mkdir -p exp
	wget -O exp/gradient.png https://upload.wikimedia.org/wikipedia/commons/thumb/0/0f/Gradient2.svg/640px-Gradient2.svg.png

clean:
	-rm $(byproducts)
	-rmdir exp

superclean:
	-rm $(byproducts) $(target).pdf
	-rmdir exp



