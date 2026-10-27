target = lec27
byproducts = *~ exp/* *.{aux,bbl,blg,dvi,log,out,toc,nav,snm,synctex.gz} 

$(target).pdf: Geometrique_exponentielle.png SG_RLS_LMS_chan_var.png Adaptive_Filter_Compact.png
	pdflatex $(target)
	pdflatex $(target)

clean:
	-rm $(byproducts)

superclean:
	-rm $(byproducts) $(target).pdf

Geometrique_exponentielle.png:
	wget https://upload.wikimedia.org/wikipedia/commons/b/b3/Geometrique_exponentielle.png

SG_RLS_LMS_chan_var.png:
	wget https://upload.wikimedia.org/wikipedia/commons/0/0e/SG_RLS_LMS_chan_var.png

Adaptive_Filter_Compact.png:
	wget https://thumb.wikimedia.org/wikipedia/commons/thumb/f/f3/Adaptive_Filter_Compact.png/960px-Adaptive_Filter_Compact.png
	mv 960px-Adaptive_Filter_Compact.png Adaptive_Filter_Compact.png
