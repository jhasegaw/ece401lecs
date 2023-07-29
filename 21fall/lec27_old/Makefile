target = lec27
figs = exp/dampedfreq.gif exp/dampedconv.gif exp/speechwave.png exp/speech_fivepulses.png exp/speech_onepulse.png exp/inversefilter.png
byproducts = *~ exp/* *.{aux,bbl,blg,dvi,log,out,toc,nav,snm,synctex.gz} 

all:	$(target).pdf

$(target).pdf:	$(target).tex $(figs)
	pdflatex $(target)
	pdflatex $(target)

$(figs):
	mkdir -p exp
	python makefigs.py

clean:
	-rm $(byproducts) $(target).pdf
	-rmdir exp



