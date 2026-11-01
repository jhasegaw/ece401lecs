target = lec31
figs = exp/dampedimpulse49.png exp/resonatorpoles.png exp/resonatorimpulse.gif exp/resonatorfreq.gif exp/resonatorconv.gif exp/dampedpoles.png exp/dampedimpulse.gif exp/dampedfreq.gif exp/speechwave.png exp/speechwaves.png exp/speechspecs.png
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



