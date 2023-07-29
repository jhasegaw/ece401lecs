target = lec14
figs = exp

byproducts = *~ exp/* *.{aux,bbl,blg,dvi,log,out,toc,nav,snm,synctex.gz} 

# Images
image1 = https://upload.wikimedia.org/wikipedia/commons/thumb/a/a4/Illustration_of_causal_and_non-causal_filters.svg/664px-Illustration_of_causal_and_non-causal_filters.svg.png
img1 = 664px-Illustration_of_causal_and_non-causal_filters.svg.png

all: $(target).pdf

$(target).pdf: $(target).tex $(figs)
	pdflatex $(target)
	pdflatex $(target)

$(figs):
	mkdir -p exp
	test -s exp/$(img1) || wget -Oexp/$(img1) $(image1)
	test -s exp/stepfunction50.png || python makefigs.py

clean:
	-rm $(byproducts)
	-rmdir exp

superclean:
	-rm $(byproducts) $(target).pdf
	-rmdir exp



