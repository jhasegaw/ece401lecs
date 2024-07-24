target = lec15
figs = exp

byproducts = *~ exp/* *.{aux,bbl,blg,dvi,log,out,toc,nav,snm,synctex.gz} 

$(target).pdf: $(target).tex exp/664px-Illustration_of_causal_and_non-causal_filters.svg.png exp/stepfunction50.png exp/unwrappedphase.jpg
	pdflatex $(target)
	pdflatex $(target)

exp/664px-Illustration_of_causal_and_non-causal_filters.svg.png:
	mkdir -p exp
	wget -O exp/664px-Illustration_of_causal_and_non-causal_filters.svg.png https://upload.wikimedia.org/wikipedia/commons/thumb/a/a4/Illustration_of_causal_and_non-causal_filters.svg/664px-Illustration_of_causal_and_non-causal_filters.svg.png

exp/stepfunction50.png: makefigs.py
	mkdir -p exp
	python makefigs.py

exp/unwrappedphase.jpg:
	mkdir -p exp
	wget -O exp/unwrappedphase.jpg https://upload.wikimedia.org/wikipedia/commons/5/5f/Phase_vs_Time%2C_wrapped_and_unwrapped.jpg

clean:
	-rm $(byproducts)
	-rmdir exp

superclean:
	-rm $(byproducts) $(target).pdf
	-rmdir exp



