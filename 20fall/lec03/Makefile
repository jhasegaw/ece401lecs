lec03.pdf: lec03.tex exp/Unfasor-0.png exp/phaseshift.png exp/Sumafasores-0.png exp/WaveInterference-0.png
	pdflatex lec03

exp/Unfasor-0.png: Unfasor.gif
	mkdir -p exp
	convert -coalesce Unfasor.gif exp/Unfasor.png

exp/Sumafasores-0.png: Sumafasores.gif
	mkdir -p exp
	convert -coalesce Sumafasores.gif exp/Sumafasores.png

exp/WaveInterference-0.png: WaveInterference.gif
	mkdir -p exp
	convert -coalesce WaveInterference.gif exp/WaveInterference.png

exp/phaseshift.png: slides03.py
	python slides03.py
