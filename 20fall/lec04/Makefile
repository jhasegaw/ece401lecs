lec04.pdf: lec04.tex exp/squarewave_animation-0.png exp/sawtooth_anim1-0.png exp/sawtooth_anim2-0.png exp/Fourier_Convergence-0.png exp/triplebeat.png exp/WaveInterference-0.png
	pdflatex lec04

exp/squarewave_animation-0.png: squarewave_animation.gif
	mkdir -p exp
	convert -coalesce squarewave_animation.gif exp/squarewave_animation.png

exp/sawtooth_anim1-0.png: sawtooth_anim1.gif
	mkdir -p exp
	convert -coalesce sawtooth_anim1.gif exp/sawtooth_anim1.png

exp/sawtooth_anim2-0.png: sawtooth_anim2.gif
	mkdir -p exp
	convert -coalesce sawtooth_anim2.gif exp/sawtooth_anim2.png

exp/Fourier_Convergence-0.png: Fourier_Convergence.gif
	mkdir -p exp
	convert -coalesce Fourier_Convergence.gif exp/Fourier_Convergence.png

exp/WaveInterference-0.png: ../lec03/WaveInterference.gif
	mkdir -p exp
	convert -coalesce ../lec03/WaveInterference.gif exp/WaveInterference.png

exp/triplebeat.png: slides04.py
	python slides04.py

