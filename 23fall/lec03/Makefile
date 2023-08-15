target = lec03
byproducts = *~ exp/* *.{aux,bbl,blg,dvi,log,out,toc,nav,snm,synctex.gz} 

$(target).pdf: exp/squarewave_animation-0.png exp/sawtooth_anim1-0.png exp/sawtooth_anim2-0.png exp/Fourier_Convergence-0.png exp/WaveInterference-0.png exp/triplebeat.png exp/squarewave.png
	pdflatex $(target)
	pdflatex $(target)

exp/squarewave.png:
	mkdir -p exp
	wget -O exp/squarewave.png https://upload.wikimedia.org/wikipedia/commons/thumb/2/2c/Fourier_Series.svg/432px-Fourier_Series.svg.png

exp/squarewave_animation-0.png:
	mkdir -p exp
	wget -O exp/squarewave_animation.gif https://upload.wikimedia.org/wikipedia/commons/1/1a/Fourier_series_square_wave_circles_animation.gif
	convert -coalesce exp/squarewave_animation.gif exp/squarewave_animation-%d.png

exp/sawtooth_anim1-0.png:
	mkdir -p exp
	wget -O exp/sawtooth_anim1.gif https://upload.wikimedia.org/wikipedia/commons/e/e8/Periodic_identity_function.gif
	convert -coalesce exp/sawtooth_anim1.gif exp/sawtooth_anim1-%d.png

exp/sawtooth_anim2-0.png:
	mkdir -p exp
	wget -O exp/sawtooth_anim2.gif https://upload.wikimedia.org/wikipedia/commons/7/7e/Fourier_series_sawtooth_wave_circles_animation.gif
	convert -coalesce exp/sawtooth_anim2.gif exp/sawtooth_anim2-%d.png

exp/Fourier_Convergence-0.png:
	mkdir -p exp
	wget -O exp/Fourier_Convergence.gif https://upload.wikimedia.org/wikipedia/commons/3/37/Example_of_Fourier_Convergence.gif
	convert -coalesce exp/Fourier_Convergence.gif exp/Fourier_Convergence-%d.png

exp/WaveInterference-0.png:
	mkdir -p exp
	wget -O exp/WaveInterference.gif https://upload.wikimedia.org/wikipedia/commons/5/5d/Waventerference.gif
	convert -coalesce exp/WaveInterference.gif exp/WaveInterference-%d.png

exp/triplebeat.png:
	mkdir -p exp
	python make_figs.py

clean:
	-rm $(byproducts)
	-rmdir exp

superclean:
	-rm $(byproducts) $(target).pdf
	-rmdir exp

