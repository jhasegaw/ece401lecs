target = lec07b
byproducts = *~ exp/* *.{aux,bbl,blg,dvi,log,out,toc,nav,snm,synctex.gz} 

$(target).pdf: exp/squarewave_animation-0.png exp/sawtooth_anim1-0.png exp/Fourier_Convergence-0.png exp/squarewave.png exp/missing_fundamental.jpg
	pdflatex $(target)
	pdflatex $(target)

exp/squarewave_animation-0.png: exp/squarewave_animation.gif
	convert -coalesce exp/squarewave_animation.gif exp/squarewave_animation-%d.png

exp/squarewave_animation.gif:
	mkdir -p exp
	wget -O exp/squarewave_animation.gif https://upload.wikimedia.org/wikipedia/commons/1/1a/Fourier_series_square_wave_circles_animation.gif

exp/sawtooth_anim1.gif:
	mkdir -p exp
	wget -O exp/sawtooth_anim1.gif https://upload.wikimedia.org/wikipedia/commons/e/e8/Periodic_identity_function.gif

exp/sawtooth_anim1-0.png: exp/sawtooth_anim1.gif
	convert -coalesce exp/sawtooth_anim1.gif exp/sawtooth_anim1-%d.png

exp/Fourier_Convergence.gif:
	mkdir -p exp
	wget -O exp/Fourier_Convergence.gif https://upload.wikimedia.org/wikipedia/commons/3/37/Example_of_Fourier_Convergence.gif

exp/Fourier_Convergence-0.png: exp/Fourier_Convergence.gif
	convert -coalesce exp/Fourier_Convergence.gif exp/Fourier_Convergence-%d.png

exp/missing_fundamental.jpg:
	mkdir -p exp
	wget -O exp/missing_fundamental.jpg https://thumb.wikimedia.org/wikipedia/commons/thumb/5/5d/Illustration_of_common_periodicity_of_full_spectrum_and_missing_fundamental_waveforms.jpg/960px-Illustration_of_common_periodicity_of_full_spectrum_and_missing_fundamental_waveforms.jpg?utm_source=commons.wikimedia.org&utm_campaign=index&utm_content=thumbnail

clean:
	-rm $(byproducts)

superclean:
	-rm $(byproducts) $(target).pdf
	-rmdir exp

