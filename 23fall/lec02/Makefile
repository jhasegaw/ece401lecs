target = lec02
byproducts = *~ exp/* *.{aux,bbl,blg,dvi,log,out,toc,nav,snm,synctex.gz} 

$(target).pdf: exp/phaser.jpg exp/Complex_conjugate_picture.png exp/Euler.png exp/Oscillating_sine_wave.png exp/Vector_Addition.png exp/sine_and_cosine.png exp/sohcahtoa.png exp/phaser.jpg exp/Sumafasores-0.png exp/Unfasor-0.png exp/WaveInterference-0.png exp/startingpoints.png exp/startrek.jpg
	pdflatex $(target)
	pdflatex $(target)

exp/startingpoints.png:
	mkdir -p exp
	python make_figs.py

exp/Unfasor-0.png:
	mkdir -p exp
	wget -O exp/Unfasor.gif https://upload.wikimedia.org/wikipedia/commons/8/89/Unfasor.gif
	convert -coalesce exp/Unfasor.gif exp/Unfasor-%d.png

exp/WaveInterference-0.png:
	mkdir -p exp
	wget -O exp/WaveInterference.gif https://upload.wikimedia.org/wikipedia/commons/5/5d/Waventerference.gif
	convert -coalesce exp/WaveInterference.gif exp/WaveInterference-%d.png

exp/Sumafasores-0.png:
	mkdir -p exp
	wget -O exp/Sumafasores.gif https://upload.wikimedia.org/wikipedia/commons/9/92/Sumafasores.gif
	convert -coalesce exp/Sumafasores.gif exp/Sumafasores-%d.png

exp/Complex_conjugate_picture.png:
	mkdir -p exp
	wget -O exp/Complex_conjugate_picture.png https://upload.wikimedia.org/wikipedia/commons/2/2a/Complex_conjugate_picture.png

exp/Euler.png:
	mkdir -p exp
	wget -O exp/Euler.png https://upload.wikimedia.org/wikipedia/commons/thumb/3/37/Euler%27s_formula_caimi.svg/864px-Euler%27s_formula_caimi.svg.png

exp/Oscillating_sine_wave.png:
	mkdir -p exp
	wget -O exp/Oscillating_sine_wave.gif https://upload.wikimedia.org/wikipedia/commons/d/d8/Oscillating_sine_wave.gif
	convert exp/Oscillating_sine_wave.gif exp/Oscillating_sine_wave.png

exp/Vector_Addition.png:
	mkdir -p exp
	wget -O exp/Vector_Addition.png https://upload.wikimedia.org/wikipedia/commons/7/7c/Vector_Addition.png

exp/sine_and_cosine.png:
	mkdir -p exp
	wget -O exp/sine_and_cosine.png https://upload.wikimedia.org/wikipedia/commons/thumb/7/71/Sine_cosine_one_period.svg/1024px-Sine_cosine_one_period.svg.png

exp/sohcahtoa.png:
	mkdir -p exp
	wget -O exp/sohcahtoa.png https://upload.wikimedia.org/wikipedia/commons/thumb/e/e5/Trigonometric_function_triangle_mnemonic.svg/640px-Trigonometric_function_triangle_mnemonic.svg.png

exp/startrek.jpg:
	mkdir -p exp
	wget -O exp/startrek.jpg https://upload.wikimedia.org/wikipedia/commons/1/1c/William_Shatner_Sally_Kellerman_Star_Trek_1966.JPG

clean:
	-rm $(byproducts)
	-rmdir exp

superclean:
	-rm $(byproducts) $(target).pdf
	-rmdir exp
