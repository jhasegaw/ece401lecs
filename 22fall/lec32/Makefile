target = lec32

byproducts = *~ exp/* *.{aux,bbl,blg,dvi,log,out,toc,nav,snm,synctex.gz} 

image1 = https://upload.wikimedia.org/wikipedia/commons/4/47/Little_girl_on_swing.jpg
img1 = Little_girl_on_swing.jpg
image2 = https://upload.wikimedia.org/wikipedia/commons/0/07/Resonance.PNG
img2 = Resonance.png

movie1 = https://upload.wikimedia.org/wikipedia/commons/b/bd/Resonancia_en_sistema_masa_resorte.gif
mov1 = Resonancia
movie2 = https://upload.wikimedia.org/wikipedia/commons/5/54/Pipe001.gif
mov2 = Pipe001
movie3 = https://upload.wikimedia.org/wikipedia/commons/2/2f/Molecule1.gif
mov3 = Molecule1
movie4 = https://upload.wikimedia.org/wikipedia/commons/a/a8/Pipe002.gif
mov4 = Pipe002
movie5 = https://upload.wikimedia.org/wikipedia/commons/7/77/Molecule2.gif
mov5 = Molecule2
movie6 = https://upload.wikimedia.org/wikipedia/commons/a/a9/Pipe003.gif
mov6 = Pipe003
movie7 = https://upload.wikimedia.org/wikipedia/commons/0/04/Molecule3.gif
mov7 = Molecule3

all:	$(target).pdf

$(target).pdf:	$(target).tex exp/damping-99.png exp/$(img1) exp/$(img2) exp/$(mov7)-0.png
	pdflatex $(target)
	pdflatex $(target)

exp/damping-99.png:
	mkdir -p exp
	python makefigs.py

exp/$(img1):
	mkdir -p exp
	wget -O exp/$(img1) $(image1)

exp/$(img2):
	mkdir -p exp
	wget -O exp/$(img2) $(image2)

exp/$(mov7)-0.png:
	test -s exp/$(mov1).gif || wget -O exp/$(mov1).gif $(movie1)
	test -s exp/$(mov1)-0.png || convert -coalesce exp/$(mov1).gif exp/$(mov1)-%d.png
	test -s exp/$(mov2).gif || wget -O exp/$(mov2).gif $(movie2)
	test -s exp/$(mov2)-0.png || convert -coalesce exp/$(mov2).gif exp/$(mov2)-%d.png
	test -s exp/$(mov3).gif || wget -O exp/$(mov3).gif $(movie3)
	test -s exp/$(mov3)-0.png || convert -coalesce exp/$(mov3).gif exp/$(mov3)-%d.png
	test -s exp/$(mov4).gif || wget -O exp/$(mov4).gif $(movie4)
	test -s exp/$(mov4)-0.png || convert -coalesce exp/$(mov4).gif exp/$(mov4)-%d.png
	test -s exp/$(mov5).gif || wget -O exp/$(mov5).gif $(movie5)
	test -s exp/$(mov5)-0.png || convert -coalesce exp/$(mov5).gif exp/$(mov5)-%d.png
	test -s exp/$(mov6).gif || wget -O exp/$(mov6).gif $(movie6)
	test -s exp/$(mov6)-0.png || convert -coalesce exp/$(mov6).gif exp/$(mov6)-%d.png
	test -s exp/$(mov7).gif || wget -O exp/$(mov7).gif $(movie7)
	test -s exp/$(mov7)-0.png || convert -coalesce exp/$(mov7).gif exp/$(mov7)-%d.png

clean:
	-rm $(byproducts) $(target).pdf
	-rmdir exp



