MOD = morse_

OUT = $(MOD).so
../$(OUT):
	f2py -c --opt=-O2 --fcompiler=gnu95 *.f -m $(MOD)
	mv -f *.so ..
#	mv -f $(OUT) ..

.PHONY: clean
clean:
	rm -f *.o ../*.so
#	rm -f *.o ../$(OUT)

