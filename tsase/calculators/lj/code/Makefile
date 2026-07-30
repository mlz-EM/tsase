MOD = lj_

OUT = $(MOD).so
../$(OUT):
	f2py -c --opt=-O2 --fcompiler=gnu95 *.f -m $(MOD)
	mv -f $(MOD)*.so ..
#	mv -f $(OUT) ..

.PHONY: clean
clean:
	rm -f *.o ../$(MOD)*.so
#	rm -f *.o ../$(OUT)

