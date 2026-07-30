try:
    from tsase.calculators.applyF import applyF
except:
    pass
try:
    from tsase.calculators.lepspho import lepspho
except:
    pass
try:
       from tsase.calculators.al import al
except:
    pass
try:
    from tsase.calculators.morse import morse
    def pt():
        return morse()
except:
    pass
try:
    from tsase.calculators.lj import lj
except:
    pass
try:
    from tsase.calculators.ljocl import ljocl
except:
    pass
try:
    import tsase.calculators.lammps_ext
except:
    pass
try:
    from tsase.calculators.lisi import lisi
except:
    pass
try:
    from tsase.calculators.mo import mo
except:
    pass
try:
    from tsase.calculators.si import si
except:
    pass
try:
    from tsase.calculators.w import w
except:
    pass
try:
    from tsase.calculators.bopfox import bopfox
except:
    pass
try:
	from tsase.calculators.voter97 import voter97
except:
	pass
try:
	from tsase.calculators.ZDP_5Gauss import ZDP_5Gauss
except:
	pass

try:
    from tsase.calculators.socorro import Socorro
except:
    pass

try:
    from tsase.calculators.gauss3 import gauss3
except:
    pass

try:
    from tsase.calculators.push import push
except:
    pass

try:
    from tsase.calculators.hyperplane import hyperplane_potential
except:
    pass

