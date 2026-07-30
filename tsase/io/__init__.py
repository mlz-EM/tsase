import ase.io
from ase.io.vasp import read_vasp_configuration
from .con import read_con, write_con
from .feff import read_feff, write_feff
from .bopfox import read_bopfox, write_bopfox
from .vasp import read_xdatcar
from .lammps import read_lammps, read_dump
from .socorro import read_socorro, write_socorro
from .aims import read_aims, write_aims
from .colors import read_colors

# THESE TRY-EXCEPTS ARE KILLING ME!!!!!!!! (sung)

def read_vasp_multiframe(filename, skip=0, every=1):
    """
    Reads a VASP trajectory file.
    
    First, it tries to read the file as an XDATCAR. If that fails or
    returns only one frame, it falls back to reading it as a 
    concatenated POSCAR file.
    """
    # --- Part 1: Attempt to read as an XDATCAR first ---
    try:
        xdat = read_xdatcar(filename, skip, every)
        # Using isinstance() is slightly more robust than type()
        if isinstance(xdat, list) and len(xdat) > 1:
            # Success! Return the list of frames from the XDATCAR.
            return xdat
    except Exception:
        # This is expected if the file is not a valid XDATCAR.
        # We 'pass' to proceed to the next reading method.
        pass

    # --- Part 2: Fallback to reading as a concatenated POSCAR ---
    frames = []
    try:
        with open(filename, 'r') as f:
            while True:
                try:
                    frame = read_vasp_configuration(f) # Or vasp.read_vasp_configuration(f)
                    frames.append(frame)
                except (EOFError, ValueError, IndexError, RuntimeError):
                    # This is the correct way to detect the end of the file.
                    break
    except FileNotFoundError:
        raise IOError(f"Error: The file {filename} was not found.")

    if not frames:
        raise IOError(f"Could not read any valid frames from {filename} in either XDATCAR or POSCAR format.")
        
    # Return a single Atoms object if only one frame, or a list if multiple.
    return frames[0] if len(frames) == 1 else frames
           

def read(filename, skip, every):

    try:
        return read_con(filename)
    except:
        pass
    try: 
        return read_bopfox(filename)
    except:
        pass
    try:
        return read_lammps(filename)
    except:
        pass
    try:
        return read_vasp_multiframe(filename, skip, every)
    except:
        pass
    try: 
        return read_dump(filename)
    except:
        pass
    try:
        return read_socorro(filename)
    except:
        pass
    try:
        return read_feff(filename)
    except:
        pass
    try:
        return ase.io.read(filename+"@:", format='xyz')
    except:
        pass
    try:
        return ase.io.read(filename+"@:", format='aims')
#        a = read_aims(filename)
#        if len(a.positions) < 1:
#            raise
#        return a
    except:
        pass
    try:
        return ase.io.read(filename+"@:")
    except:
        pass
    raise IOError("Could not read file %s." % filename)
