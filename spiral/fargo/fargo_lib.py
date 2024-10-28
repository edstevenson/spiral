"""
Library for reading FARGO3D data from file.
"""

__all__ = [ "FARGO_Sim" ]

import os
import re
import numpy as np


class Domain(object):
    '''Simulation domain class'''
    def __init__(self, Sim='./'):
        tmp = np.genfromtxt(os.path.join(Sim, 'dimensions.dat'), names=True)

        self._size = np.array([tmp['NX'], tmp['NY'], tmp['NZ']], np.int32)

        self._xmin = np.array([tmp['XMIN'], tmp['YMIN'], tmp['ZMIN']])
        self._xmax = np.array([tmp['XMAX'], tmp['YMAX'], tmp['ZMAX']])
                              
        self._nghost = np.array([0, tmp['NGHY'], tmp['NGHZ']], np.int32)
        self._ncpu   = np.array([1, tmp['CPUS_Y'], tmp['CPUS_Z']], np.int32)

    @property
    def grid_size(self):
        return self._size

    @property
    def dimensions(self):
        return ''.join([ d for n,d in zip(self._size, 'xyz') if n > 1])

    @property
    def min(self):
        return self._xmin
    @property
    def max(self):
        return self._xmax

    @property
    def num_ghost_cells(self):
        return self._nghost

    @property
    def cpus_per_dimension(self):
        return self._ncpu


class Patch(object):
    '''Region of simulation domain on a specific processor'''
    def __init__(self, n, Sim='./'):
        tmp = np.genfromtxt(os.path.join(Sim, 'grid{:03d}.inf'.format(n)),
                            names=True)

        self._rank  = int(tmp['CPU_Rank'])
        self._index = tuple(map(int,[0, tmp['IndexY'], tmp['IndexZ']]))

        self._start = np.array([0,  tmp['Y0'],   tmp['Z0']  ], np.int32)
        self._end   = np.array([-1, tmp['YN']+1, tmp['ZN']+1], np.int32)

    @property
    def rank(self):
        return self._rank

    @property
    def index(self):
        return self._index

    @property
    def grid_range(self):
        return np.array([self._start, self._end])

    @property
    def shape(self):
        return self._end - self._start

def _if_merged(Sim, field):
    '''Checked whether a given snaphot of a field is merged'''
    for f in os.listdir(Sim):
        if f.startswith(field):
            tail = f.replace(field, '')
            if '_' in tail:
                return False
            else:
                return True
    else:
        raise AttributeError('Field {} not found'.format(field))

class Geometry(object):
    '''Base geometry class for handling FARGO-3d's geometrical aliasing'''
    def __init__(self, aliases={}, dim_aliases={}):
        self._aliases = dict(aliases)
        for a in dim_aliases:
            self._aliases[a] = dim_aliases[a]
            self._aliases[a+'med'] = dim_aliases[a] + 'med'
            
    def get_alias(self, alias):
        return self._aliases[alias]
    
    def aliases(self):
        '''Get the list of aliases'''
        return self._aliases.keys()

class Cylindrical(Geometry):
    _dim_aliases = { 'R'   : 'x2',
                     'phi' : 'x1',
                     'z'   : 'x3',
                     }
    def __init__(self):
        super(Cylindrical, self).__init__({}, self._dim_aliases)

class Cartesian(Geometry):
    _dim_aliases = { 'x' : 'x1',
                     'y' : 'x2',
                     'z' : 'x3',
                     }
    def __init__(self):
        super(Cartesian, self).__init__({}, self._dim_aliases)


class Spherical(Geometry):
    _dim_aliases = { 'r'     : 'x2',
                     'phi'   : 'x1',
                     'theta' : 'x3',
                     }
    def __init__(self):
        super(Spherical, self).__init__({}, self._dim_aliases)

def get_geometry(geom):
    if geom == "cartesian":
        return Cartesian()
    elif geom == 'cylindrical':
        return Cylindrical()
    elif geom == 'spherical':
        return Spherical()
    else:
        raise AttributeError("Error Geometry is not recognised")


class FARGO_Sim(object):

    def __init__(self, Sim):
        self._sim = Sim
        self._domain = Domain(Sim)

        self._setup_patches()
        self._setup_grid_spacing()

        self._load_params()

        self._geometry = get_geometry(self.get_parameter("COORDINATES"))

    def _setup_patches(self):
        '''Setup of the grid of simulation patches'''

        self._patches = []
        self._patch_ids = - np.ones(self._domain.cpus_per_dimension, np.int32)
        
        n_patch = np.prod(self._domain.cpus_per_dimension)
        for i in range(n_patch):
            patch = Patch(i, self._sim)
            assert(patch.rank == i)


            self._patches.append(patch)
            self._patch_ids[patch.index] = patch.rank
        
        assert( not any(self._patch_ids.flat == -1))

    def _setup_grid_spacing(self):
        self._xe = [None, None, None]
        self._xm = [None, None, None]

        for i, x in zip(range(3), 'xyz'):
            if x in self._domain.dimensions: 
                self._xe[i] = np.genfromtxt(os.path.join(self._sim, 
                                                         'domain_'+x+'.dat'))
                #FARGO-3D always uses the arithmetic average:
                self._xm[i] = 0.5*(self._xe[i][1:] + self._xe[i][:-1])

                ngh = self._domain.num_ghost_cells[i]
                self._xe[i] = self._xe[i][slice(ngh, len(self._xe[i]) - ngh)]
                self._xm[i] = self._xm[i][slice(ngh, len(self._xm[i]) - ngh)]


    def get_snap_numbers(self):
        '''Find the snapshot numbers for this simulation'''

        Sim = self._sim

        # This function works by examining the number of density snapshots
        files = [f for f in os.listdir(Sim) if f.startswith('gasdens')]

        snap_nums = []
        for f in os.listdir(Sim):
            if f.startswith('gasdens'):
                if '_' in f:
                    if f.endswith('_0.dat'):
                        snap_nums.append(int(f[7:-6]))
                else:
                    snap_nums.append(int(f[7:-4])) 
        
        if 9999 in snap_nums:
            snap_nums.remove(9999)

        return sorted(snap_nums)
        
        
    def load_field(self, field, n=None):
        '''Load the given field (with optional number)'''
        if n is not None:
            field += str(n)

        Sim = self._sim

        if _if_merged(Sim, field):
            dims = self._domain.grid_size
            fName = os.path.join(Sim, field+'.dat')
            field = np.fromfile(fName).reshape(dims[::-1])
            field = np.array(np.swapaxes(field, 0, 2))
        else:
            field = self._load_field_multi(field)

        # Flatten empty axes
        shape = [s for s in field.shape if s > 1]
        return field.reshape(*shape)

    def _load_field_multi(self, field):
        '''Load and join field from multiple files'''

        # First load the sub-arrays
        cpus = self._domain.cpus_per_dimension

        fields = []
        for i in range(cpus[0]):
            fields.append([])
            for j in range(cpus[1]):                
                # First load the sub-arrays
                fields[-1].append([])
                for k in range(cpus[2]):
                    rank = self._patch_ids[i,j,k]
                    patch = self._patches[rank]

                    name = os.path.join(self._sim,field+ '_{}.dat'.format(rank))
                    shape = patch.shape[::-1]
                    fields[-1][-1].append(np.fromfile(name).reshape(shape))

                # Stack along z:
                fields[-1][-1] = np.vstack(fields[-1][-1])
            # Stack along y:
            fields[-1] = np.hstack(fields[-1])
        fields = np.dstack(fields)
        return np.array(np.swapaxes(fields, 0, 2))

    def _load_params(self):
        self._params = {}
        with open(os.path.join(self._sim, 'variables.par'), 'r') as f:
            for line in f:
                skip = re.match("\s*#+",line)
                if skip != None:
                    continue    
                
                key = re.search("(\w+)\s",line)
                if key is None: continue
                
                # Real
                var = re.match("(\w+)\s+(\+?-?\d+\.\d+e?[+-]?\d*)",line)
                if var is not None:
                    name  = var.group(1).upper()
                    value = var.group(2)
                    self._params[name] = float(value)
                    continue

                # Int / Bool
                var = re.match("(\w+)\s+(\+?-?\d+)\s+",line)
                if var is not None:
                    name  = var.group(1).upper()
                    value = var.group(2)
                    self._params[name] = int(value)
                    continue
                
                # String
                var = re.match("(\w+)\s+(.*)\s?",line)
                if var is not None:
                    name  = var.group(1).upper()
                    value = var.group(2)
                    self._params[name] = value
                    continue

                # Error
                raise AttributeError("Error!:\nCould not parse parameter: {}"
                                     "".format(line))

    def get_parameter(self, key):
        '''Load a parameter from the simulation.'''
        return self._params[key.upper()]

    def __getattr__(self, attr):
        '''Finds dimension aliases from the geometry package'''
        try:
            return self.__getattribute__(self._geometry.get_alias(attr))
        except KeyError:
            raise AttributeError("FARGO_Sim has no such attribute or alias")

    def __dir__(self):
        '''Lists attributes'''
        return dir(type(self)) + self._geometry.aliases()        

    @property
    def x1(self):
        return self._xe[0]
    @property
    def x2(self):
        return self._xe[1]
    @property
    def x3(self):
        return self._xe[2]

    @property
    def x1med(self):
        return self._xm[0]
    @property
    def x2med(self):
        return self._xm[1]
    @property
    def x3med(self):
        return self._xm[2]


    @property
    def domain(self):
        return self._domain


if __name__  == "__main__":
    import sys
    import matplotlib.pyplot as plt

    Sim = FARGO_Sim(sys.argv[1])

    if len(sys.argv) > 2:
        n = int(sys.argv[2])
    else:
        n = Sim.get_snap_numbers()[-1]

    d = Sim.load_field('gasdens', n)

    plt.pcolormesh(Sim.y[:-1], Sim.x[:-1], np.log10(d[...,0]))
    plt.colorbar(label=r'$\Sigma$')
    plt.xlim(Sim.y[0], Sim.y[-2])
    plt.ylim(Sim.x[0], Sim.x[-2])
    plt.xlabel(r'$R$')
    plt.ylabel(r'$\phi$')
    plt.xscale('log')



    h0 = Sim.get_parameter('ASPECTRATIO')
    kF = Sim.get_parameter('FLARINGINDEX')
    
    cs = h0 * Sim.Rmed**(kF - 0.5)
    P0 = Sim.get_parameter('SIGMA0') * h0*h0

    halfn = Sim.domain.grid_size[0]/2
    ncut = 5
    for i in np.linspace(0, n, 6):
        i = int(i)
        d = Sim.load_field('gasdens', i)
        d_bar = 0.5*(d[:halfn-ncut,:,0].mean(0) + d[halfn+ncut:,:,0].mean(0))

        plt.figure(2)
        plt.plot(Sim.Rmed, d_bar, label=str(i))

        plt.figure(3)
        P = d_bar*cs*cs
        plt.plot(Sim.Rmed, P/P0, label=str(i))


    plt.figure(2)
    plt.xlabel(r'$R$')
    plt.ylabel(r'$\Sigma$')
    plt.legend()

    plt.figure(3)
    plt.xlabel(r'$R$')
    plt.ylabel(r'$P/P_0$')
    plt.legend()

    plt.show()