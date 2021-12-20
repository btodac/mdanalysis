from collections import defaultdict 
"""
Created on Wed Nov 24 13:09:52 2021

@author: mtolladay
"""
class mol_graph:
    G = defaultdict(list)
    atoms = set()
    def __init__(self, atoms=None, bonds=None):
        if atoms is not None:
            self.atoms = set(atoms)
        
        if bonds is not None:
            self.add_bonds(bonds)
            #self.atoms = set(G.keys())
            testatoms = set(bonds.flatten())
            if len(self.atoms) == 0:
                self.atoms = testatoms
            else:
                if not testatoms.issubset(self.atoms):
                    self.atoms = self.atoms.union(testatoms)
    
    def keys(self):
        return self.G.keys()
    
    def add_bond(self, bond):
        a, b = bond
        self.G[a].append(b)
        self.G[b].append(a)
    
    def add_bonds(self, bonds):
        for bond in bonds:
            self.add_bond(bond)

    def remove_bond(self, bond):
        a, b = bond
        self.G[a].remove(b)
        self.G[b].remove(a)
        
    def remove_bonds(self, bonds):
        for bond in bonds:
            self.remove_bond(bond)
              
    def add_atoms(self, atoms):
        if type(atoms) is int:
            atoms = set([atoms])
        elif type(atoms) is not set:
            atoms = set(atoms)
        for atom in atoms - self.atoms:
            self.G[atom] = []
        self.atoms = self.atoms.union(atoms)
                   
    def remove_atoms(self, atoms):
        if type(atoms) is int:
            atoms = set([atoms])
        elif type(atoms) is not set:
            atoms = set(atoms)
        
        for atom in atoms:
            del self.G[atom]
        self.atoms = self.atoms - atoms
        for key in self.G.keys():
            self.G[key] = list(set(self.G[key]) - atoms)
    
    def bonded_components(self):
        seen = set()
        components = []
        for v in self.atoms:
            if v not in seen:
                c = self.bfs(v)
                seen.update(c)
                components.append(c)
        return components
    
    def bfs(self,atom_indx):
        """A fast BFS node generator"""
        seen = set()
        nextlevel = {atom_indx}
        while nextlevel:
            thislevel = nextlevel
            nextlevel = set()
            for v in thislevel:
                if v not in seen:
                    seen.add(v)
                    nextlevel.update(self.G[v])
        return seen