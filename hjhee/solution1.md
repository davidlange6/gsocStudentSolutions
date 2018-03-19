
Author: [Jianhang He (hjhee)](https://hjhee.github.io/about.html)

Contact: hjhee0@gmail.com

# Task

Write a functional analysis chain that identifies Higgs bosons

# Analysis

It took me a day to understand the problem, to be more certain, I would describe the problem in my own word.

'H' stands for Higgs boson, 'Z' for Z boson, 'E' for electron and 'M' for muon, and '->' means decay. So H->2Z, and Z->(2M|2E). Ultimately:

H->(4M|4E|2M2E)

# Data

What I all have is the measuremnt of each event, in which I should focus on `event.muons` and `event.electrons`. From the problem description I've known that through the equation between momentum P, energy E, and mass M. So I should compute mass M with a combination of `event.muons` and `event.electrons`. Then make the result stored in an array for further analysis (for example make a histogram).

The first step is to figure out the data structure of `event.muons` and `event.electrons`. It's something like the following:

```python
# each event has a set of measurement of muons and electrons
events == [[event.muons, event.electrons], [event.muons, event.electrons], ..., [event.muons, event.electrons]]
# event.muons has the same attributes as event.electrons
event.muons == [energy, px, py, pz]
```

Next I should generate a combination of muons and electrons like the following:

```python
# generate combination for each event
combs = [comb, comb, ..., comb]
# the combination of electron and muon
comb = [(0M4E), (2M2E), (4M0E), (0M8E, 2M6E, 4M4E, ..., 8M0E), ..., (64M0E, ..., 0M64E), ...]
# note that each tuple above have also different possible combinations, for example:
event.electrons == (E0, E1, E2, E3, E4, E5, E6)
(0M4E) = ((E0, E1, E2, E3), (E0, E1, E2, E4), (E0, E1, E2, E5), ..., (E3, E4, E5, E6))
```

I can now apply functional chain to `combs` and compute mass M.


```python
import oamap.source.root
import uproot
events = uproot.open("file:///home/hjhee/HEP/gsoc2018-prep-problem/HZZ.root")["events"].oamap()
events.schema.content.rename("NElectron", "electrons")
events.schema.content["electrons"].content.rename("Electron_Px", "px")
events.schema.content["electrons"].content.rename("Electron_Py", "py")
events.schema.content["electrons"].content.rename("Electron_Pz", "pz")
events.schema.content["electrons"].content.rename("Electron_E", "energy")
events.schema.content["electrons"].content.rename("Electron_Iso", "isolation")
events.schema.content["electrons"].content.rename("Electron_Charge", "charge")
events.schema.content.rename("NMuon", "muons")
events.schema.content["muons"].content.rename("Muon_Px", "px")
events.schema.content["muons"].content.rename("Muon_Py", "py")
events.schema.content["muons"].content.rename("Muon_Pz", "pz")
events.schema.content["muons"].content.rename("Muon_E", "energy")
events.schema.content["muons"].content.rename("Muon_Iso", "isolation")
events.schema.content["muons"].content.rename("Muon_Charge", "charge")
events.schema.content.rename("NPhoton", "photons")
events.schema.content["photons"].content.rename("Photon_Px", "px")
events.schema.content["photons"].content.rename("Photon_Py", "py")
events.schema.content["photons"].content.rename("Photon_Pz", "pz")
events.schema.content["photons"].content.rename("Photon_E", "energy")
events.schema.content["photons"].content.rename("Photon_Iso", "isolation")
events.schema.content.rename("NJet", "jets")
events.schema.content["jets"].content.rename("Jet_Px", "px")
events.schema.content["jets"].content.rename("Jet_Py", "py")
events.schema.content["jets"].content.rename("Jet_Pz", "pz")
events.schema.content["jets"].content.rename("Jet_E", "energy")
events.schema.content["jets"].content.rename("Jet_ID", "id")
events.schema.content["jets"].content.rename("Jet_btag", "btag")
events.regenerate()
```


```python
import functional
from math import *
def mass(*particles):
    energy = particles.map(lambda particle: particle.energy).sum
    px = particles.map(lambda particle: particle.px).sum
    py = particles.map(lambda particle: particle.py).sum
    pz = particles.map(lambda particle: particle.pz).sum
    ret = sqrt(energy**2 - px**2 - py**2 - pz**2)
    return ret
```

# Combination

I tried to generate combination in various way by utilizing the method in `functional.py`, including `reduce`, but quickly find the result unreadable, as demostrated in the following cell:


```python
# if I choose to continue with this way, I should find a way to flatten these structure
[1, 2, 3, 4, 5].aggregate(lambda x, y: [[x], [x, y], [y]], [])
```




    [[[[[[[[[[[]], [[], 1], [1]]], [[[[]], [[], 1], [1]], 2], [2]]],
         [[[[[[]], [[], 1], [1]]], [[[[]], [[], 1], [1]], 2], [2]], 3],
         [3]]],
       [[[[[[[[]], [[], 1], [1]]], [[[[]], [[], 1], [1]], 2], [2]]],
         [[[[[[]], [[], 1], [1]]], [[[[]], [[], 1], [1]], 2], [2]], 3],
         [3]],
        4],
       [4]]],
     [[[[[[[[[[]], [[], 1], [1]]], [[[[]], [[], 1], [1]], 2], [2]]],
         [[[[[[]], [[], 1], [1]]], [[[[]], [[], 1], [1]], 2], [2]], 3],
         [3]]],
       [[[[[[[[]], [[], 1], [1]]], [[[[]], [[], 1], [1]], 2], [2]]],
         [[[[[[]], [[], 1], [1]]], [[[[]], [[], 1], [1]], 2], [2]], 3],
         [3]],
        4],
       [4]],
      5],
     [5]]



Then I found list generator very useful. First duplicate all elements in the list using enumerate, then apply filter mask on the list.


```python
# sample code for generating combination
comb = events[4].electrons
[comb
 .enumerate # [[(0, <Record at index 0>), (1, <Record at index 1>)], [(0, <Record at index 0>), (1, <Record at index 1>)], ..., [(0, <Record at index 0>), (1, <Record at index 1>)]]
 .filter(lambda x: i&2**x[0]) # [[], [(0, <Record at index 0>)], [(1, <Record at index 1>)], [(0, <Record at index 0>), (1, <Record at index 1>)]]
 .map(lambda x: x[1]) # the final result
 for i in range(2**comb.size)]
```




    [[],
     [<Record at index 0>],
     [<Record at index 1>],
     [<Record at index 0>, <Record at index 1>]]



# Solution 1

I will use the primitive way to compute mass M:

1. generate a combination `comb_muons` of muons
2. generate a combination `comb_electrons` of electrons
3. generate a combination `comb` of muons and electrons
4. apply function `mass` to `comb`


```python
comb_electrons = (events
          .lazy
          .filter(lambda event: event.muons.size + event.electrons.size >= 4)
          # step 1 and 2 using the list generator
          .map(lambda event: 
               [[p
                 .enumerate # [("0", E1), ("1", E2), ...]
                 # generate all possible combination (without repetitions) in bit representation
                 .filter(lambda x: i&(2**x[0])) # [("0", E1), ("1", E2)] for i=0x3 ('11' in binary form)
                 # remove index number
                 .map(lambda x: x[1]) # [E1, E2]
                 # generate all possible combination of particles
                 for i in range(2**p.size)] # [[], [E1], ..., [E1, E2], ...]
               # generate a combination of particles for both electrons and muons
               for p in [event.electrons, event.muons]]) # [[[], [E1], ..., [E1, E2], ...], [[], [M1], ..., [M1, M2], ...]]
            # step 3
            .map(lambda event: 
               event
               .table(lambda x, y: x+y) # [[M1, M2, E1], [E1, E2], [M1, M2, E1, E2], ...]
               .filter(lambda event: event.size % 4 == 0) # [[], [M1, M2, E1, E2], ...]
               .flatten) # [[M1, M2, E1, E2], ...]
          # step 4
          .map(lambda event: mass(*event)) # mass(*[M1, M2, E1, E2])
          )
comb_electrons.take(10)
```




    [346.099158473565,
     178.28400666312837,
     176.44328803498277,
     217.4654207657191,
     561.5150465241483,
     183.75591764913608,
     256.18687699522224,
     202.54880254271853,
     219.6200274474909,
     145.5279483330699]



It is also possible to add a new method `comber` to `functional.py`, that would make the problem a lot easier:
```python
def comber(lst):
    '''
    Generate all combinations of elements from input list.
    
    Examples: [1, 2, 3].comb(lambda x: x.size % 2 == 0) == [[], [1, 2], [1, 3], [2, 3]]
    '''
    def comb(lis):
        if lis:
            yield [lis[0]]
            for r in comb(lis[1:]):
                # workaround for flattener needed
                if isinstance(r, (list, tuple, ListProxy)):
                    yield [r].flatten
                    if isinstance(lst[0], (list, tuple, ListProxy)):
                        yield [lis[0], r]
                    else:
                        yield [[lis[0]], r].flatten
                else:
                    yield [r]
    def gen(f):
        if f([]):
            yield []
        for x in comb(lst):
            if f(x):
                yield x
    out = gen
    out.func_name = "[...].comb"
    out.__doc__ = comber.__doc__
    return out
```


```python
# example code refering https://stackoverflow.com/questions/38254304/can-generators-be-recursive
from oamap.proxy import ListProxy
def recursive_generator(lis):
    if lis:
        yield [lis[0]]
        for r in recursive_generator(lis[1:]):
            if isinstance(r, (list, tuple, ListProxy)):
                yield [r].flatten
                if isinstance(lis[0], (list, tuple, ListProxy)):
                    yield [lis[0], r]
                else:
                    yield [[lis[0]], r].flatten
            else:
                yield [r]
for k in recursive_generator([6,3,9,1]):
    print(k)
```

    [6]
    [3]
    [6, 3]
    [9]
    [6, 9]
    [3, 9]
    [6, 3, 9]
    [1]
    [6, 1]
    [3, 1]
    [6, 3, 1]
    [9, 1]
    [6, 9, 1]
    [3, 9, 1]
    [6, 3, 9, 1]



```python
import functional
[1, 2, 3].comb(lambda x: x.size % 2 == 0).collect
```




    [[], [1, 2], [1, 3], [2, 3]]



# Solution 2

With the new method I can write the functional chain a lot clearer, as the following cell shows. But I still wonder if I can implement `comber` just by using `reduce`, that would make `reduce` more powerful as I suppose.

Notice the following code need to be reviewed. Because `table` can't be applied to `generator`, I must apply `collect` to the combinations, which may slow down the computation.


```python
comb_electrons = (events
          .lazy
          .filter(lambda event: event.muons.size + event.electrons.size >= 4)
          # step 1 and 2 using the list generator
          .map(lambda event: 
               # I need to add .collect here, otherwise these won't work as expected
               # review needed
               [event.electrons.comb(lambda x: True).collect,
                event.muons.comb(lambda x: True).collect]) # [[[], [E1], ..., [E1, E2], ...], [[], [M1], ..., [M1, M2], ...]]
          .map(lambda event: 
               event
               .table(lambda x, y: x+y) # [[M1, M2, E1], [E1, E2], [M1, M2, E1, E2], ...]
               .filter(lambda event: event.size % 4 == 0) # [[], [M1, M2, E1, E2], ...]
               .flatten) # [[M1, M2, E1, E2], ...]
          # step 4
          .map(lambda event: mass(*event)) # mass(*[M1, M2, E1, E2])
          )
comb_electrons.take(10)
```




    [346.099158473565,
     178.28400666312837,
     176.44328803498277,
     217.4654207657191,
     561.5150465241483,
     183.75591764913608,
     256.18687699522224,
     202.54880254271853,
     219.6200274474909,
     145.5279483330699]


