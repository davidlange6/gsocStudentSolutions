
Author: [Jianhang He (hjhee)](https://hjhee.github.io/about.html)

Contact: hjhee0@gmail.com


```python
>>> import uproot
>>> columnar_events = uproot.open("./HZZ.root")["events"]
>>> columns = columnar_events.arrays(["*Muon*"])
>>> Muon_Px = columns["Muon_Px"].content
>>> Muon_Py = columns["Muon_Py"].content
>>> Muon_Pz = columns["Muon_Pz"].content
import numpy
Muon_P = numpy.sqrt(Muon_Px**2 + Muon_Py**2 + Muon_Pz**2)
Muon_E = columns["Muon_E"].content
column_len = len(columns["Muon_E"])
print column_len
print len(Muon_P)
```

    2421
    3825


# Problem to consider: Vectorizing mass calculations efficiently
The problem for you to solve is the following: perform Z mass calculations in the fewest possible vectorized steps. There are three scales in this problem: (1) the number of events, (2) the number of muons per event, and (3) the number of muon pairs per event. There will be multiple Z candidates in each event, not simply because the Higgs decays into two of them, but also because the same list of muons can be combined in multiple ways.

Bonus for also reducing the multiple Z candidates per event to the single best Z candidate per event (closest to 91 GeV). Double bonus for optimizing Higgs candidates. Triple bonus for hiding the vectorized function under a functional interface. These things are what the summer project is about (though we'll have more problems than just computing masses).

# Analysis

Goal: perform Z mass calculation
First implement Z mass calculation in loops, then vectorize the loop.

I saw that in the example the program flattens the events, as the following code has shown:

```python
print len(columns["Muon_Px"])
2421
print len(Muon_Px) # flatten Muon_Px
3825
```

I think the reason is that GPU can only accept a flat list, so in order to preserve the structure of each event the example store this informations in `starts` and `stops`. This could be useful for filtering for the best candidates per event. Following the example, I assume that I can precompute a lookup table for pairs like the following:


```python
# naive
pairs = [[] for i in range(column_len)]
for i in range(column_len): # loop events
    muon_len = len(columns["Muon_E"][i])
    # print 'event {0} has {1} muons'.format(i, muon_len)
    for j in range(muon_len):
        for k in range(j+1, muon_len):
            pairs[i] += [(j, k)]
# print pairs
```

Next I would start calculate Z mass, utilizing the existing code:

1. calculate P for each muon `P[index]**2 == Px[index]**2 + Py[index]**2 + Pz[index]**2`
2. calculate E-P `Mass[index]**2 == E[index]**2 - P[index]**2 == E[index]**2 - Px[index]**2 - Py[index]**2 - Pz[index]**2`

Since I've generate the pairs, the next step is store the result to a new list `M`, so that `len(M[index]) == 1 + len(pairs[index])`

1. calculate sum of Mass pairs `M[index+i]**2 == Mass[index+pairs[index][i][0]]**2 + Mass[index+pairs[index][i][1]]**2`, where `i == range(len(pairs[index]))`

2. (bonus) choose the best Mass for each event `M_best[index] == min(abs(91 - M[index+[0..len(pairs[index])]]))`

There are too many indexes in the above code, which may has impact on page fault. I can also use the lookup table version of pairs like in the example, which is a flat list. But only when I encountered problems with `index` when applying my function to `vectorize`, I'm aware that choosing a proprate `index` matters a lot. Some data:

```python
print len(pairs)
print len(Muon_Px)
print len(Muon_E)
2421
3825
3825
```

I would like to use something like the following in the vectorized function, and let the `index` as `len(pairs)` in order to iterate paris easier:

`M[index]**2 == E[index_pa]**2 - P[index_pa]**2 + E[index_pb]**2 - P[index_pb]**2`, where `index_pa = pairs_flat[i][0]`, `index_pb = pairs_flat[i][1]`, and `i == range(pairs_flat_start[index], pairs_flat_stop[index])`

I should be more clear about the length of each vector to be passed to vector_mass:


```python
# optimize1 with flatten
pairs_flat = [[] for i in range(column_len)]
pairs_flat_counter = 0
for i in range(column_len): # loop events
    muon_len = len(columns["Muon_E"][i])
    # print 'event {0} has {1} muons'.format(i, muon_len)
    for j in range(muon_len):
        for k in range(j+1, muon_len):
            pairs_flat[i] += [(pairs_flat_counter+j, pairs_flat_counter+k)]
    pairs_flat_counter += muon_len
#print pairs_flat
```

After some thought, I think I should avoid using the loop and so make the structure as flat as possible, otherwise I would encounter the stall problem presented in problem 2 description.


```python
# optimize1.1 with flatten
pairs_flat = []
pairs_flat_counter = 0
for i in range(column_len): # loop events
    muon_len = len(columns["Muon_E"][i])
    # print 'event {0} has {1} muons'.format(i, muon_len)
    for j in range(muon_len):
        for k in range(j+1, muon_len):
            pairs_flat += [(pairs_flat_counter+j, pairs_flat_counter+k)]
    pairs_flat_counter += muon_len
#print pairs_flat

# len(Muon_P) == 3825 <- the number of muons in all events
# len(start) == 2421 <- the number of events
# len(pairs) == ? <- the number of muon pairs all events
# Muon_M = numpy.empty(len(pairs))
from math import *
def vector_mass(index, pairs, Muon_Px, Muon_Py, Muon_Pz, Muon_E, Muon_M):
    p = pairs[index]
    index_pa = p[0]
    index_pb = p[1]
    
    e = Muon_E[index_pa] + Muon_E[index_pb]
    px = Muon_Px[index_pa] + Muon_Px[index_pb]
    py = Muon_Py[index_pa] + Muon_Py[index_pb]
    pz = Muon_Pz[index_pa] + Muon_Pz[index_pb]
        
    Muon_M[index] = sqrt(e**2 - px**2 - py**2 - pz**2)

def vectorize_test(f, l, pairs_flat, Muon_Px, Muon_Py, Muon_Pz, Muon_E, Muon_M):
    for i in range(l):
        f(i, pairs_flat, Muon_Px, Muon_Py, Muon_Pz, Muon_E, Muon_M)
        
Muon_M = [0] * len(pairs_flat)
#vectorize_test(vector_mass, len(pairs_flat), pairs_flat, Muon_Px, Muon_Py, Muon_Pz, Muon_E, Muon_M)
vectorize(vector_mass, len(pairs_flat), pairs_flat, Muon_Px, Muon_Py, Muon_Pz, Muon_E, Muon_M) # 9
#Muon_M
```

    leading step 0 (100.0% at leading): 
        p = pairs[index]
        ...advancing 1
    
    leading step 1 (100.0% at leading): 
        index_pa = p[0]
        ...advancing 2
    
    leading step 2 (100.0% at leading): 
        index_pb = p[1]
        ...advancing 3
    
    leading step 3 (100.0% at leading): 
        e = (Muon_E[index_pa] + Muon_E[index_pb])
        ...advancing 4
    
    leading step 4 (100.0% at leading): 
        px = (Muon_Px[index_pa] + Muon_Px[index_pb])
        ...advancing 5
    
    leading step 5 (100.0% at leading): 
        py = (Muon_Py[index_pa] + Muon_Py[index_pb])
        ...advancing 6
    
    leading step 6 (100.0% at leading): 
        pz = (Muon_Pz[index_pa] + Muon_Pz[index_pb])
        ...advancing 7
    
    leading step 7 (100.0% at leading): 
        Muon_M[index] = sqrt(((((e ** 2) - (px ** 2)) - (py ** 2)) - (pz ** 2)))
        ...advancing 8
    





    9



It is necessary to keep the structure of pairs when I want to reduce the result and choose the best Z candidates for each event.


```python
# optimize2 for best candidates
pairs = [[] for i in range(column_len)]
pairs_counter = 0
for i in range(column_len): # loop events
    muon_len = len(columns["Muon_E"][i])
    
    for j in range(muon_len):
        for k in range(j+1, muon_len):
            pairs[i] += [(pairs_counter+j, pairs_counter+k)]
            
    pairs_counter += muon_len
    
#print pairs

def vector_mass_bonus(index, pairs, Muon_Px, Muon_Py, Muon_Pz, Muon_E, Muon_M):
    min_diff = float('nan')
    best_M = 0
    
    for p in pairs[index]:
        index_pa = p[0]
        index_pb = p[1]

        e = Muon_E[index_pa] + Muon_E[index_pb]
        px = Muon_Px[index_pa] + Muon_Px[index_pb]
        py = Muon_Py[index_pa] + Muon_Py[index_pb]
        pz = Muon_Pz[index_pa] + Muon_Pz[index_pb]
        
        m = sqrt(e**2 - px**2 - py**2 - pz**2)
        if numpy.isnan(min_diff) or min_diff > abs(91-m):
            min_diff = abs(91-m)
            best_M = m
        
    Muon_M[index] = best_M
    
def vectorize_test1(f, l, pairs, Muon_Px, Muon_Py, Muon_Pz, Muon_E, Muon_M):
    for i in range(l):
        f(i, pairs, Muon_Px, Muon_Py, Muon_Pz, Muon_E, Muon_M)
        
Muon_M = [0] * column_len # number of events
#vectorize_test1(vector_mass_bonus, len(pairs), pairs, Muon_Px, Muon_Py, Muon_Pz, Muon_E, Muon_M)
vectorize(vector_mass_bonus, len(pairs), pairs, Muon_Px, Muon_Py, Muon_Pz, Muon_E, Muon_M) # 59
#Muon_M
```

    leading step 0 (100.0% at leading): 
        min_diff = float('nan')
        ...advancing 1
    
    leading step 1 (100.0% at leading): 
        best_M = 0
        ...advancing 2
    
    leading step 2 (100.0% at leading): 
        for p in pairs[index]:
            index_pa = p[0]
            index_pb = p[1]
            e = (Muon_E[index_pa] + Muon_E[index_pb])
            px = (Muon_Px[index_pa] + Muon_Px[index_pb])
            py = (Muon_Py[index_pa] + Muon_Py[index_pb])
            pz = (Muon_Pz[index_pa] + Muon_Pz[index_pb])
            m = sqrt(((((e ** 2) - (px ** 2)) - (py ** 2)) - (pz ** 2)))
            if (numpy.isnan(min_diff) or (min_diff > abs((91 - m)))):    
                min_diff = abs((91 - m))
                best_M = m
        ...advancing 3
    
    leading step 13 (41.64% at leading): 
        Muon_M[index] = best_M
        ...catching up 4 (41.64% at leading)
        ...catching up 5 (41.64% at leading)
        ...catching up 6 (41.64% at leading)
        ...catching up 7 (41.64% at leading)
        ...catching up 8 (41.64% at leading)
        ...catching up 9 (41.64% at leading)
        ...catching up 10 (41.64% at leading)
        ...catching up 11 (41.64% at leading)
        ...catching up 12 (41.64% at leading)
        ...catching up 13 (41.64% at leading)
        ...catching up 14 (98.27% at leading)
        ...catching up 15 (98.27% at leading)
        ...catching up 16 (98.27% at leading)
        ...catching up 17 (98.27% at leading)
        ...catching up 18 (98.27% at leading)
        ...catching up 19 (98.27% at leading)
        ...catching up 20 (98.27% at leading)
        ...catching up 21 (98.27% at leading)
        ...catching up 22 (98.27% at leading)
        ...catching up 23 (98.27% at leading)
        ...catching up 24 (98.27% at leading)
        ...catching up 25 (98.27% at leading)
        ...catching up 26 (98.27% at leading)
        ...catching up 27 (98.27% at leading)
        ...catching up 28 (98.27% at leading)
        ...catching up 29 (98.27% at leading)
        ...catching up 30 (98.97% at leading)
        ...catching up 31 (98.97% at leading)
        ...catching up 32 (99.5% at leading)
        ...catching up 33 (99.5% at leading)
        ...catching up 34 (99.67% at leading)
        ...catching up 35 (99.67% at leading)
        ...catching up 36 (99.67% at leading)
        ...catching up 37 (99.67% at leading)
        ...catching up 38 (99.67% at leading)
        ...catching up 39 (99.67% at leading)
        ...catching up 40 (99.67% at leading)
        ...catching up 41 (99.67% at leading)
        ...catching up 42 (99.67% at leading)
        ...catching up 43 (99.67% at leading)
        ...catching up 44 (99.67% at leading)
        ...catching up 45 (99.67% at leading)
        ...catching up 46 (99.67% at leading)
        ...catching up 47 (99.67% at leading)
        ...catching up 48 (99.67% at leading)
        ...catching up 49 (99.67% at leading)
        ...catching up 50 (99.67% at leading)
        ...catching up 51 (99.67% at leading)
        ...catching up 52 (99.67% at leading)
        ...catching up 53 (99.67% at leading)
        ...catching up 54 (99.79% at leading)
        ...catching up 55 (99.79% at leading)
        ...catching up 56 (99.88% at leading)
        ...catching up 57 (99.88% at leading)
        ...advancing 58
    





    59



When I try to select the best candidates in the loop, it seems that the `vector_mass_bonus` would speen far more time than `vector_mass`, which I thought it would be better if I perform a second pass like the following:


```python
# optimize3 for lookup table
pairs_flat = []
pairs_flat_start = []
pairs_flat_stop = []
pairs_flat_counter = 0
pairs_flat_counter_event = 0
for i in range(column_len): # loop events
    muon_len = len(columns["Muon_E"][i])
    for j in range(muon_len):
        for k in range(j+1, muon_len):
            pairs_flat += [(pairs_flat_counter+j, pairs_flat_counter+k)]
    pairs_flat_start.append(pairs_flat_counter_event / 2)
    pairs_flat_counter += muon_len
    pairs_flat_counter_event += muon_len**2 - muon_len
    pairs_flat_stop.append(pairs_flat_counter_event / 2)
        
Muon_M = [0] * len(pairs_flat)
vectorize_test(vector_mass, len(pairs_flat), pairs_flat, Muon_Px, Muon_Py, Muon_Pz, Muon_E, Muon_M)
vectorize(vector_mass, len(pairs_flat), pairs_flat, Muon_Px, Muon_Py, Muon_Pz, Muon_E, Muon_M) # 9

# use lookup table to select the best candidates
def vector_mass_bonus1(index, start, stop, Muon_M, Muon_M_best):
    min_diff = float('nan')
    best_M = 0
    
    for i in range(start[index], stop[index]):
        m = Muon_M[i]
        if numpy.isnan(min_diff) or min_diff > abs(91-m):
            min_diff = abs(91-m)
            best_M = m
        
    Muon_M_best[index] = best_M
    
Muon_M_best = [0] * column_len # number of events
def vectorize_test2(f, l, start, stop, Muon_M, Muon_M_best):
    for i in range(l):
        f(i, start, stop, Muon_M, Muon_M_best)
vectorize_test2(vector_mass_bonus1, len(pairs_flat_start), pairs_flat_start, pairs_flat_stop, Muon_M, Muon_M_best)
vectorize(vector_mass_bonus1, len(pairs_flat_start), pairs_flat_start, pairs_flat_stop, Muon_M, Muon_M_best) # 23
```

    leading step 0 (100.0% at leading): 
        p = pairs[index]
        ...advancing 1
    
    leading step 1 (100.0% at leading): 
        index_pa = p[0]
        ...advancing 2
    
    leading step 2 (100.0% at leading): 
        index_pb = p[1]
        ...advancing 3
    
    leading step 3 (100.0% at leading): 
        e = (Muon_E[index_pa] + Muon_E[index_pb])
        ...advancing 4
    
    leading step 4 (100.0% at leading): 
        px = (Muon_Px[index_pa] + Muon_Px[index_pb])
        ...advancing 5
    
    leading step 5 (100.0% at leading): 
        py = (Muon_Py[index_pa] + Muon_Py[index_pb])
        ...advancing 6
    
    leading step 6 (100.0% at leading): 
        pz = (Muon_Pz[index_pa] + Muon_Pz[index_pb])
        ...advancing 7
    
    leading step 7 (100.0% at leading): 
        Muon_M[index] = sqrt(((((e ** 2) - (px ** 2)) - (py ** 2)) - (pz ** 2)))
        ...advancing 8
    
    leading step 0 (100.0% at leading): 
        min_diff = float('nan')
        ...advancing 1
    
    leading step 1 (100.0% at leading): 
        best_M = 0
        ...advancing 2
    
    leading step 2 (100.0% at leading): 
        for i in range(start[index], stop[index]):
            m = Muon_M[i]
            if (numpy.isnan(min_diff) or (min_diff > abs((91 - m)))):    
                min_diff = abs((91 - m))
                best_M = m
        ...advancing 3
    
    leading step 7 (41.64% at leading): 
        Muon_M_best[index] = best_M
        ...catching up 4 (41.64% at leading)
        ...catching up 5 (41.64% at leading)
        ...catching up 6 (41.64% at leading)
        ...catching up 7 (41.64% at leading)
        ...catching up 8 (98.27% at leading)
        ...catching up 9 (98.27% at leading)
        ...catching up 10 (98.27% at leading)
        ...catching up 11 (98.27% at leading)
        ...catching up 12 (98.97% at leading)
        ...catching up 13 (98.97% at leading)
        ...catching up 14 (99.5% at leading)
        ...catching up 15 (99.5% at leading)
        ...catching up 16 (99.67% at leading)
        ...catching up 17 (99.67% at leading)
        ...catching up 18 (99.79% at leading)
        ...catching up 19 (99.79% at leading)
        ...catching up 20 (99.88% at leading)
        ...catching up 21 (99.88% at leading)
        ...advancing 22
    





    23



With lookup table I can see a big difference between steps: from 58 to 32. The reason is that I move some calculation out of the loop, so there is fewer stall. So I conclude that in order to minimize the stall step one should avoid `if` and `for` by precomputing lookup tables. But there is a question remain: at what scale is it better to build lookup table, considering CPU time and cost of memory transfer between processor and GPU, or bus? Because this technique can effectively reduce steps, so I would also build a lookup table for finding the best candidates of Higgs. And I should also consider whether this lookup table can be expressed in functional programming interface.
