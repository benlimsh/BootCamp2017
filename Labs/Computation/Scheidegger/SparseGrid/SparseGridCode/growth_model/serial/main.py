#======================================================================
#
#     This routine solves an infinite horizon growth model
#     with dynamic programming and sparse grids
#
#     The model is described in Scheidegger & Bilionis (2017)
#     https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2927400
#
#     external libraries needed:
#     - IPOPT (https://projects.coin-or.org/Ipopt)
#     - PYIPOPT (https://github.com/xuy/pyipopt)
#     - TASMANIAN (http://tasmanian.ornl.gov/)
#
#     Simon Scheidegger, 11/16 ; 07/17
#======================================================================

import nonlinear_solver_initial as solver     #solves opt. problems for terminal VF
import nonlinear_solver_iterate as solviter   #solves opt. problems during VFI
from parameters import *                      #parameters of model
import interpolation as interpol              #interface to sparse grid library/terminal VF
import interpolation_iter as interpol_iter    #interface to sparse grid library/iteration
import postprocessing as post                 #computes the L2 and Linfinity error of the model

import TasmanianSG                            #sparse grid library
import numpy as np

#======================================================================
def main(n_agents, iDepth, thetavec):
    # Start with Value Function Iteration

    # terminal value function

    valnew=TasmanianSG.TasmanianSparseGrid()
    if (numstart==0):
        valnew=interpol.sparse_grid(n_agents, iDepth, theta = thetavec[2])
        valnew.write("valnew_1." + str(numstart) + ".txt") #write file to disk for restar

        # value function during iteration
    else:
        valnew.read("valnew_1." + str(numstart) + ".txt")  #write file to disk for restar\

    valold0=TasmanianSG.TasmanianSparseGrid()
    valold1=TasmanianSG.TasmanianSparseGrid()
    valold2=TasmanianSG.TasmanianSparseGrid()
    valold3=TasmanianSG.TasmanianSparseGrid()
    valold4=TasmanianSG.TasmanianSparseGrid()

    valold0.copyGrid(valnew)
    valold1.copyGrid(valnew)
    valold2.copyGrid(valnew)
    valold3.copyGrid(valnew)
    valold4.copyGrid(valnew)

    valold = [valold0, valold1, valold2, valold3, valold4]

    for i in range(numstart, numits):

        valnew0 = TasmanianSG.TasmanianSparseGrid()
        valnew1 = TasmanianSG.TasmanianSparseGrid()
        valnew2 = TasmanianSG.TasmanianSparseGrid()
        valnew3 = TasmanianSG.TasmanianSparseGrid()
        valnew4 = TasmanianSG.TasmanianSparseGrid()

        valnew0 = interpol_iter.sparse_grid_iter(n_agents, iDepth, valold, thetavec[0])
        evalpoints0 = valnew0.getPoints()
        print "valnew0", valnew0.evaluateBatch(evalpoints0)[:,0]
        valnew1 = interpol_iter.sparse_grid_iter(n_agents, iDepth, valold, thetavec[1])
        evalpoints1 = valnew1.getPoints()
        print "valnew1", valnew1.evaluateBatch(evalpoints1)[:,0]
        valnew2 = interpol_iter.sparse_grid_iter(n_agents, iDepth, valold, thetavec[2])
        evalpoints2 = valnew2.getPoints()
        print "valnew2", valnew2.evaluateBatch(evalpoints2)[:,0]
        valnew3 = interpol_iter.sparse_grid_iter(n_agents, iDepth, valold, thetavec[3])
        evalpoints3 = valnew3.getPoints()
        print "valnew3", valnew3.evaluateBatch(evalpoints3)[:,0]
        valnew4 = interpol_iter.sparse_grid_iter(n_agents, iDepth, valold, thetavec[4])
        evalpoints4 = valnew4.getPoints()
        print "valnew4", valnew4.evaluateBatch(evalpoints4)[:,0]
        valold0.copyGrid(valnew0)
        valold1.copyGrid(valnew1)
        valold2.copyGrid(valnew2)
        valold3.copyGrid(valnew3)
        valold4.copyGrid(valnew4)

        valold = [valold0, valold1, valold2, valold3, valold4]

        for j in range(5):
            valold[j].write("valnew_1." + str(i+1) + str(j) + ".txt")
    # compute errors
    avg_err=post.ls_error(n_agents, numstart, numits, No_samples)

    return avg_err

print(main(2,2,thetavec))

'''
errlist = []
for i in range(1,3):
    for j in range(2,4):
        errlist.append(main(i,j,thetavec))

print "Max and Avg Error: ", '\n'
print "Agents: 1, Depth: 2", errlist[0], '\n'
print "Agents: 1, Depth: 3", errlist[1], '\n'
print "Agents: 2, Depth: 2", errlist[2], '\n'
print "Agents: 2, Depth: 3", errlist[3], '\n'
'''
