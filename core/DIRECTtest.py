import math, time
import numpy as np
# import pyswarms as ps
import logging

from DevelopingApproach.BO import functions

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation


from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

from DevelopingApproach.BayesianOptimizer import BayesianOptimizer
from DevelopingApproach.ParticleSwarmOptimizer import ParticleSwarmOptimizer
from hyperopt import hp
from hyperopt import fmin, tpe
from hyperopt.pyll.base import scope

# Fixing random state for reproducibility
np.random.seed(19680801)

def cost_func(x):
    # if x[0] > myfunction.bounds[0][1]:
    #     x[0] = myfunction.bounds[0][1]
    # elif x[0] < myfunction.bounds[0][0]:
    #     x[0] = myfunction.bounds[0][0]
    # if x[1] > myfunction.bounds[1][1]:
    #     x[1] = myfunction.bounds[1][1]
    # elif x[1] < myfunction.bounds[1][0]:
    #     x[1] = myfunction.bounds[1][0]
    #return -1.0 * myfunction.func(x)
    #x = np.floor(x)
    for i in x:
        if i != int(i):
            left = -1.0 * myfunction.func(np.floor(x))
            right = -1.0 * myfunction.func(np.ceil(x))
            slope = (right - left)/np.linalg.norm(np.ceil(x) - np.floor(x))
            #print("slope:",slope)
            result = -1.0 * myfunction.func(np.floor(x)) + np.linalg.norm(x - np.floor(x))*slope
            #print("result:", result)
            return result
    return -1.0 * myfunction.func(x)

def test1DFunc(x):
    return -np.exp(-(x - 2) ** 2) + np.exp(-(x - 6) ** 2 / 10) + 1 / (x ** 2 + 1)

#myfunction = functions.branin()

# x1 = [i for i in range(-5,10)]
# x2 = [i for i in range(0,15)]
# x3 = np.linspace(-5, 10, 100)
# # print("x1:",x1)
# # print("x2:",x2)
#
# Xt = []
# tz = []
# for i in x1:
#     for j in x2:
#         tz.append(-myfunction.func([i,j]))
#         Xt.append([i,j])
#
# Xt2 = []
# tz2 = []
# for i in x3:
#     for j in x2:
#         tz2.append(-myfunction.func([i, j]))
#         Xt2.append([i,j])
# #print("ox:",tz.shape)
#
# # fig = plt.figure(1)
# # ax = Axes3D(fig)
# # for i in range(0,Xt.__len__()):
# #     ax.scatter(Xt[i][0], Xt[i][1], tz[i], color="black")
#
#
# #plt.show()
# xtarget = np.linspace(-5, 15, 100).reshape(-1, 1)
# ytarget = test1DFunc(xtarget)
#
# fig = plt.figure(1)
# plt.plot(xtarget,ytarget)
# #plt.show()
#
# def subDivide(interval):
#     result = []
#     delta = (interval[1] - interval[0])/3
#     result.append([interval[0], interval[0] + delta])
#     result.append([interval[0] + delta, interval[1] - delta])
#     result.append([interval[1] - delta, interval[1]])
#     return result
#
# def checkPotentialOptimal(interval, sigma=0.01):
#     return
#
# def DetermineFcnType(Problem,impcons):
#     retval = 0
#
#     if Problem.constraint == None:
#         if len(Problem.constraint[0].func) == len(Problem.f):
#             if float(Problem.constraint[0].func) == float(Problem.f):
#                 retval = 2
#             else:
#                 retval = 3
#         else:
#             retval = 3
#     else:
#         if impcons:
#             retval = 0
#         else:
#             retval = 1
#
#     if impcons:
#         if retval == 0:
#             retval = 4
#         else:
#             retval = 5

def CallObjFcn(Problem,x,a,b,impcon,calltype,**args):
    fcn_value = 0
    cons_value = 0
    feas_flag = 0
    # Scale variable back to original space
    #print("x:", x)
    #print("a:", a, " b:",b)
    point = np.abs(b-a)*(np.array(x))+a
    #print("point:", point)
    if calltype == 1:
        fcn_value = Problem(point)
    elif calltype == 2:
        fcn_value = Problem(point)
    # if feas_flag == 1:
    #     fcn_value = 10**9
    #     cons_value = 0
    return fcn_value, cons_value, feas_flag

def calc_lbound(lengths,fc,hull,szes):
    # print("hull:",hull)

    hull_length = len(hull)
    hull_lengths = [lengths[i] for i in hull]

    # print("hull length:",hull)
    # print("hull_lengths:", hull_lengths)

    lb = [0]*hull_length
    for i in range(0,hull_length):
        X = np.sum(hull_lengths,1) > np.sum(lengths[hull[i]])
        # print("left equation:", np.sum(hull_lengths,1))
        # print("right equation:", np.sum(lengths[hull[i]]))
        # print("X:", X)
        tmp_rects = np.where(X != 0)[0]
        # print("tmp_rects:", tmp_rects)
        if len(tmp_rects) > 0:
            tmp_f = [fc[hull[tmp]] for tmp in tmp_rects]
            tmp_szes = [szes[hull[tmp]] for tmp in tmp_rects]
            # print("tmp_f:",tmp_f)
            # print("tmp_szes:", tmp_szes)
            tmp_lbs = [(fc[hull[i]]-tmp_f[j])/(szes[hull[i]]-tmp_szes[j]) for j in range(0, len(tmp_f))]
            # print("tmp_ubs:",tmp_ubs)
            lb[i] = np.max(tmp_lbs)
        else:
            lb[i] = -1.976e14
    return lb

def calc_ubound(lengths,fc,hull,szes):
    hull_length = len(hull)
    hull_lengths = [lengths[i] for i in hull]
    ub = [0]*hull_length
    for i in range(0,hull_length):
        X = np.sum(hull_lengths,1) < np.sum(lengths[hull[i]])
        tmp_rects = np.where(X != 0)[0]
        if len(tmp_rects) > 0:
            tmp_f = [fc[hull[tmp]] for tmp in tmp_rects]
            tmp_szes = [szes[hull[tmp]] for tmp in tmp_rects]
            tmp_ubs = [(tmp_f[j]-fc[hull[i]])/(tmp_szes[j] - szes[hull[i]]) for j in range(0, len(tmp_f))]
            ub[i] = np.min(tmp_ubs)
        else:
            ub[i] = 1.976e14
    return ub

def find_po(fc, lengths, minval, ep, szes):
    # print("--Input--")
    # print("fc:", fc)
    # print("len(lengths):", len(lengths))
    if len(lengths) == 1:
        diff_szes = np.array([0])
        tmp_max = int(np.max(diff_szes))
        sum_lengths = np.array([0])
    else:
        diff_szes = np.sum(lengths,1)
        tmp_max = int(np.max(diff_szes))
        sum_lengths = np.sum(lengths,1)
    j = 0
    hull = []
    # print("--In Find Po--")
    # print("diff_szes:", diff_szes)
    # print("tmp_max:", tmp_max)
    # print("lengths:", lengths)
    # print("sum_lengths:", sum_lengths)
    for i in range(0, tmp_max+1):
        tmp_idx = np.where(sum_lengths == i)[0]
        #print("tmp_idx:", tmp_idx)
        tmpArr = [fc[tmpIdx] for tmpIdx in tmp_idx]
        # print("tmpArr:", tmpArr)
        if len(tmpArr) == 0:
            tmp_n = np.array([])
        else:
            tmp_n = np.min(tmpArr)
        hullidx = np.where(tmpArr==tmp_n)[0]
        # print("tmp_n:", tmp_n)
        # print("hullidx:", hullidx)
        # print("hull:", hull)
        if len(hullidx) > 0:
            for tmpIdx in hullidx:
                hull.append(tmp_idx[tmpIdx]) #<<< append all hull idex
            j+=1
            tmpCal = np.array([abs(fc[tmpId]-tmp_n) for tmpId in tmp_idx])
            # print("tmpCal:", tmpCal)
            test = [item for item in tmpCal if item <= 1e-13]
            # print("test:", test)
            ties = np.where(tmpCal <= 1e-13)[0]
            # print("ties:", ties)
            if len(ties) > 1:
                # print("ties:", ties)
                # print("tmp_idx:", ties)
                # print("hull:", hull)
                # print("j:", j)
                # print("tmp_idx[1]:",tmp_idx[1])
                # print("j-1 = ",j-1)
                # print("hull[0]:", hull[0])
                # print("len hull:", type(hull))
                #test = [tmp_idx[idx] != hull[j-1] for idx in ties]
                # test = hull[j - 1]
                # print("test:",test)
                mod_ties = np.where([tmp_idx[tmpIdx] != hull[j-1] for tmpIdx in ties])[0]
                #hull = [hull, tmp_idx[ties[mod_ties]]]
                hull.append(tmp_idx[ties[mod_ties[0]]])
                j = len(hull)
    # print("--Before calc bounds--")
    # print("lengths:", lengths)
    # print("fc:", fc)
    # print("hull:", hull)
    # print("szes len:", len(szes))
    lbound = calc_lbound(lengths, fc, hull, szes)
    #print("lbound:",lbound)
    ubound = calc_ubound(lengths, fc, hull, szes)
    #print("ubound:", ubound)
    lminusb = np.array(lbound) - np.array(ubound)
    #print("lminusb:",lminusb)

    maybe_po = np.where(lminusb <= 0)[0]
    # print("maybe_po:", maybe_po)
    hullMaybePo = [hull[i] for i in maybe_po]
    # print("hullMaybePo:",hullMaybePo)
    t_len = len(hullMaybePo)
    # print("tlen:",t_len)
    if minval != 0:
        fc_hull_maybepo = np.array([fc[i] for i in hullMaybePo])
        szes_hull_maybepo = np.array([szes[i] for i in hullMaybePo])
        ubound_maybepo = np.array([ubound[i] for i in maybe_po])
        leftEq = (minval - fc_hull_maybepo) / abs(minval) + szes_hull_maybepo * ubound_maybepo / abs(minval)
        po = np.where(leftEq >= ep)[0]
    else:
        fc_hull_maybepo = np.array([fc[i] for i in hullMaybePo])
        szes_hull_maybepo = np.array([szes[i] for i in hullMaybePo])
        ubound_maybepo = np.array([ubound[i] for i in maybe_po])
        leftEq = fc_hull_maybepo - szes_hull_maybepo*ubound_maybepo
        po = np.where(leftEq <= 0)[0]
    maybebo_po = [maybe_po[i] for i in po]
    # print("po:", po)
    # print("maybebo_po:", maybebo_po)
    # print("hull:", hull)
    # print("szes:", szes)
    final_pos = [hull[i] for i in maybebo_po]
    #szes_finalpos = [szes[i] for i in final_pos]
    szes_finalpos = [szes[i] for i in final_pos]
    # print("final_pos:", final_pos, " szes_finalpos:",szes_finalpos)
    return final_pos, szes_finalpos

def DIRdivide(a, b, Problem, index, thirds,p_lengths,p_fc,p_c,p_con, p_feas_flags,p_fcncounter,p_szes,impcons,calltype,**args):
    lengths = p_lengths
    fc = p_fc
    c = p_c
    szes = p_szes
    fcncounter = p_fcncounter
    con = p_con
    feas_flags = p_feas_flags

    #1. Determine which sides are the largest
    # print("lengths:",lengths)
    # print("c:", c)
    li = lengths[index]
    biggy = math.floor(np.min(li))
    ls = np.where(li == biggy)[0]
    lssize = len(ls);
    j = 0;
    # print("li:",li)
    # print("biggy:", biggy)
    # print("ls:", ls)
    # print("lssize:", lssize)
    #2. Evaluate function in directions of biggest size
    # to determine which direction to make divisions
    oldc = c[index]
    #print("oldc:",oldc)
    #print("thirds:", thirds)
    delta = thirds[biggy]
    # print("c:", c)
    # print("oldc:",oldc)
    # print("delta:", delta)
    # test = [(oldc)]*2
    # print("test:",test)
    # print("oldc:", oldc)
    newc_left = np.copy(oldc)
    # print("type newcleft:",type(newc_left))
    newc_right = np.copy(oldc)
    if(lssize > 1):
        newc_left = [np.copy(oldc)]
        newc_right = [np.copy(oldc)]
        for i in range(1, lssize):
            newc_left.append(np.copy(oldc))
            newc_right.append(np.copy(oldc))
    newc_left = np.array(newc_left)
    newc_right = np.array(newc_right)
    # print("newc_left before for:", newc_left)
    # print("newc_right before for:", newc_right)
    # print("len newc_left before for:", len(newc_left))
    # print("len newc_right before for:", len(newc_right))
    #print("newc_left:", newc_left)
    # f_left = np.zeros((1, lssize))
    # f_right = np.zeros((1, lssize))
    f_left = [0.0]*lssize
    f_right = [0.0]*lssize
    # print("f_left:", f_left)
    for i in range(0,lssize):
        # print("inside fcncounter:",fcncounter)
        lsi = ls[i]
        # print("i: ", i, " lsi:", lsi)
        if lssize == 1:
            newc_left[lsi] = newc_left[lsi] - delta
            newc_right[lsi] = newc_right[lsi] + delta
            # print("input newc left:", newc_left, "i",i)
            # print("input newc right:", newc_right, "i",i)
            f_left[i], _, _ = CallObjFcn(Problem, newc_left, a, b, impcons, calltype)
            f_right[i], _, _ = CallObjFcn(Problem, newc_right, a, b, impcons, calltype)
        else:
            newc_left[i][lsi] = newc_left[i][lsi] - delta
            newc_right[i][lsi] = newc_right[i][lsi] + delta
            # print("input newc left:", newc_left[i], "i lssize:",i)
            # print("input newc right:", newc_right[i], "i lssize:",i)
            f_left[i], _, _ = CallObjFcn(Problem, newc_left[i], a, b, impcons, calltype)
            f_right[i], _, _ = CallObjFcn(Problem, newc_right[i], a, b, impcons, calltype)
        # print("newc_left in for:", newc_left)
        # print("newc_right in for:", newc_right)

        # print("i:",i)
        fcncounter+=2
    # print("newc_left:", newc_left)
    # print("newc_right:", newc_right)
    # print("fcncounter:",fcncounter)
    # print("f_left:",f_left)
    # print("f_right:", f_right)
    # test = np.minimum(f_left, f_right)
    # test = [[test[i], ls[i]] for i in range (0,len(ls))]
    # print("test:", test)
    w = np.minimum(f_left, f_right)
    w = w.tolist()
    w = [[w[i], ls[i]] for i in range(0, len(ls))]
    # print("w:",w)

    # 3. Sort w for division order
    V = np.sort(w,axis=0)
    order = np.argsort(w,axis=0)
    # print("ynew:",ynew)
    # print("indx:", indx)

    # 4. Make divisions in order specified by order
    for i in range(0, len(order)):
        newleftindex = p_fcncounter + 2 * (i);
        newrightindex = p_fcncounter + 2 * (i) + 1;
        if newleftindex >= len(lengths) or newrightindex >= len(lengths):
            isDone = 0
            return lengths, fc, c, con, feas_flags, szes, fcncounter, isDone
        # print("newleftindex:", newleftindex)
        # print("newrightindex:", newrightindex)
        # print("len lengths:", len(lengths))
        # 4.1 create new rectangles identical to the old one
        oldrect = lengths[index]
        # print("lengths[newleftindex]:", lengths[newleftindex])
        #if len(lengths) < newrightindex):
        lengths[newleftindex] = oldrect.copy();
        lengths[newrightindex] = oldrect.copy();
        # print("after lengths[newleftindex]:", lengths[newleftindex])
        # print("newrightindex:", newrightindex)
        # print("index:",index)
        # print("oldrect:", oldrect)
        #print("lengths before slice:", lengths)
        # old, and new rectangles have been sliced in order(i) direction
        lengths[newleftindex,ls[order[i, 0]]] = lengths[index,ls[order[i, 0]]] + 1
        lengths[newrightindex,ls[order[i, 0]]] = lengths[index,ls[order[i, 0]]] + 1
        lengths[index,ls[order[i, 0]]] = lengths[index,ls[order[i, 0]]] + 1
        # print("lengths[ls[order[i,0]]]:",lengths[ls[order[i,0]],newleftindex])
        # print("ls:", ls)
        # print("order:", order[i,0])
        # print("lengths:", lengths)

        # add new columns to c
        # print("newc_left:", newc_left)
        # print("newc_left[order[i]]:", newc_left[0,order[i]])
        #c = np.matrix(c)
        # print("order:", order)
        # print("order[i]:", order[i], " i:", i)

        if lssize == 1:
            c[newleftindex] = newc_left.copy()
            c[newrightindex] = newc_right.copy()
            # print("add newc_left:", newc_left.copy())
            # print("add newc_right:", newc_right.copy())
        else:
            tmpOrder = order.ravel()
            # print("tmpOrder:",tmpOrder)
            # for te in order:
            #     print("check order:", te)
            # for te in newc_left:
            #     print("check new left te", te)
            # for te in newc_right:
            #     print("check new right te", te)
            # print("test lsi:", ls[i])
            # print("test linear index:",newc_left[2])
            # print("test left:", newc_left[tmpOrder[i]], " tmpOrder:", tmpOrder)
            # print("test right:",newc_right[tmpOrder[i]])
            c[newleftindex] = newc_left[tmpOrder[i]].copy()
            c[newrightindex] = newc_right[tmpOrder[i]].copy()
            # print("add newc_left:", c[newleftindex])
            # print("add newc_right:", c[newrightindex])
        #c = np.append(c, [newc_left[0, order[i]].copy(), newc_right[0, order[i]].copy()], axis=0)
        # = np.ndarray(c)


        # add new values to fc
        # print("fc:", fc)
        # print("order[i]:", order.ravel()[i])
        # print("f_left[order[i]]:", f_left[order.ravel()[i]])
        fc[newleftindex] = f_left[order.ravel()[i]]
        fc[newrightindex] = f_right[order.ravel()[i]]
        #print("fc:", fc)

        # store sizes of each rectangle
        #test = np.power(1,lengths[newleftindex])
        # test = 1.0/3*np.ones((len(lengths[0])))
        # test = np.power(test, lengths[newleftindex])
        # test = np.linalg.norm(test)
        # tmp = 1.0/2 * np.linalg.norm(np.power(1.0/3*np.ones((len(lengths[0]))), lengths[newleftindex]))
        # print("tmp:",tmp)
        szes[newleftindex] = 1.0 / 2 * np.linalg.norm(np.power(1.0 / 3 * np.ones((len(lengths[0]))), lengths[newleftindex]))
        szes[newrightindex] = 1.0 / 2 * np.linalg.norm(np.power(1.0 / 3 * np.ones((len(lengths[0]))), lengths[newrightindex]))
        # print("lengths[newleftindex]:", lengths[newleftindex])
    szes[index] = 1.0 / 2 * np.linalg.norm(np.power(1.0 / 3 * np.ones((len(lengths[0]))), lengths[index]))
    #print("szes:", szes)
    isDone = 1

    return lengths,fc,c,con,feas_flags,szes,fcncounter, isDone

def DIRini(Problem,n,a,b,
           p_lengths,p_c,p_fc,p_con, p_feas_flags, p_szes,theglobalmin,
        maxdeep,tflag,g_nargout,impcons,**args):
    l_lengths = np.array(p_lengths)
    l_c = np.array(p_c)
    l_fc = np.array(p_fc)
    l_con = np.array(p_con)
    l_feas_flags = np.array(p_feas_flags)
    szes = p_szes

    # start by calculating the thirds array
    # here we precalculate (1/3)^i which we will use frequently
    l_thirds = [0]*maxdeep
    l_thirds[0] = 1/3
    for i in range(1,maxdeep):
        l_thirds[i] = (1 / 3) * l_thirds[i - 1]
    # first rectangle is the whole unit hyperrectangle
    l_lengths[0] = np.zeros(n)
    # store size of hyperrectangle in vector szes
    szes[0] = 1
    # first element of c is the center of the unit hyperrectangle
    l_c[0] = 1.0 / 2

    # Determine if there are constraints
    #calltype = DetermineFcnType(Problem, impcons)
    calltype = 1 # No constraints at all
    l_fc[0], l_con[0], l_feas_flags[0] = CallObjFcn(Problem, l_c[0], a, b, impcons, calltype)

    fcncounter = 1

    # initialize minval and xatmin to be center of hyper-rectangle
    xatmin = l_c[0]
    minval = l_fc[0]
    if tflag:
        if theglobalmin != 0:
            perror = 100*(minval - theglobalmin)/abs(theglobalmin)
        else:
            perror = 100*minval
    else:
        perror = 2

    history = np.zeros((1,3))
    history[0,0] = 0
    history[0,1] = 0
    history[0,2] = 0

    return l_thirds,l_lengths,l_c,l_fc,l_con, l_feas_flags, minval,xatmin,perror, history,szes,fcncounter,calltype


def DiRect(f, bounds, **args):
    foundMin = 0
    foundXatMin = 0
    hist = 0

    #Initialize the variables
    lengths = []
    c = []
    fc = []
    con = []
    szes = []
    feas_flags = []
    om_lower = np.array([i[0] for i in bounds])
    om_upper = np.array([i[1] for i in bounds])
    fcncounter = 0
    perror = 0
    itctr = 1
    done = 0
    n = len(bounds)

    #Options
    maxits = args.get('max_iters')
    maxevals = args.get('max_evals')
    maxdeep = args.get('max_deep')
    testflag = False
    globalmin = 0
    ep = 1e-4
    tol = 0.01
    showits = 1
    impcons = 0
    pert = 1e-6

    theglobalmin = 0.3979;
    tflag = testflag;

    if tflag == False:
        lengths = np.zeros((n,maxevals + math.floor(.10*maxevals))).T
        # print("Init lengths size: ", lengths.shape)
        c = lengths
        fc = [0.0]*(maxevals + math.floor(.10 * maxevals))
        szes = fc
        con = fc
        feas_flags = fc

    # Call DIRini
    thirds, lengths, c, fc, con, feas_flags, minval, xatmin, perror, history, szes, fcncounter, calltype = \
    DIRini(f,n, om_lower, om_upper, lengths, c, fc, con, feas_flags, szes, theglobalmin, maxdeep,tflag,0,0)

    ret_minval = minval
    ret_xatmin = xatmin

    minval = fc[0] + con[0]

    # print("Main thirds:", thirds)
    # print("Main lengths:",lengths)
    # print("Main ret_minval:", ret_minval)
    # print("Main ret_xatmin:", ret_xatmin)
    # print("Main minval:", minval)

    # print("Main con:", con)
    # print("Main fc:", fc)
    # print("Main szes:", szes)
    # print("Main perror:", perror)

    while perror > tol:
        fc_cons = [fc[i] + con[i] for i in range(0,fcncounter)]
        szesFC = [szes[i] for i in range(0,fcncounter)]
        inLengths = []



        inLengths = [lengths[tmpIdx] for tmpIdx in range(0,fcncounter)]
        print("============================itctr:", itctr, "fcncounter:", fcncounter, " minval:", minval, "at x:",xatmin)
        # print("Main szes:", szes)
        # print("fc_cons:", fc_cons)
        # print("szesFC:", szesFC)
        # print("lengths:", lengths)
        # print("inLengths:",inLengths)

        # Create list S of potentially optimal hyper-rectangles
        S = find_po(fc_cons, inLengths, minval, ep, szesFC)
        # print("S len:",len(S[0]))
        # print("S:", S)
        # print("S[0]:", S[0])
        #%-- Loop through the potentially optimal hrectangles -----------%
        #%-- and divide -------------------------------------------------%
        for i in range(0,len(S[0])):
            # print("start:", S[0][i])
            #print("In loop lengths size: ", lengths.shape)
            lengths, fc, c, con, feas_flags, szes, fcncounter, success = DIRdivide(om_lower, om_upper, f, S[0][i], thirds, lengths, fc, c, con, feas_flags, fcncounter, szes, impcons, calltype)
            if success == 0:
                break
        #-- update minval, xatmin - ------------------------------------- %
        # print("fc:", fc)
        #print("fcncounter:",fcncounter, " maxevals:",maxevals)
        if success == 1:
            tmp = []
            for i in range(0, fcncounter):
                tmp.append(fc[i] + con[i])
            minval = np.min(tmp)
            fminindex = np.argmin(tmp)
            penminval = minval + con[fminindex]
            # print("minval:",minval)
            # print("fminindex:", fminindex)
            # print("penminval:", penminval)
            # print("tmp:", tmp)
            xatmin = (om_upper - om_lower) * c[fminindex] + om_lower;

            # --- update return values
            ret_minval = minval
            ret_xatmin = xatmin
        #print("\n")
        if tflag:
            if theglobalmin != 0:
                perror = 100*(minval - theglobalmin)/abs(theglobalmin)
            else:
                perror = 100*minval
        else:
            if itctr >= maxits:
                print("Exceeded max iterations. Increase maxits")
                break
            if fcncounter > maxevals:
                print("Exceeded max fcn evals. Increase maxevals")
                break
            if np.max(np.where(lengths)) >= maxdeep:
                print("Exceeded Max depth. Increse maxdeep")
                perror = -1
        itctr+=1

    foundMin = ret_minval
    foundXatMin = ret_xatmin

    return foundMin, foundXatMin, c



# l = -5
# u = 15
# bound = ((-5,10),(0,15))


#opts
maxits = 10
maxevals = 97
maxdeep = 100
testflag = 0
globalmin = 0
ep = 1e-4
tol = 0.01

theglobalmin = globalmin
tflag = testflag

myfunction = functions.hartman_6d()
# myfunction = functions.branin()
# myfunction = functions.ackley(3)

b = np.array([[-5,10],[0,15]])
#b = np.array([[0,1],[0,1],[0,1],[0,1],[0,1],[0,1]])
# b = np.array([[-32.768, 32.768],[-32.768, 32.768],[-32.768, 32.768]])
#print("outside b:",b)
#center = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
center = np.array([0.5, 0.5])
# ycenter = cost_func(center)
# print("center:",ycenter)
DIRECTstart_time = time.time()
a1,a2,a3 = DiRect(cost_func, b, max_iters=100, max_evals=10000, max_deep=20000)
DIRECTstop_time = time.time()

print("DIRECT optimum:",a1,  " time: --- %s seconds ---" % (DIRECTstop_time - DIRECTstart_time))
print("x at optimum:",a2)
#print("c:",a3)

lb = [-5, 0]
ub = [10, 15]
#lb = [0,0,0,0,0,0]
#ub = [1,1,1,1,1,1]
# lb = [-32.768, -32.768, -32.768]
# ub = [32.768, 32.768, 32.768]

PSOptimizer = ParticleSwarmOptimizer()
PSOstart_time = time.time()
#xopt, fopt = PSOptimizer.minimize(cost_func, lb, ub, f_ieqcons=None, maxiter=50, swarmsize=20)
xopt, fopt = PSOptimizer.myMinimize(cost_func, lb, ub, f_ieqcons=None, maxiter=1000, swarmsize=2000, minstep=1e-16)
#xopt = np.floor(xopt)
PSOstop_time = time.time()
print("PSO result:",xopt, fopt, " put x into func:", cost_func(xopt), " time: --- %s seconds ---" % (PSOstop_time - PSOstart_time))

# x1 = np.linspace(-32.768, 32.768, 100)
# x2 = np.linspace(-32.768, 32.768, 100)
# Xt = []
# Yt = []
# for i in x1:
#     for j in x2:
#         Yt.append(-myfunction.func([i,j]))
#         Xt.append([i,j])
# y = cost_func(Xt)
#
# plt.plot(Xt,Yt, label='true function')
# plt.show()
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend(loc='best')
# i = 0
# for hist in direct.l_hist:
#     plt.plot(hist[0], hist[1], 'r.')
#     plt.text(hist[0], hist[1]+0.05, i)
#     i += 1
# plt.show()