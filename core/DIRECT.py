''' DIRECT Algorithm
This is a Python version of DIRECT algorithm written by Dan Finkel (definkel@unity.ncsu.edu) (Last Update: 06/21/2004)
Function     : DIRECTAlgo.minimize
Rewritten by : Phuc Luong (pluong@deakin.edu.au)
Created on   : 04/15/2019
Purpose      : Reimplementation of Direct optimization algorithm for Python language.
'''
import math, time
import numpy as np

class DIRECTAlgo:
    def CallObjFcn(self, Problem,x,a,b,impcon,calltype,**args):
        fcn_value = 0
        cons_value = 0
        feas_flag = 0
        # Scale variable back to original space
        point = np.abs(b-a)*(np.array(x))+a
        if calltype == 1:
            fcn_value = Problem(point)
        elif calltype == 2:
            fcn_value = Problem(point)
        return fcn_value, cons_value, feas_flag

    def calc_lbound(self, lengths,fc,hull,szes):
        hull_length = len(hull)
        hull_lengths = [lengths[i] for i in hull]

        lb = [0]*hull_length
        for i in range(0,hull_length):
            X = np.sum(hull_lengths,1) > np.sum(lengths[hull[i]])
            tmp_rects = np.where(X != 0)[0]
            if len(tmp_rects) > 0:
                tmp_f = [fc[hull[tmp]] for tmp in tmp_rects]
                tmp_szes = [szes[hull[tmp]] for tmp in tmp_rects]
                tmp_lbs = [(fc[hull[i]]-tmp_f[j])/(szes[hull[i]]-tmp_szes[j]) for j in range(0, len(tmp_f))]
                lb[i] = np.max(tmp_lbs)
            else:
                lb[i] = -np.inf
        return lb

    def calc_ubound(self, lengths,fc,hull,szes):
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
                ub[i] = np.inf
        return ub

    def find_po(self, fc, lengths, minval, ep, szes):
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
        for i in range(0, tmp_max+1):
            tmp_idx = np.where(sum_lengths == i)[0]
            tmpArr = [fc[tmpIdx] for tmpIdx in tmp_idx]
            # print("tmpArr:",tmpArr)
            if len(tmpArr) == 0:
                tmp_n = np.array([])
            else:
                tmp_n = np.min(tmpArr)
            # print(tmp_n)
            hullidx = np.where(tmpArr==tmp_n)[0]
            # print("hullidx:",hullidx)
            if len(hullidx) > 0:
                for tmpIdx in hullidx:
                    if tmp_idx[tmpIdx] not in hull:
                        hull.append(tmp_idx[tmpIdx])
                j+=1
                tmpCal = np.array([abs(fc[tmpId]-tmp_n) for tmpId in tmp_idx])
                ties = np.where(tmpCal <= 1e-13)[0]
                if len(ties) > 1:
                    mod_ties = np.where([tmp_idx[tmpIdx] != hull[j-1] for tmpIdx in ties])[0]
                    if tmp_idx[ties[mod_ties[0]]] not in hull:
                        hull.append(tmp_idx[ties[mod_ties[0]]])
                    j = len(hull)
        # print("hull:",hull)

        lbound = self.calc_lbound(lengths, fc, hull, szes)
        ubound = self.calc_ubound(lengths, fc, hull, szes)
        lminusb = np.array(lbound) - np.array(ubound)

        maybe_po = np.where(lminusb <= 0)[0]

        hullMaybePo = [hull[i] for i in maybe_po]
        t_len = len(hullMaybePo)
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
        final_pos = [hull[i] for i in maybebo_po]
        szes_finalpos = [szes[i] for i in final_pos]
        return final_pos, szes_finalpos

    def DIRdivide(self, a, b, Problem, index, thirds,p_lengths,p_fc,p_c,p_con, p_feas_flags,p_fcncounter,p_szes,impcons,calltype,**args):
        lengths = p_lengths
        fc = p_fc
        c = p_c
        szes = p_szes
        fcncounter = p_fcncounter
        con = p_con
        feas_flags = p_feas_flags

        #1. Determine which sides are the largest
        li = lengths[index]
        biggy = int(np.min(li))
        ls = np.where(li == biggy)[0]
        lssize = len(ls);
        j = 0;
        #2. Evaluate function in directions of biggest size
        # to determine which direction to make divisions
        oldc = c[index]
        delta = thirds[biggy]
        newc_left = np.copy(oldc)
        newc_right = np.copy(oldc)
        if(lssize > 1):
            newc_left = [np.copy(oldc)]
            newc_right = [np.copy(oldc)]
            for i in range(1, lssize):
                newc_left.append(np.copy(oldc))
                newc_right.append(np.copy(oldc))
        newc_left = np.array(newc_left)
        newc_right = np.array(newc_right)
        f_left = [0.0]*lssize
        f_right = [0.0]*lssize
        for i in range(0,lssize):
            lsi = ls[i]
            if lssize == 1:
                newc_left[lsi] = newc_left[lsi] - delta
                newc_right[lsi] = newc_right[lsi] + delta
                f_left[i], _, _ = self.CallObjFcn(Problem, newc_left, a, b, impcons, calltype)
                f_right[i], _, _ = self.CallObjFcn(Problem, newc_right, a, b, impcons, calltype)
            else:
                newc_left[i][lsi] = newc_left[i][lsi] - delta
                newc_right[i][lsi] = newc_right[i][lsi] + delta
                f_left[i], _, _ = self.CallObjFcn(Problem, newc_left[i], a, b, impcons, calltype)
                f_right[i], _, _ = self.CallObjFcn(Problem, newc_right[i], a, b, impcons, calltype)
            fcncounter+=2
        # print("newc_left:",newc_left)
        # print("newc_right:", newc_right)
        # while(isinstance(f_left[0], list)):
        #     f_left = f_left[0]
        # while (isinstance(f_right[0], list)):
        #     f_right = f_right[0]
        # f_left = np.asarray(f_left)
        # f_right = np.asarray(f_right)

        w = np.minimum(f_left, f_right)
        w = w.tolist()
        # print("w before:", w, " type:", type(w))
        w = [[w[i], ls[i]] for i in range(0, len(ls))]

        # print("w:", w, "f_left:", f_left, "f_right:",f_right)
        # 3. Sort w for division order
        V = np.sort(w,axis=0)
        order = np.argsort(w,axis=0)
        # print("V:",V)
        #print("order:", order+1)
        # print("len order:", len(order))
        # 4. Make divisions in order specified by order
        for i in range(0, len(order)):
            newleftindex = p_fcncounter + 2 * (i);
            newrightindex = p_fcncounter + 2 * (i) + 1;
            # print("new left index:" , newleftindex)
            # print("new right index:", newrightindex)
            if newleftindex >= len(lengths) or newrightindex >= len(lengths):
                isDone = 0
                return lengths, fc, c, con, feas_flags, szes, fcncounter, isDone

            # 4.1 create new rectangles identical to the old one
            oldrect = lengths[index]
            lengths[newleftindex] = oldrect.copy();
            lengths[newrightindex] = oldrect.copy();

            # old, and new rectangles have been sliced in order(i) direction
            lengths[newleftindex,ls[order[i, 0]]] = lengths[index,ls[order[i, 0]]] + 1
            lengths[newrightindex,ls[order[i, 0]]] = lengths[index,ls[order[i, 0]]] + 1
            lengths[index,ls[order[i, 0]]] = lengths[index,ls[order[i, 0]]] + 1

            #print("lengths:",lengths[0:fcncounter + 5])

            # if lssize == 1:
            #     c[newleftindex] = newc_left.copy()
            #     c[newrightindex] = newc_right.copy()
            # else:
                #tmpOrder = order.ravel()
                #c[newleftindex] = newc_left[tmpOrder[i]].copy()
                #c[newrightindex] = newc_right[tmpOrder[i]].copy(
            if lssize == 1:
                c[newleftindex] = newc_left.copy()
                c[newrightindex] = newc_right.copy()
            else:
                c[newleftindex] = newc_left[order[i][0]].copy()
                c[newrightindex] = newc_right[order[i][0]].copy()

            #print("c:", c[0:fcncounter])
            #print("fc:", fc[0:fcncounter])
            # add new values to fc
            # fc[newleftindex] = f_left[order.ravel()[i]]
            # fc[newrightindex] = f_right[order.ravel()[i]]
            fc[newleftindex] = f_left[order[i][0]]
            fc[newrightindex] = f_right[order[i][0]]
            # print("i:",i, " order.ravel()[i]:", order.ravel())
            # print("newleftindex:",newleftindex)
            # print("newrightindex:", newrightindex)
            # print("f_left:",f_left)
            # print("f_right:", f_right)
            # print("inner fc:",fc[0:fcncounter])
            # store sizes of each rectangle
            szes[newleftindex] = 1.0 / 2 * np.linalg.norm(np.power(1.0 / 3 * np.ones((len(lengths[0]))), lengths[newleftindex]))
            szes[newrightindex] = 1.0 / 2 * np.linalg.norm(np.power(1.0 / 3 * np.ones((len(lengths[0]))), lengths[newrightindex]))
        szes[index] = 1.0 / 2 * np.linalg.norm(np.power(1.0 / 3 * np.ones((len(lengths[0]))), lengths[index]))
        isDone = 1

        return lengths,fc,c,con,feas_flags,szes,fcncounter, isDone

    def DIRini(self, Problem,n,a,b,
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
        # calltype = DetermineFcnType(Problem, impcons)
        calltype = 1 # No constraints at all
        l_fc[0], l_con[0], l_feas_flags[0] = self.CallObjFcn(Problem, l_c[0], a, b, impcons, calltype)

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


    def minimize(self, f, bounds, **args):
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
        tol = 0.1
        showits = 1
        impcons = 0
        pert = 1e-6

        theglobalmin = 0.3979;
        tflag = testflag;
        success = 0
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
        self.DIRini(f,n, om_lower, om_upper, lengths, c, fc, con, feas_flags, szes, theglobalmin, maxdeep,tflag,0,0)

        ret_minval = minval
        ret_xatmin = xatmin

        minval = fc[0] + con[0]
        prevMin = minval
        dupMin = 0
        while perror > tol:
            #fc_cons = [fc[i] + con[i] for i in range(0,fcncounter)]
            fc_cons = [fc[i] for i in range(0, fcncounter)]
            szesFC = [szes[i] for i in range(0,fcncounter)]
            inLengths = []

            inLengths = [lengths[tmpIdx] for tmpIdx in range(0,fcncounter)]
            # print("============================itctr:", itctr, "fcncounter:", fcncounter, " minval:", minval, "at x:",xatmin)

            # Create list S of potentially optimal hyper-rectangles
            S = self.find_po(fc_cons, inLengths, minval, ep, szesFC)
            # print("S:",S)
            #%-- Loop through the potentially optimal hrectangles -----------%
            #%-- and divide -------------------------------------------------%
            for i in range(0,len(S[0])):
                lengths, fc, c, con, feas_flags, szes, fcncounter, success = self.DIRdivide(om_lower, om_upper, f, S[0][i], thirds, lengths, fc, c, con, feas_flags, fcncounter, szes, impcons, calltype)
                if success == 0:
                    break
            # print("fc:",fc[0:fcncounter])
            #-- update minval, xatmin - ------------------------------------- %
            if success == 1:
                tmp = []
                for i in range(0, fcncounter):
                    # tmp.append(fc[i] + con[i])
                    tmp.append(fc[i])
                minval = np.min(tmp)
                fminindex = np.argmin(tmp)
                #penminval = minval + con[fminindex]
                xatmin = (om_upper - om_lower) * c[fminindex] + om_lower;

                # --- update return values
                ret_minval = minval
                ret_xatmin = xatmin

            if tflag:
                if theglobalmin != 0:
                    perror = 100*(minval - theglobalmin)/abs(theglobalmin)
                else:
                    perror = 100*minval
            else:
                if itctr >= maxits:
                    #print("Exceeded max iterations. Increase maxits")
                    break
                if fcncounter > maxevals:
                    #print("Exceeded max fcn evals. Increase maxevals")
                    break
                #if np.max(np.where(lengths)) >= maxdeep:
                if np.max(lengths) >= maxdeep:
                    #print("Exceeded Max depth. Increse maxdeep")
                    perror = -1
                    break
            itctr+=1

        foundMin = ret_minval
        foundXatMin = ret_xatmin
        xobs = (om_upper - om_lower) * c + om_lower;
        return foundXatMin, foundMin, xobs, fc