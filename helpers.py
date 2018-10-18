import pickle
import os
import numpy as np
import pandas as pd
from constants_v1 import *


# Extract observation dataframe
def get_obs(name):
    obs_path = './observations/' + name + ".tss"
    return pd.read_table(obs_path)


# Function to add column ID, with first letter capitalized to be able to merge obs vs. sim.
def newlabel(row, plot):
    return plot.capitalize() + '-' + str(int(row['Jdays']))


def get_results_thin(new=False):
    """
    :param new: If True, will recreate new Map, else, just open existing one.
    :return: If False, dictionary will be unpickled and opened.
    """
    try:
        all_results = pickle.load(open("resultsmap.p", "rb"))
    except IOError:
        all_results = dict()

    if new:
        ups = get_upper_v1(bounds_v1)  # Upper bounds to re-scale input matrix
        for pc in computers:  # Multiple computers, with specific names running models
            for model in models:  # Two types: static (fix) and dynamic (var)
                RUNS = True  # Keeps track of how many sets have been tested on each machine
                version = 1  # Counter for 'experiments' made, eg. var1, var2, etc...
                while RUNS:  # Will turn to False, when no more experiments of this model version
                    # Versions (multiple per computer)
                    path = 'LHS_' + pc + model + str(version) + '/'  # Simulation path
                    print(path)
                    if os.path.exists(path):
                        matrix = np.loadtxt(path + "/lhs_vectors.txt")  # Sets for this experiment
                        sets = matrix.shape[0]  # No. of rows/sets in experiment/matrix
                        for row in range(sets):
                            if row == 0:
                                print('extracting sets for path: ', path)
                            # Folders
                            path = 'LHS_' + pc + model + str(version) + '/' + str(row + 1) + '/'
                            if os.path.exists(path):
                                set_results = dict()
                                vector = matrix[row] * ups  # Rescaled vector
                                for param in params:  # Parameter name tested: [x1, x2, x3, ..]
                                    set_results[param] = [{'val': vector[params.index(param)]}]
                                    for measure in measures:  # Assign metrics' dict to each parameter by data-level
                                        set_results[param].append({measure: get_likelihood(path, measure)})

                                # Append each set's results to all_results
                                all_results[path] = set_results

                        version += 1
                    else:
                        RUNS = False
    else:
        return all_results

    if all_results:
        pickle.dump(all_results, open("resultsmap.p", "wb"))


def get_likelihood(sim_path, measure, only_KGE=False):
    if measure == 'CONC':
        col = 'ug.g'
    elif measure == 'd13C':
        col = measure
    elif measure == 'theta':
        col = measure
    elif measure == 'Q_out':
        col = 'Qm3'
    elif measure == 'CONC_out':
        col = 'ug.L'
    elif measure == 'd13C_out':
        col = 'd13C'
    elif measure == 'LDS_out':
        col = 'smloads.g'

    metric_dict = dict()

    if '_out' in measure:
        df = get_dataframe(sim_path, measure, 'outlet')
        metric_dict['KGE'] = get_kge(df, col)
        if not only_KGE:
            metric_dict['NSE'] = get_nash(df, col)
            metric_dict['MAE'] = get_mae(df, col)
            metric_dict['BIAS'] = get_bias(df, col)
    else:
        df1 = get_dataframe(sim_path, measure, 'bulk')
        df2 = get_dataframe(sim_path, measure, 'transects')
        df3 = get_dataframe(sim_path, measure, 'detailed')
        df4 = pd.concat([df2, df3])  # Transect + detailed = Tot

        metric_dict['KGE'] = dict([('blk', get_kge(df1, col)),
                                   ('tra', get_kge(df2, col)),
                                   ('det', get_kge(df3, col)),
                                   ('tot', get_kge(df4, col))
                                   ])

        if not only_KGE:
            metric_dict['NSE'] = dict([('blk', get_nash(df1, col)),
                                       ('tra', get_nash(df2, col)),
                                       ('det', get_nash(df3, col)),
                                       ('tot', get_nash(df4, col))
                                       ])

            metric_dict['MAE'] = dict([('blk', get_mae(df1, col)),
                                       ('tra', get_mae(df2, col)),
                                       ('det', get_mae(df3, col)),
                                       ('tot', get_mae(df4, col))
                                       ])
            metric_dict['BIAS'] = dict([('blk', get_bias(df1, col)),
                                        ('tra', get_bias(df2, col)),
                                        ('det', get_bias(df3, col)),
                                        ('tot', get_bias(df4, col))
                                        ])

    return metric_dict


def get_nash_log(df1, col):
    """
    Simulated df, has to have:
        - Simulated col name as 'SIM'

    :param df1: merged obs vs sim dataframe
    :param col: observation column name
    :return: nse value
    """

    # Nash
    mean = df1[col].mean()
    # Diff sim vs. obs
    dfn = df1.assign(diff_sim=lambda row: (row['SIM'] - row[col]) ** 2)
    err_sim = dfn['diff_sim'].sum()
    # Variance
    dfn = dfn.assign(diff_obs=lambda row: (row[col] - mean) ** 2)
    err_obs = dfn['diff_obs'].sum()
    error = err_sim / err_obs
    nash = 1 - error

    if 'ug' in col:  # Log only for concentrations
        lnmean = np.log(df1[col]).mean()
        # Log Diff sim vs. obs
        dfn = dfn.assign(lndiff_sim=lambda row: (np.log(row['SIM']) - np.log(row[col])) ** 2)
        err_lnsim = dfn['lndiff_sim'].sum()
        # Log variance
        dfn = dfn.assign(lndiff_obs=lambda row: (np.log(row[col]) - lnmean) ** 2)
        err_lnobs = dfn['lndiff_obs'].sum()
        error += err_lnsim / err_lnobs
        error *= 0.5
        nash = 1 - error
    return nash


def get_nash(df1, col):
    """
    Simulated df, has to have:
        - Simulated col name as 'SIM'

    :param df1: merged obs vs sim dataframe
    :param col: observation column name
    :return: nse value
    """

    # Nash
    mean = df1[col].mean()
    # Diff sim vs. obs
    dfn = df1.assign(diff_sim=lambda row: (row['SIM'] - row[col]) ** 2)
    err_sim = dfn['diff_sim'].sum()
    # Variance
    dfn = dfn.assign(diff_obs=lambda row: (row[col] - mean) ** 2)
    err_obs = dfn['diff_obs'].sum()
    error = err_sim / err_obs
    nash = 1 - error

    return nash


def get_nash_lutz(df1, col, weigh=False):
    """
    Simulated df, has to have:
        - Simulated col name as 'SIM'

    :param df1: merged obs vs sim dataframe
    :param col: observation column name
    :return: nse value
    """

    # Nash
    mean = df1[col].mean()
    # Diff sim vs. obs
    if weigh:
        dfn = df1.assign(diff_sim=lambda row: (row['weigh'] * (row['SIM'] - row[col]) ** 2))
    else:
        dfn = df1.assign(diff_sim=lambda row: (row['SIM'] - row[col]) ** 2)
    err_sim = dfn['diff_sim'].sum()
    # Variance
    if weigh:
        dfn = dfn.assign(diff_obs=lambda row: (row['weigh'] * (row[col] - mean) ** 2))
    else:
        dfn = dfn.assign(diff_obs=lambda row: (row[col] - mean) ** 2)
    err_obs = dfn['diff_obs'].sum()
    error = err_sim / err_obs
    nash = 1 - error

    if 'ug' in col:  # Log only for concentrations
        lnmean = np.log(df1[col]).mean()
        # Log Diff sim vs. obs
        if weigh:
            dfn = dfn.assign(lndiff_sim=lambda row: (row['weigh'] * (np.log(row['SIM']) - np.log(row[col])) ** 2))
        else:
            dfn = dfn.assign(lndiff_sim=lambda row: (np.log(row['SIM']) - np.log(row[col])) ** 2)
        err_lnsim = dfn['lndiff_sim'].sum()
        # Log variance
        if weigh:
            dfn = dfn.assign(lndiff_obs=lambda row: (row['weigh'] * (np.log(row[col]) - lnmean) ** 2))
        else:
            dfn = dfn.assign(lndiff_obs=lambda row: (np.log(row[col]) - lnmean) ** 2)
        err_lnobs = dfn['lndiff_obs'].sum()
        error += err_lnsim / err_lnobs
        error *= 0.5
        nash = 1 - error
    return nash


def get_kge(df, col):
    if df is None:
        return None
    # KGE, @Gupta2009
    # Linear correlation, alpha and beta
    r = np.corrcoef(df[col], df['SIM'])[1, 0]
    alpha = np.std(df['SIM']) / np.std(df[col])
    beta = np.mean(df['SIM']) / np.mean(df[col])
    kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    return kge


def get_mae(df, col):
    # Mean absolute error
    mae = np.sum(np.absolute(df['SIM'] - df[col])) / np.size(df[col])
    return mae


def get_bias(df, col):
    bias = 100 * np.sum(df['SIM'] - df[col]) / np.sum(df[col])
    return bias


def get_dataframe(sim_path, measure, level):
    tran = False
    det = False
    blk = False
    tot = False
    out = False
    if level == 'complete':
        tran = True
        det = True
        tot = True
    elif level == 'transects':
        tran = True
    elif level == 'detailed':
        det = True
        if measure == 'theta':
            return None
    elif level == 'bulk':
        blk = True
    elif level == "outlet":
        out = True

    if tran:
        # Composites
        matches = []
        name_obs = measure.lower() + '_comp_cal'
        for transect in transects:
            # Simulated
            if measure == 'theta':
                filename = "resW_z0_theta_" + transect + ".tss"
            else:
                filename = "resM_" + transect + measure + ".tss"

            sim = pd.read_table(sim_path + filename,
                                skiprows=4, delim_whitespace=True,
                                names=['Jdays', 'SIM'],
                                header=None)
            # create new ID column
            letter = transect[0]
            sim['IDcal'] = sim.apply(lambda row: newlabel(row, letter), axis=1)

            # Observed
            obs = get_obs(name_obs)

            # Merge
            match = pd.merge(obs, sim, how='inner', on='IDcal')
            matches.append(match)

        df_comp = pd.concat(matches)
        if not det:
            return df_comp

    if det and measure is not 'theta':
        # Detailed
        matches = []
        name_obs = measure.lower() + '_det_cal'
        for plot in plots:
            # Simulated
            filename = "resM_" + plot + measure + ".tss"
            sim = pd.read_table(sim_path + filename,
                                skiprows=4, delim_whitespace=True,
                                names=['Jdays', 'SIM'],
                                header=None)
            # create new ID column
            letter = plot
            sim['IDcal'] = sim.apply(lambda row: newlabel(row, letter), axis=1)

            # Observed
            obs = get_obs(name_obs)
            # Merge
            match = pd.merge(obs, sim, how='inner', on='IDcal')
            matches.append(match)

        df_det = pd.concat(matches)
        if not tran:
            return df_det

    if tot and measure is not 'theta':
        return pd.concat([df_det, df_comp])

    if blk:
        masses = []
        thetas = []

        # Observations (to merge with later)
        name_obs = measure.lower() + '_bulk_cal'
        obs = get_obs(name_obs)

        # Bulk...anizing Simulated values
        p_b = get_obs("p_bAve")
        # Get sim conc, convert mass, ug/g -> ug
        for transect in transects:
            if measure == 'theta':
                filename = "resW_z0_theta_" + transect + ".tss"
                tr = transect[0] + 'SimTheta'
                sim = pd.read_table(sim_path + filename,
                                    skiprows=4, delim_whitespace=True,
                                    names=['Jdays', tr],
                                    header=None)
                thetas.append(sim)
            else:
                # Append masses and conc.
                filename = "resM_" + transect + 'CONC_real' + ".tss"

                tr = transect[0] + 'SimCon'
                sim = pd.read_table(sim_path + filename,
                                    skiprows=4, delim_whitespace=True,
                                    names=['Jdays', tr],
                                    header=None)

                colname = transect[0:3] + "Mass"
                sim[colname] = sim[tr] * p_b['pbAve'] * 4.0 * 10.0 * 10 ** 3
                masses.append(sim)

                if measure == 'd13C':
                    # Append deltas
                    filename = "resM_" + transect + 'd13C_real' + ".tss"
                    tr = transect[0] + 'Simd13C'
                    sim = pd.read_table(sim_path + filename,
                                        skiprows=4, delim_whitespace=True,
                                        names=['Jdays', tr],
                                        header=None)
                    masses.append(sim)

        if measure == 'theta':
            nThetas = reduce(lambda x, y: pd.merge(x, y, on='Jdays'), thetas)
        else:
            # Merge all transects
            nMasses = reduce(lambda x, y: pd.merge(x, y, on='Jdays'), masses)

        # Bulk concentration
        if measure == 'CONC':
            nMasses['SIM'] = (nMasses['nSimCon'] * nMasses['norMass'] +
                              nMasses['vSimCon'] * nMasses['valMass'] +
                              nMasses['sSimCon'] * nMasses['souMass']
                              ) / (nMasses['norMass'] + nMasses['valMass'] + nMasses['souMass'])
            # Merge Observed
            match = pd.merge(obs, nMasses, how='inner', on='Jdays')
            return match[['Jdays', 'IDcal', 'ug.g', 'SIM']]
        # Bulk deltas
        if measure == 'd13C':
            nMasses['SIM'] = (nMasses['nSimd13C'] * nMasses['norMass'] +
                              nMasses['vSimd13C'] * nMasses['valMass'] +
                              nMasses['sSimd13C'] * nMasses['souMass']
                              ) / (nMasses['norMass'] + nMasses['valMass'] + nMasses['souMass'])

            # Merge Observed
            match = pd.merge(obs, nMasses, how='inner', on='Jdays')
            return match[['Jdays', 'IDcal', 'd13C', 'SIM']]
        # Bulk thetas
        if measure == 'theta':
            nThetas['SIM'] = (nThetas['nSimTheta'] +
                              nThetas['vSimTheta'] +
                              nThetas['sSimTheta'])/3.
            match = pd.merge(obs, nThetas, how='inner', on='Jdays')
            return match[['Jdays', 'IDcal', 'theta', 'SIM']]

    if out:
        # Observed
        name_obs = measure.lower() + '_cal'
        obs = get_obs(name_obs)
        if measure == 'Q_out':
            # Simulated
            filename = "resW_accQ_m3.tss"
            sim = pd.read_table(sim_path + filename,
                                skiprows=4, delim_whitespace=True,
                                names=['Jdays', 'SIM'],
                                header=None)
        elif measure == 'CONC_out':
            # Simulated
            filename = "resM_oCONC_ugL.tss"
            sim = pd.read_table(sim_path + filename,
                                skiprows=4, delim_whitespace=True,
                                names=['Jdays', 'SIM'],
                                header=None)

        elif measure == 'd13C_out':
            # Simulated
            filename = "resM_outISO_d13C.tss"
            sim = pd.read_table(sim_path + filename,
                                skiprows=4, delim_whitespace=True,
                                names=['Jdays', 'SIM'],
                                header=None)
        elif measure == 'LDS_out':
            # Simulated
            filename = "resM_EXP_light_g.tss"
            sim = pd.read_table(sim_path + filename,
                                skiprows=4, delim_whitespace=True,
                                names=['Jdays', 'SIM'],
                                header=None)
        else:
            print("No appropriate measure selected for the outlet!")
            return
        # Merge
        match = pd.merge(obs, sim, how='inner', on='Jdays')
        return match


def get_dff_thin(metric, new=False):
    if new:
        get_results_thin(new=new)
        new = False

    res = get_results_thin(new=new)
    sets = res.values()
    set_names = res.keys()
    df_array = []
    for setx in range(len(sets)):
        param_vals = sets[setx].values()
        param_names = sets[setx].keys()

        df = pd.DataFrame(columns=('x', 'Param', metric, 'Measure', 'Level', 'Set'))

        for param in range(len(param_vals)):

            # Same param Xi val for all results of this set
            val = param_vals[param][0]['val']

            # Soils - Conc
            likes = param_vals[param][1]['CONC'][metric].values()
            levels = param_vals[param][1]['CONC'][metric].keys()

            for like in range(len(likes)):
                df = df.append({'x': val,
                                'Param': param_names[param],
                                metric: likes[like],
                                'Measure': 'CONC',
                                'Level': levels[like],
                                'Set': set_names[setx]
                                }, ignore_index=True)

            # Soils - d13C
            likes = param_vals[param][2]['d13C'][metric].values()
            levels = param_vals[param][2]['d13C'][metric].keys()

            for like in range(len(likes)):
                df = df.append({'x': val,
                                'Param': param_names[param],
                                metric: likes[like],
                                'Measure': 'd13C',
                                'Level': levels[like],
                                'Set': set_names[setx]
                                }, ignore_index=True)

            # Outlet (level == Outlet)
            outlet_vars = ['Q_out', 'CONC_out', 'LDS_out', 'd13C_out']
            indx = 3
            for var in outlet_vars:
                metric_val = param_vals[param][indx][var][metric]
                df = df.append({'x': val,
                                'Param': param_names[param],
                                metric: metric_val,
                                'Measure': var,
                                'Level': 'out',
                                'Set': set_names[setx]
                                }, ignore_index=True)
                indx += 1

        df_array.append(df)

    dff = pd.concat(df_array)
    return dff


def get_dff_fat(new=False, copy=False, version=None):
    if new:
        get_results_free(new=new, copy=copy)
        new = False

    res = get_results_free(new=new, copy=copy, version=version)
    return pd.DataFrame.from_dict(res, orient='index')


# Old version for LHS sampling, includes all metrics
def get_results_fat(metrics, levels, new=False, gen=None, params=None, only_KGE=False):
    """
    Generates a dictionary that easy easy to convert into a data-frame via:
        df = pd.DataFrame.from_dict(all_results, orient='index')

    :param new: If True, will recreate new Map, else, just open existing one.
    :return: If False, dictionary will be unpickled and opened.
    """
    try:
        filename = "results" + gen + ".p"
        all_results = pickle.load(open(filename, "rb"))
    except IOError:
        all_results = dict()

    if new:
        for pc in computers:  # Multiple computers, with specific names running models
            for model in models:  # Two types: static (fix) and dynamic (var)
                VERS = True  # Keeps track of how many versions have been tested on each machine
                version = 1
                print("Using bounds set 1")
                ups = get_upper(bds[0])  # Upper bounds to re-scale input matrix
                failed = 0  # Counter keeping track of how many failed paths per machine
                while VERS:  # Will turn to False, when no more experiments of this model version
                    # Versions (multiple per computer)

                    if version > 60:  # v61+, (Gen10 and Gen 11)
                        print("Using bounds set 15")
                        ups = get_upper(bds[14])  #
                    elif version > 54:  # v55+,
                        print("Using bounds set 14")
                        ups = get_upper(bds[13])  #
                    elif version > 49:  # v50+,
                        print("Using bounds set 13")
                        ups = get_upper(bds[12])  #
                    elif version > 43:  # v44+ to 49
                        print("Using bounds set 12")
                        ups = get_upper(bds[11])  #
                    elif version > 35:  # v36+,
                        print("Using bounds set 11")
                        ups = get_upper(bds[10])  #
                    elif version > 31:  # v32+,
                        # print("Using bounds set 10")
                        ups = get_upper(bds[9])  #
                    elif version > 29:  # v30+,
                        # v30+ <- increased volat. days (Gen5)
                        # print("Using bounds set 9")
                        ups = get_upper(bds[8])  #
                    elif version > 23:  # v24+,
                        # print("Using bounds set 8")
                        ups = get_upper(bds[7])  #
                    elif version > 16:  # v17+,
                        # print("Using bounds set 7")
                        ups = get_upper(bds[6])  #
                    elif version > 8:  # v9+,
                        # v10 <- New generation second application omitted.
                        # v15 <- Lower dosage on Burger and Kopp
                        # print("Using bounds set 6")
                        ups = get_upper(bds[5])  #
                    elif version > 6:  # v7, v8
                        # print("Using bounds set 5")
                        ups = get_upper(bds[4])  #
                    elif version > 5:  # v6
                        # print("Using bounds set 4")
                        ups = get_upper(bds[3])  #
                    elif version > 4:  # v5
                        # print("Using bounds set 3")
                        ups = get_upper(bds[2])  # All versions >= 5 have new Hydro bnds
                    elif version > 2:  # v3, v4
                        # print("Using bounds set 2")
                        ups = get_upper(bds[1])  # All versions >= 3 have new bnds

                    read_path = gen + '/LHS_' + pc + model + str(version) + '/'
                    print(read_path)

                    if os.path.exists(read_path):

                        matrix = np.loadtxt(read_path + "/lhs_vectors.txt")  # Sets for this experiment
                        sets = matrix.shape[0]  # No. of rows/sets in experiment/matrix
                        for row in range(sets):
                            if row == 0:
                                print('extracting sets for path: ', read_path)
                            # Folders
                            read_path = gen + '/LHS_' + pc + model + str(version) + '/' + str(row + 1) + '/'
                            if os.path.exists(read_path):
                                write_path = 'LHS_' + pc + model + str(version) + '/' + str(row + 1) + '/'
                                if row % 10 == 0:
                                    print('...row: ', str(row + 1))

                                vector = matrix[row] * ups  # Rescaled vector

                                # Check if sample has been already evaluated
                                try:
                                    all_results[write_path] in all_results.keys()
                                    # print("wait, already did this sample, skipping: ", read_path)
                                    continue
                                except KeyError:
                                    pass

                                # Start of data set building:
                                all_results[write_path] = dict()
                                all_results[write_path]['Model'] = model  # Var or Fix
                                for param in params:  # Parameter name tested: [x1, x2, x3, ..]
                                    all_results[write_path][param] = vector[params.index(param)]
                                    for metric in metrics:  # KGE, NSE, etc..
                                        for outm in outMeasures:
                                            out_metric = get_likelihood(read_path, outm, only_KGE=only_KGE)
                                            m = metric + "-" + outm
                                            all_results[write_path][m] = out_metric[metric]
                                        for soilm in soilMeasures:
                                            soil_metric = get_likelihood(read_path, soilm, only_KGE=only_KGE)
                                            for level in levels:
                                                m = metric + "-" + soilm + "-" + level
                                                all_results[write_path][m] = soil_metric[metric][level]
                        version += 1

                    else:
                        version += 1
                        failed += 1
                        # print("Version path attempted, no.: " + str(failed) + " Moving to version: " + str(version))

                    if failed > 70:  # No more
                        print("No more versions of this model type")
                        VERS = False
        # Save
        pickle.dump(all_results, open(filename, "wb"))
    else:
        return all_results


# New version, just with KGE metrics
def get_results_free(metrics, levels, new=False, gen=None, params=None, only_KGE=True):
    """
        Generates a dictionary that easy easy to convert into a data-frame via:
            df = pd.DataFrame.from_dict(all_results, orient='index')

        :param new: If True, will recreate new Map, else, just open existing one.
        :return: If False, dictionary will be unpickled and opened.
        """
    try:
        filename = "results" + gen + ".p"
        all_results = pickle.load(open(filename, "rb"))
    except IOError:
        all_results = dict()

    if new:
        for pc in computers:  # Multiple computers, with specific names running models
            for model in models:  # Two types: static (fix) and dynamic (var)
                VERS = True  # Keeps track of how many versions have been tested on each machine
                version = 1
                print("Using bounds set 1")
                ups = get_upper(bds[0])  # Upper bounds to re-scale input matrix
                failed = 0  # Counter keeping track of how many failed paths per machine
                while VERS:  # Will turn to False, when no more experiments of this model version
                    # Versions (multiple per computer)

                    if version > 60:  # v61+, (Gen10 and Gen 11)
                        print("Using bounds set 15")
                        ups = get_upper(bds[14])  #
                    elif version > 54:  # v55+,
                        print("Using bounds set 14")
                        ups = get_upper(bds[13])  #
                    elif version > 49:  # v50+,
                        print("Using bounds set 13")
                        ups = get_upper(bds[12])  #
                    elif version > 43:  # v44+ to 49
                        print("Using bounds set 12")
                        ups = get_upper(bds[11])  #
                    elif version > 35:  # v36+,
                        print("Using bounds set 11")
                        ups = get_upper(bds[10])  #
                    elif version > 31:  # v32+,
                        # print("Using bounds set 10")
                        ups = get_upper(bds[9])  #
                    elif version > 29:  # v30+,
                        # v30+ <- increased volat. days (Gen5)
                        # print("Using bounds set 9")
                        ups = get_upper(bds[8])  #
                    elif version > 23:  # v24+,
                        # print("Using bounds set 8")
                        ups = get_upper(bds[7])  #
                    elif version > 16:  # v17+,
                        # print("Using bounds set 7")
                        ups = get_upper(bds[6])  #
                    elif version > 8:  # v9+,
                        # v10 <- New generation second application omitted.
                        # v15 <- Lower dosage on Burger and Kopp
                        # print("Using bounds set 6")
                        ups = get_upper(bds[5])  #
                    elif version > 6:  # v7, v8
                        # print("Using bounds set 5")
                        ups = get_upper(bds[4])  #
                    elif version > 5:  # v6
                        # print("Using bounds set 4")
                        ups = get_upper(bds[3])  #
                    elif version > 4:  # v5
                        # print("Using bounds set 3")
                        ups = get_upper(bds[2])  # All versions >= 5 have new Hydro bnds
                    elif version > 2:  # v3, v4
                        # print("Using bounds set 2")
                        ups = get_upper(bds[1])  # All versions >= 3 have new bnds

                    read_path = gen + '/LHS_' + pc + model + str(version) + '/'
                    print(read_path)

                    if os.path.exists(read_path):

                        matrix = np.loadtxt(read_path + "/lhs_vectors.txt")  # Sets for this experiment
                        sets = matrix.shape[0]  # No. of rows/sets in experiment/matrix
                        for row in range(sets):
                            if row == 0:
                                print('extracting sets for path: ', read_path)
                            # Folders
                            read_path = gen + '/LHS_' + pc + model + str(version) + '/' + str(row + 1) + '/'
                            if os.path.exists(read_path):
                                write_path = 'LHS_' + pc + model + str(version) + '/' + str(row + 1) + '/'
                                if row % 10 == 0:
                                    print('...row: ', str(row + 1))

                                vector = matrix[row] * ups  # Rescaled vector

                                # Check if sample has been already evaluated
                                try:
                                    all_results[write_path] in all_results.keys()
                                    # print("wait, already did this sample, skipping: ", read_path)
                                    continue
                                except KeyError:
                                    pass

                                # Start of data set building:
                                all_results[write_path] = dict()
                                all_results[write_path]['Model'] = model  # Var or Fix
                                for param in params:  # Parameter name tested: [x1, x2, x3, ..]
                                    all_results[write_path][param] = vector[params.index(param)]
                                    for metric in metrics:
                                        for outm in outMeasures:
                                            out_metric = get_likelihood(read_path, outm, only_KGE=only_KGE)
                                            m = metric + "-" + outm
                                            all_results[write_path][m] = out_metric[metric]
                                        for soilm in soilMeasures:
                                            # Dictionary with metrics
                                            soil_metric = get_likelihood(read_path, soilm, only_KGE=only_KGE)
                                            for level in levels:
                                                m = metric + "-" + soilm + "-" + level  # e.g., KGE-CONC-tra, ...
                                                all_results[write_path][m] = soil_metric[metric][level]
                        version += 1

                    else:
                        version += 1
                        failed += 1
                        # print("Version path attempted, no.: " + str(failed) + " Moving to version: " + str(version))

                    if failed > 70:  # No more
                        print("No more versions of this model type")
                        VERS = False
        # Save
        pickle.dump(all_results, open(filename, "wb"))
    else:
        return all_results


# ATT!! ->  Gen ???

# Gen1
# get_results_free(new=True, gen="Gen1")

# Gen 2
# get_results_free(new=True, gen="Gen2")

# Gen 3
# get_results_free(new=True, gen="Gen3")

# Gen 4
# get_results_free(new=True, gen="Gen4")

# Gen 5
# get_results_free(new=True, gen="Gen5")

# Gen 6
# get_results_free(new=True, gen="Gen6")

# Gen 7
# get_results_free(new=True, gen="Gen7")

# Gen 8
# get_results_free(new=True, gen="Gen8")

# Gen 9
# get_results_free(new=True, gen="Gen9", params=params_v2)

# Gen 10
# get_results_free(new=True, gen="Gen10", params=params_v2)
get_results_free(metrics_v2, levels_v2, new=True, gen="Gen10_test", params=params_v2)

# Gen 11
# get_results_free(new=True, gen="Gen11", params=params_v2)
