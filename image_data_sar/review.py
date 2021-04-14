import re
from itertools import count
import pickle
import gzip
import numpy as np
import astropy.units as u

from cxotime import CxoTime
from starcheck.parser import get_cat
from Chandra.Maneuver import duration


def get_options():
    import argparse
    parser = argparse.ArgumentParser(
        description="Image data SAR tool")
    parser.add_argument('pkl',
                        type=str,
                        help="Proseco pkl")
    parser.add_argument('starcheck',
                        type=str,
                        help="Starcheck txt file")
    parser.add_argument('--sched-stop',
                        type=str,
                        help="Schedule stop time")
    args = parser.parse_args()
    return args


def custom_read_starcheck(starcheck_file):
    sc_text = open(starcheck_file, 'r').read()
    chunks = re.split(r"={20,}\s?\n?\n", sc_text)
    catalogs = []
    for chunk, idx in zip(chunks, count()):
        obs = get_cat(chunk)
        if obs:
            obs['orig_text'] = chunk
            catalogs.append(obs)
    return catalogs


def starcheck_for_obsid(obsid, sc):
    for cat in sc:
        if cat['obsid'] == int(obsid):
            return cat


def obs1_setup_ok(sc_cat):
    imgsz = np.array([row['sz'] for row in sc_cat['catalog']])
    return ((sc_cat['obs']['dither_y_amp'] == 16.0)
            & (sc_cat['obs']['dither_z_amp'] == 16.0)
            & (sc_cat['obs']['dither_y_period'] == 2000.0)
            & (sc_cat['obs']['dither_z_period'] == 1414.8)
            & (np.all(imgsz == '8x8')))


def obs2_setup_ok(sc_cat):
    imgsz = np.array([row['sz'] for row in sc_cat['catalog'] if row['type'] in ['GUI', 'BOT']])
    return ((sc_cat['obs']['dither_y_amp'] == 8.0)
            & (sc_cat['obs']['dither_z_amp'] == 8.0)
            & (sc_cat['obs']['dither_y_period'] == 1000.0)
            & (sc_cat['obs']['dither_z_period'] == 707.1)
            & (np.all(imgsz == '6x6')))


def obs3_setup_ok(sc_cat):
    imgsz = np.array([row['sz'] for row in sc_cat['catalog']])
    return ((sc_cat['obs']['dither_y_amp'] == 8.0)
            & (sc_cat['obs']['dither_z_amp'] == 8.0)
            & (sc_cat['obs']['dither_y_period'] == 1000.0)
            & (sc_cat['obs']['dither_z_period'] == 707.1)
            & (np.all(imgsz == '8x8')))


def has_sc_critical(obs_text, ok6x6=False):
    warnlines = [w for w in obs_text.split("\n")
                 if re.compile(r"^\>\>\s+CRITICAL.*").match(w)]
    if ok6x6:
        warnlines = [w for w in warnlines if '6x6 Should be 8x8' not in w]
    return len(warnlines) > 0


def has_proseco_critical(aca):
    acar = aca.get_review_table()
    acar.run_aca_review()
    return len(acar.messages == 'critical') > 0


def get_durations(aca_arr):
    durations = []
    for aca, next_aca in zip(aca_arr[:-1], aca_arr[1:]):
        if isinstance(next_aca, dict):
            dwell_end = next_aca['sched_stop']
        else:
            man_dur = duration(aca.att, next_aca.att)
            dwell_end = CxoTime(next_aca.date) - man_dur * u.s
        durations.append(dwell_end - CxoTime(aca.date))
    return durations


def check_duration(obsid, obsids, durations):
    idx = np.flatnonzero(obsids == obsid)[0]
    return durations[idx] > 4000 * u.s


def same_guide_stars(obs1, obs2, obs3, acas):
    gs1 = list(acas[obs1].guides['id'])
    gs2 = list(acas[obs2].guides['id'])
    gs3 = list(acas[obs3].guides['id'])
    return set(gs1) == set(gs2) == set(gs3)


def same_attitude(obs1, obs2, obs3, acas):
    att1 = acas[obs1].call_args['att']
    att2 = acas[obs2].call_args['att']
    att3 = acas[obs3].call_args['att']
    dq2 = att1.dq(att2)
    dq3 = att1.dq(att3)
    return ((np.abs(dq2.pitch * 3600) < 1)
            and (np.abs(dq2.yaw * 3600) < 1)
            and (np.abs(dq3.pitch * 3600) < 1)
            and (np.abs(dq3.yaw * 3600) < 1)
            and np.allclose([att2.roll, att3.roll], att1.roll, atol=0.05, rtol=0))


def pok(ok):
    return 'OK' if ok else 'Not OK'


def main():
    opt = get_options()
    sc = custom_read_starcheck(opt.starcheck)

    acas = pickle.load(gzip.open(opt.pkl))
    aca_arr = sorted(acas.values(), key=lambda aca: aca.date)
    obsids = np.array([aca.obsid for aca in aca_arr])

    er_cands = [aca.obsid for aca in aca_arr
                if aca.obsid > 39000 and np.abs(aca.call_args['dither_guide'][0] - 16) < .1]

    for obsid in er_cands:

        idx = np.flatnonzero(obsids == obsid)[0]
        obs1 = obsid
        obs2 = obsids[idx + 1]
        obs3 = obsids[idx + 2]
        if idx + 3 == len(obsids):
            if opt.sched_stop is None:
                raise ValueError(f"Need --sched-stop to check duration of {obs3}")
            else:
                aca_arr.append({'sched_stop': CxoTime(opt.sched_stop)})

        cat1 = starcheck_for_obsid(obs1, sc)
        cat2 = starcheck_for_obsid(obs2, sc)
        cat3 = starcheck_for_obsid(obs3, sc)

        check1 = ((not has_sc_critical(cat1['orig_text']))
                  and (not has_sc_critical(cat2['orig_text'], ok6x6=True))
                  and (not has_sc_critical(cat3['orig_text']))
                  and (not has_proseco_critical(acas[obs1]))
                  and (not has_proseco_critical(acas[obs2]))
                  and (not has_proseco_critical(acas[obs3])))

        print(f"For SAR activity during obsids {obs1, obs2, obs3}")
        print(f"[{pok(check1)}] - Each obs meets normal ACA review requirements (except 6x6 in ER)")

        check2 = obs1_setup_ok(cat1)

        print(f"[{pok(check2)}] - First obs has 16x16 dither with period=2000.0, 1414.8, ",
              "and image size=8x8 pixels")

        check3 = obs2_setup_ok(cat2)

        print(f"[{pok(check3)}] - Second obs has 8x8 dither with period=1000.0, 707.1, ",
              "and image size=6x6 pixels on guide stars")

        check4 = obs3_setup_ok(cat3)

        print(f"[{pok(check4)}] - Third obs has 8x8 dither with period=1000.0, 707.1, ",
              "and image size=8x8 pixels")

        durations = get_durations(aca_arr)
        check5 = (check_duration(obs1, obsids, durations)
                  and check_duration(obs2, obsids, durations)
                  and check_duration(obs3, obsids, durations))

        print(f"[{pok(check5)}] - Each obs at least 4 ks")

        check6 = same_guide_stars(obs1, obs2, obs3, acas)
        print(f"[{pok(check6)}] - All guide stars are the same")

        check7 = same_attitude(obs1, obs2, obs3, acas)
        print(f"[{pok(check7)}] - All obs are at the same attitude")

        print("")
        print("Compare included stars to previously provided list by hand:")
        ids = acas[obs1].call_args.get('include_ids_guide')
        if len(ids) > 0:
            ids = list(np.array(ids).astype(int))
        print(f"include_ids_guide stars are {ids}")


if __name__ == '__main__':
    main()
