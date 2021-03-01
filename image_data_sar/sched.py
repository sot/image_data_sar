import re
import pickle
import gzip
from jinja2 import Template
from pathlib import Path
import traceback

import sparkles
import chandra_aca
from sparkles.core import ACAReviewTable, get_summary_text, get_acas_from_pickle
import proseco
from Chandra.Time import DateTime
from Chandra.Maneuver import duration

MIN_DWELL = 13000
FILEDIR = Path(__file__).parent
SPARKLES_DIR = Path(sparkles.core.__file__).parent


def get_options():
    import argparse
    parser = argparse.ArgumentParser(
        description="Image data SAR tool")
    parser.add_argument('load_name',
                        type=str,
                        help="Pickle")
    args = parser.parse_args()
    return args


def _run_aca_review(load_name=None, *, acars=None, make_html=True, report_dir=None,
                    report_level='none', roll_level='none', roll_args=None,
                    loud=False, obsids=None, open_html=False, context=None):

    if acars is None:
        acars = get_acas_from_pickle(load_name, loud)

    if obsids:
        acars = [aca for aca in acars if aca.obsid in obsids]

    if not acars:
        raise ValueError('no catalogs founds (check obsid filtering?)')

    if roll_args is None:
        roll_args = {}

    # Make output directory if needed
    if make_html:
        # Generate outdir from load_name if necessary
        if report_dir is None:
            if not load_name:
                raise ValueError('load_name must be provided if outdir is not specified')
            # Chop any directory path from load_name
            load_name = Path(load_name).name
            report_dir = re.sub(r'(_proseco)?.pkl(.gz)?', '', load_name) + '_sparkles'
        report_dir = Path(report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)

    # Do the sparkles review for all the catalogs
    for aca in acars:
        if not isinstance(aca, ACAReviewTable):
            raise TypeError('input catalog for review must be an ACAReviewTable')

        if loud:
            print(f'Processing obsid {aca.obsid}')

        # Don't clear messages
        # aca.messages.clear()
        aca.context.clear()

        aca.set_stars_and_mask()  # Not stored in pickle, need manual restoration
        aca.check_catalog()

        # Find roll options if requested
        if roll_level == 'all' or aca.messages >= roll_level:
            # Get roll selection algorithms to try
            max_roll_options = roll_args.pop('max_roll_options', 10)
            methods = roll_args.pop(
                'method', ('uniq_ids', 'uniform') if aca.is_OR else 'uniq_ids')
            if isinstance(methods, str):
                methods = [methods]

            try:
                # Set roll_options, roll_info attributes
                for method in methods:
                    aca.roll_options = None
                    aca.roll_info = None
                    aca.get_roll_options(method=method, **roll_args)
                    aca.roll_info['method'] = method

                    # If there is at least one option with no messages at the
                    # roll_level (typically "critical") then declare success and
                    # stop looking for roll options.
                    if any(not roll_option['acar'].messages >= roll_level
                           for roll_option in aca.roll_options):
                        break

                aca.sort_and_limit_roll_options(roll_level, max_roll_options)

            except Exception:  # as err:
                err = traceback.format_exc()
                aca.add_message('critical', text=f'Running get_roll_options() failed: \n{err}')
                aca.roll_options = None
                aca.roll_info = None

        if make_html:

            # Output directory for the main prelim review index.html and for this obsid.
            # Note that the obs{aca.obsid} is not flexible because it must match the
            # convention used in ACATable.make_report().  Oops.
            aca.preview_dir = Path(report_dir)
            aca.obsid_dir = aca.preview_dir / f'obs{aca.obsid}'
            aca.obsid_dir.mkdir(parents=True, exist_ok=True)

            aca.make_starcat_plot()

            if report_level == 'all' or aca.messages >= report_level:
                try:
                    aca.make_report()
                except Exception:
                    err = traceback.format_exc()
                    aca.add_message('critical', text=f'Running make_report() failed:\n{err}')

            if aca.roll_info:
                aca.make_roll_options_report()

            aca.context['text_pre'] = aca.get_text_pre()
            aca.context['call_args'] = aca.get_call_args_pre()

    # noinspection PyDictCreation
    if make_html:
        # from . import __version__
        __version__ = "0.1"

        # Create new context or else use a copy of the supplied dict
        if context is None:
            context = {}
        else:
            context = context.copy()

        context['load_name'] = load_name.upper()
        context['proseco_version'] = proseco.__version__
        context['sparkles_version'] = __version__
        context['chandra_aca_version'] = chandra_aca.__version__
        context['acas'] = acars
        context['summary_text'] = get_summary_text(acars)

        # Special case when running a set of roll options for one obsid
        is_roll_report = all(aca.is_roll_option for aca in acars)

        label_frame = 'ACA' if aca.is_ER else 'Target'
        context['id_label'] = f'{label_frame} roll' if is_roll_report else 'Obsid'

        template_file = SPARKLES_DIR / 'index_template_preview.html'
        template = Template(open(template_file, 'r').read())
        out_html = template.render(context)

        out_filename = report_dir / 'index.html'
        if loud:
            print(f'Writing output review file {out_filename}')
        with open(out_filename, 'w') as fh:
            fh.write(out_html)

        if open_html:
            import webbrowser
            out_url = f'file://{out_filename.absolute()}'
            print(f'Open URL in browser: {out_url}')
            webbrowser.open(out_url)


def sub_in_stars(aca):

    # Sort the candidates by mag
    aca.guides.cand_guides.sort('mag')
    args = aca.call_args.copy()

    # Sub in the faintest and stop if it still has no criticals
    for i, n in enumerate(range(len(aca.guides.cand_guides) - 1, len(aca.guides) - 1, -1)):
        star1 = aca.guides.cand_guides['id'][n]
        star2 = aca.guides.cand_guides['id'][n - 1]
        print(f"trying candidates {star1} {star2}")
        args['include_ids_guide'] = [star1, star2]
        naca = proseco.get_aca_catalog(**args)
        nacar = naca.get_review_table()
        nacar.run_aca_review()
        if len(nacar.messages == 'critical') == 0:
            break
    return naca, star1, star2


def do(load_name):
    acas = pickle.load(gzip.open(load_name))
    aca_arr = [acas[cat] for cat in acas]
    aca_arr = sorted(aca_arr, key=lambda k: k.meta['date'])

    # Do we want to parse the ms to check the last one?  Otherwise no duration.
    candidates = []
    for idx, aca in enumerate(aca_arr[0:-1]):
        obsid = aca.meta['obsid']
        next_aca = aca_arr[idx + 1]
        man_dur = duration(aca.meta['att'], next_aca.meta['att'])
        man_start = DateTime(next_aca.meta['date']).secs - man_dur
        obs_dur = man_start - DateTime(aca.meta['date']).secs
        if (obsid > 40000) and (obsid < 59000) and (obs_dur >= MIN_DWELL):
            print(f"Found candidate {obsid} of dur {obs_dur} at {aca.meta['date']}")
            candidates.append(
                {'obsid': obsid,
                 'dwell_end': man_start})

    acars = []

    for cand in candidates:

        obsid = cand['obsid']
        # Reselect stars with 16 arcsec dither
        cargs = acas[obsid].call_args.copy()
        cargs['dither_acq'] = (16, 16)
        cargs['dither_guide'] = (16, 16)
        cat = proseco.get_aca_catalog(**cargs)

        ncat, star1, star2 = sub_in_stars(cat)

        # Use those stars in the three observations in the splits
        cargs['include_ids_guide'] = list(ncat.guides['id'])

        sar_args = [cargs.copy(), cargs.copy(), cargs.copy()]
        sar_args[0]['obsid'] = obsid

        sar_args[1]['date'] = DateTime(DateTime(sar_args[0]['date']).secs + 4250).date
        sar_args[1]['obsid'] = obsid + 0.1
        sar_args[1]['dither_acq'] = (8, 8)
        sar_args[1]['dither_guide'] = (8, 8)
        sar_args[1]['img_size_guide'] = 6

        sar_args[2]['date'] = DateTime(DateTime(sar_args[0]['date']).secs + 8500).date
        sar_args[2]['obsid'] = obsid + 0.2
        sar_args[2]['dither_acq'] = (8, 8)
        sar_args[2]['dither_guide'] = (8, 8)

        for sa in sar_args:
            new_cat = proseco.get_aca_catalog(**sa)
            acar = new_cat.get_review_table()
            acar.messages.append(
                {'text': f"Image SAR manual stars {[star1, star2]}",
                 'category': 'info'})
            acars.append(acar)
        acars[-1].messages.append(
            {'text': f"Dwell stop {DateTime(cand['dwell_end']).date}",
             'category': 'info'})

    if len(acars) > 0:
        _run_aca_review(load_name=load_name, acars=acars, loud=True,
                        make_html=True, report_level='all')
    else:
        print("No opportunities found")


def main():
    opt = get_options()
    do(opt.load_name)


if __name__ == '__main__':
    main()
