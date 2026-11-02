"""Export screenshot-ready PNG/SVG figures and CSV tables from a saved report.

Reads reports only; never runs inference, training, or evaluation.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.diagnostics import report_diagnostics


def write_csv(path, rows, keys):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=ROOT / 'evaluation/metrics_report.json')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'evaluation/figures')
    parser.add_argument('--scope', choices=['overall', 'flood', 'wildfire', 'landslide'], default='overall')
    args = parser.parse_args(argv)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        parser.error('Install figure dependencies: .venv\\Scripts\\python.exe -m pip install -r requirements-evaluation.txt')
    report = json.loads(args.report.read_text(encoding='utf-8-sig'))
    detail = report_diagnostics(report)
    classification = detail.get('classification', {}).get(args.scope, {})
    probability = classification.get('probability_diagnostics', {})
    scope = args.scope
    smoke = detail.get('provenance') == 'generated_sample_smoke_test'
    banner = 'GENERATED-DATA SMOKE DIAGNOSTICS — NOT BENCHMARK PERFORMANCE' if smoke else 'SAVED REFERENCE / PREDICTION EVALUATION — CHECK COVERAGE AND READINESS'
    if detail.get('provenance') == 'real_public_data_transfer_test':
        banner = 'REAL-DATA TRANSFER TEST - LIMITED SAMPLE; TRAINING LINEAGE UNVERIFIED'
    output = args.output_dir / scope
    output.mkdir(parents=True, exist_ok=True)
    artifacts = []
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    def save(fig, name):
        fig.suptitle(banner, fontsize=10, color='#a2581d' if smoke else '#326446', y=.995)
        fig.text(.5, .008, f'Scope: {scope} | Classified cases: {classification.get("sample_count",0)} | Source: {args.report.name}', ha='center', fontsize=8)
        fig.tight_layout(rect=(0,.035,1,.95))
        for suffix in ('png','svg'):
            path = output / f'{name}.{suffix}'
            fig.savefig(path, dpi=240, facecolor='white', bbox_inches='tight')
            artifacts.append(path.name)
        plt.close(fig)
    def unavailable(ax, reason):
        ax.axis('off')
        ax.text(.5,.55,'NOT EVALUATED',ha='center',va='center',fontsize=20,color='#68776a',transform=ax.transAxes)
        ax.text(.5,.35,reason,ha='center',va='center',fontsize=10,wrap=True,transform=ax.transAxes)
    labels = classification.get('labels', [])
    n = classification.get('sample_count', 0)
    for normalized, name in [(False,'confusion_matrix'),(True,'confusion_matrix_normalized')]:
        fig,ax = plt.subplots(figsize=(8,6))
        if n:
            matrix = np.array(classification['confusion_matrix_row_normalized' if normalized else 'confusion_matrix'],dtype=float)
            im = ax.imshow(np.ma.masked_invalid(matrix),cmap='Greens',vmin=0,vmax=1 if normalized else max(1,float(np.nanmax(matrix))))
            fig.colorbar(im,ax=ax,label='Row fraction' if normalized else 'Number of cases')
            ax.set_xticks(range(len(labels)),labels,rotation=25)
            ax.set_yticks(range(len(labels)),labels)
            ax.set_xlabel('Predicted class'); ax.set_ylabel('Actual class')
            for i in range(len(labels)):
                for j in range(len(labels)):
                    value = matrix[i,j]
                    text = 'N/A' if not np.isfinite(value) else f'{value:.1%}' if normalized else str(int(value))
                    ax.text(j,i,text,ha='center',va='center',fontsize=14)
            ax.set_title('Confusion matrix · row normalized' if normalized else 'Confusion matrix · counts')
        else:
            unavailable(ax,'No eligible saved prediction/reference pairs.')
        save(fig,name)
    rows = classification.get('per_class', [])
    fig,ax = plt.subplots(figsize=(9,5))
    if n:
        positions = np.arange(len(rows))
        for i,(key,color) in enumerate([('precision','#237b51'),('recall','#4974b7'),('f1','#c8703f')]):
            ax.bar(positions+(i-1)*.24,[np.nan if r[key] is None else r[key] for r in rows],width=.24,label=key,color=color)
        ax.set_xticks(positions,[f'{r["class"]}\n(n={r["support"]})' for r in rows]); ax.set_ylim(0,1.1)
        ax.set_ylabel('Score'); ax.set_title('Per-class precision, recall and F1'); ax.legend()
        ax.text(0,-.2,'Undefined class scores are omitted. Inspect class support before interpreting results.',transform=ax.transAxes,fontsize=9)
    else:
        unavailable(ax,'Per-class scores require verified references and saved predictions.')
    save(fig,'per_class_scores')
    for mode in ('roc','precision_recall'):
        fig,ax = plt.subplots(figsize=(8,5))
        curves = [(label,curve) for label,curve in probability.get('curves',{}).items() if curve['status']=='available']
        if curves:
            # Nested widths and dash patterns reveal coincident curves without
            # offsetting any FPR, TPR, recall or precision values.
            curve_styles = {
                'flood': dict(color='#0072B2', linewidth=6, linestyle='-', zorder=3),
                'wildfire': dict(color='#E69F00', linewidth=4, linestyle='--', zorder=4),
                'landslide': dict(color='#009E73', linewidth=2, linestyle=':', zorder=5),
            }
            for label,curve in curves:
                style = curve_styles.get(label, {})
                if mode=='roc':
                    ax.plot(curve['roc']['fpr'],curve['roc']['tpr'],label=f'{label} AUC={curve["roc_auc"]:.3f}', **style)
                else:
                    ax.step(curve['precision_recall']['recall'],curve['precision_recall']['precision'],where='post',label=f'{label} AP={curve["average_precision"]:.3f}', **style)
            ax.text(.98, .04, 'Coincident curves share the same coordinates; styles distinguish hazards.',
                    transform=ax.transAxes, ha='right', va='bottom', fontsize=8,
                    bbox=dict(facecolor='white', alpha=.85, edgecolor='none'), zorder=6)
            if mode=='roc':
                ax.plot([0,1],[0,1],'--',color='gray',label='No-skill diagonal')
            ax.set(xlabel='False positive rate' if mode=='roc' else 'Recall',ylabel='True positive rate' if mode=='roc' else 'Precision',xlim=(0,1),ylim=(0,1.02))
            ax.set_title('One-versus-rest ROC curves' if mode=='roc' else 'One-versus-rest precision–recall curves'); ax.legend(handlelength=3.5)
        else:
            unavailable(ax,'Requires full class probabilities and positive AND negative references per class.')
        save(fig,mode+'_curves')
    # Each class is evaluated one-versus-rest on the complete mixed-class set.
    # Filtering to true positives only would make ROC undefined.
    available_curves = [(name, item) for name, item in probability.get('curves', {}).items()
                        if item.get('status') == 'available']
    colors = {'flood': '#0072B2', 'wildfire': '#D55E00', 'landslide': '#009E73'}
    for label, item in available_curves:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(item['roc']['fpr'], item['roc']['tpr'], color=colors.get(label),
                linewidth=2.5, marker='o', markersize=3,
                label=f'{label} AUC={item["roc_auc"]:.3f}')
        ax.plot([0, 1], [0, 1], '--', color='gray', label='No-skill diagonal')
        ax.set(xlabel='False positive rate', ylabel='True positive rate',
               xlim=(-.02, 1.02), ylim=(-.02, 1.02), title=f'{label.title()} - one-versus-rest ROC')
        ax.legend(loc='lower right'); ax.grid(alpha=.15)
        save(fig, 'roc_' + label)
    if available_curves:
        fig, axes = plt.subplots(1, len(available_curves), figsize=(5 * len(available_curves), 4.7), squeeze=False)
        for ax, (label, item) in zip(axes[0], available_curves):
            ax.plot(item['roc']['fpr'], item['roc']['tpr'], color=colors.get(label), linewidth=2.5)
            ax.plot([0, 1], [0, 1], '--', color='gray')
            ax.set(xlabel='False positive rate', ylabel='True positive rate',
                   xlim=(-.02, 1.02), ylim=(-.02, 1.02),
                   title=f'{label.title()} | AUC={item["roc_auc"]:.3f}')
            ax.grid(alpha=.15)
        save(fig, 'three_hazard_roc')
    fig,ax = plt.subplots(figsize=(7,5))
    bins = [b for b in probability.get('calibration_bins',[]) if b['count']]
    if bins:
        ax.plot([0,1],[0,1],'--',color='gray',label='Perfect calibration')
        ax.scatter([b['confidence'] for b in bins],[b['accuracy'] for b in bins],s=[35+10*b['count'] for b in bins],color='#237b51')
        ax.set(xlabel='Mean confidence',ylabel='Observed accuracy',xlim=(0,1),ylim=(0,1.02),title='Reliability diagram · 10 confidence bins')
        ax.legend()
    else:
        unavailable(ax,'Complete probability vectors are unavailable. Top confidence is insufficient.')
    save(fig,'calibration')
    write_csv(output/'per_class_report.csv',rows,['class','support','predicted_count','tp','fp','fn','tn','precision','recall','f1','specificity','negative_predictive_value'])
    write_csv(output/'confusion_matrix.csv',[dict(actual=label,**dict(zip(labels,row))) for label,row in zip(labels,classification.get('confusion_matrix',[]))],['actual',*labels])
    write_csv(output/'per_case_predictions.csv',classification.get('per_case',[]),['case_id','true_hazard','predicted_hazard'])
    distribution = detail.get('metric_distributions',{}).get(scope,{})
    coverage = report.get('metric_coverage',{}).get(scope,{})
    write_csv(output/'metric_coverage_and_distributions.csv',[{'metric':name,**value,**distribution.get(name,{})} for name,value in coverage.items()],['metric','eligible_count','test_count','reason','mean','std_population','median','p25','p75','p95','minimum','maximum'])
    (output/'diagnostics.json').write_text(json.dumps(detail,indent=2,allow_nan=False),encoding='utf-8')
    import html
    summary = classification.get('summary', {})
    metric_keys = [('accuracy', 'Accuracy'), ('macro_precision', 'Macro precision'),
                   ('macro_recall', 'Macro recall'), ('macro_f1', 'Macro F1')]
    def percentage(value):
        return 'N/A' if value is None else f'{value:.2%}'
    cards = ''.join('<li><b>' + title + ':</b> ' + percentage(summary.get(key)) + '</li>'
                    for key, title in metric_keys)
    table_rows = ''.join('<tr><td>' + html.escape(row['class']) + '</td><td>' + str(row['support'])
                         + '</td><td>' + percentage(row['precision']) + '</td><td>'
                         + percentage(row['recall']) + '</td><td>' + percentage(row['f1']) + '</td></tr>'
                         for row in rows)
    limitation_items = ''.join('<li>' + html.escape(str(item)) + '</li>'
                               for item in report.get('metadata', {}).get('limitations', []))
    overview = '<section><h2>Saved model results</h2><p>Evaluated cases: ' + str(n) + '</p><ul>' + cards + '</ul>'
    overview += '<table cellpadding="9" style="border-collapse:collapse;background:white" border="1"><tr><th>Hazard</th><th>Cases</th><th>Precision</th><th>Recall</th><th>F1</th></tr>' + table_rows + '</table>'
    if limitation_items:
        overview += '<details open><summary><b>Protocol and interpretation limits</b></summary><ul>' + limitation_items + '</ul></details>'
    overview += '</section>'
    gallery = ''.join(f'<figure><img src="{html.escape(name)}" alt="{html.escape(name)}"><figcaption>{html.escape(name)}</figcaption></figure>' for name in artifacts if name.endswith('.png'))
    (output/'index.html').write_text('<!doctype html><html><head><meta charset="utf-8"><title>Evaluation figures</title><style>body{font:16px system-ui;margin:30px;background:#f5f7f5;color:#263b2d}h1{font-size:23px}figure{background:white;padding:20px;margin:20px 0;break-inside:avoid}img{max-width:100%}strong{color:#925820}@media print{body{background:white}}</style></head><body><h1>'+html.escape(banner)+'</h1><p>'+html.escape(detail.get('notice',''))+'</p><p>Source: '+html.escape(str(args.report))+'</p>'+overview+gallery+'</body></html>',encoding='utf-8')
    print(f'Figures and tables saved to {output.resolve()}')
    print(f'Open {(output / "index.html").resolve()} for a screenshot-ready gallery.')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
