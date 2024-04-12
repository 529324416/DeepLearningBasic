from pyecharts import options as opts
from pyecharts.charts import Line


def make_line(x, y, title, path) -> Line:
    c = (
        Line()
        .add_xaxis(x)
        .add_yaxis(title, y)
        .set_global_opts(
            title_opts=opts.TitleOpts(title=title),
            # yaxis_opts=opts.AxisOpts(
            #     axislabel_opts=opts.LabelOpts(formatter="{value:.2f}")
            # )
        )
    )
    c.render(path)
