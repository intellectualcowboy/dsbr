# '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'

plot_config = {
    "TentFull": {
        "color": "#1f77b4",    # blue
        "marker": "o",         # circle
        "linestyle": "-",      # solid
        "markersize": 4
    }, 
    "TentFull_0.1lr": {
        "color": "#1f77b4",    # blue
        "marker": "o",         # circle
        "linestyle": "-.",      # dash-dot
        "markersize": 4
    }, 
    "TentFull_0.01lr": {
        "color": "#1f77b4",    # blue
        "marker": "o",         # circle
        "linestyle": "--",      # dashed
        "markersize": 4
    }, 
    "SAR": {
        "color": "#ff7f0e",    # orange
        "marker": "D",         # diamond
        "linestyle": "-",      # solid
        "markersize": 4
    }, 
    "SAR_10.0lr": {
        "color": "#ff7f0e",    # orange
        "marker": "D",         # diamond
        "linestyle": "-.",      # dash-dot
        "markersize": 4
    }, 
    "SAR_100.0lr": {
        "color": "#ff7f0e",    # orange
        "marker": "D",         # diamond
        "linestyle": "--",     # dashed
        "markersize": 4
    }, 
    "SAR_NO_SAM": {
        "color": "#ff7f0e",    # orange
        "marker": "D",         # diamond
        "linestyle": "-.",     # dash-dot
        "markersize": 4
    }, 
    "COME": {
        "color": "#2ca02c",    # green
        "marker": "s",         # square
        "linestyle": "-",      # solid
        "markersize": 4
    }, 
    "COME_NO_REG": {
        "color": "#2ca02c",    # green
        "marker": "s",         # square
        "linestyle": "-.",     # dash-dot
        "markersize": 4
    }, 
    "DeYO_PIXEL": {
        "color": "#d62728",    # red
        "marker": "^",         # triangle
        "linestyle": "-",      # solid
        "markersize": 4
    },
    "DeYO_OCC": {
        "color": "#d62728",    # red
        "marker": "^",         # triangle
        "linestyle": "-.",     # dash-dot
        "markersize": 4
    }, 
    "DeYO_PATCH": {
        "color": "#d62728",    # red
        "marker": "^",         # triangle
        "linestyle": ":",      # dotted
        "markersize": 4
    }, 
    "No Adaptation": {
        "color": "#9467bd",    # purple
        "linestyle": "--",     # dashed
        "markersize": 4
    },
    "Source Train Results": {
        "color": "#8c564b",    # brown
        "linestyle": "--",     # dashed
        "markersize": 4
    },
    "Source Eval Results": {
        "color": "#e377c2",    # pink
        "linestyle": "--",     # dashed
        "markersize": 4
    },
}