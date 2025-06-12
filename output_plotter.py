import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors

def plot_output_timewise(
    df, time, output, discrimination, legend=True, sort_legend=True,
    lighten_factor=0.35, ylim=None, justswarm=False, justpoint=False,
    colors=['C0', 'C2', 'C1', 'C4', 'C5', 'C6'],
    figsize=(3.5, 2.5), index=False,
    symbols=['o', 'o', 's', 's', '^', '^', '^', 's', '^']
):
    """
    Plots time-wise output from a DataFrame using swarm and point plots.

    Parameters:
        df (pd.DataFrame): The data to plot.
        time (str): Column name for the x-axis (time variable).
        output (str): Column name for the y-axis (output variable).
        discrimination (str): Column name for hue (categorical group).
        legend (bool): Whether to display the legend. Default is True.
        sort_legend (bool): Whether to sort legend/groups by discrimination. Default is True.
        lighten_factor (float): Amount to lighten swarmplot colors (0–1). Default is 0.35.
        ylim (tuple or None): Y-axis limits, e.g., (0, 10). Default is None.
        justswarm (bool): If True, only the swarmplot is created. Default is False.
        justpoint (bool): If True, only the pointplot is created. Default is False.
        colors (list): List of colors (Matplotlib codes, names, hex, or RGB).
        figsize (tuple): Figure size in inches. Default is (3.5, 2.5).
        index (bool): If True, resets and cleans index columns. Default is False.
        symbols (list): Marker styles for pointplot. Default: ['o', 'o', 's', ...].
    """

    df4plot = df.copy()
    if index:
        df4plot = df4plot.reset_index()
        df4plot = df4plot.loc[:, ~df4plot.columns.duplicated()]

    plt.figure(figsize=figsize)
    plt.rcParams['figure.dpi'] = 300

    if sort_legend:
        df4plot = df4plot.sort_values(by=discrimination)

    def to_rgb(color):
        """Convert a Matplotlib color code, hex, or RGB tuple to a normalized RGB tuple."""
        if isinstance(color, str):
            try:
                return mcolors.to_rgb(color)
            except ValueError:
                if color.startswith('C'):
                    # Matplotlib cycle color, fallback
                    idx = int(color[1:])
                    prop_cycle = plt.rcParams['axes.prop_cycle']
                    return mcolors.to_rgb(prop_cycle.by_key()['color'][idx])
                raise ValueError(f"Unknown color code: {color}")
        elif isinstance(color, tuple) and len(color) == 3:
            if all(0 <= channel <= 1 for channel in color):
                return color
            elif all(0 <= channel <= 255 for channel in color):
                return tuple(channel / 255 for channel in color)
            else:
                raise ValueError(f"RGB values must be in range 0-1 or 0-255: {color}")
        else:
            raise ValueError(f"Invalid color format: {color}")

    # Prepare color palettes
    colors_rgb = [to_rgb(c) for c in colors]
    lightened_colors = [
        tuple(min(1.0, channel + lighten_factor * (1.0 - channel)) for channel in color)
        for color in colors_rgb
    ]

    # Plotting
    if not justpoint:
        sns.swarmplot(
            x=time, y=output, hue=discrimination, data=df4plot, size=3,
            zorder=1, palette=lightened_colors
        )

    if not justswarm:
        point = sns.pointplot(
            x=time, y=output, hue=discrimination, data=df4plot, errorbar='sd',
            markersize=5, markers=symbols, palette=colors_rgb, zorder=100, dodge=0.3,
            err_kws={'linewidth': 1.5}, capsize=0.1,
        )
        plt.setp(point.lines, linewidth=0.75)

    if ylim:
        plt.ylim(ylim)
    # Legend handling
    leg = plt.legend()
    if leg:
        leg.set_visible(legend)
    plt.tight_layout()
    plt.show()

def mpl_color_to_rgb(code):
    """
    Converts a Matplotlib standard color code (e.g., 'C0', 'C1') to an RGB tuple.

    Args:
        code (str): The Matplotlib color code (e.g., 'C0').

    Returns:
        tuple: An RGB tuple (e.g., (0, 0, 0)).
    """
    # Get the color from Matplotlib's default color cycle
    color_list = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_index = int(code[1:]) % len(color_list)  # Handle cyclic color cycle

    # Convert the color name to RGB
    return mcolors.to_rgb(color_list[color_index])
