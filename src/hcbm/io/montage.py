"""Channel selection per dataset: 'shared' = the 12 channels in every corpus; 'full' = FACED's 30 EEG channels where the
dataset has them (protocol.yaml full_channels / full_channel_idx; DREAMER has no full montage)."""


def channels(cfg, ds, montage="shared"):
    """Return (channel names, 0-based indices into the dataset's raw channel axis), in protocol order."""
    if montage == "shared":
        names = list(cfg["shared_channels"])
        m = ({c: e - 1 for c, e in cfg["dens"]["shared_channel_egi"].items()} if ds == "dens"
             else cfg[ds]["shared_channel_idx"])
    elif montage == "full":
        full = cfg["full_channel_idx"]["dens_egi" if ds == "dens" else ds]
        names = [c for c in cfg["full_channels"] if c in full]
        m = {c: v - 1 if ds == "dens" else v for c, v in full.items()}
    else:
        raise ValueError(montage)
    return names, [m[c] for c in names]
