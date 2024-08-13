import matplotlib.pyplot as plt
import numpy as np
import mne
import matplotlib

from mpl_toolkits.axes_grid1 import ImageGrid

def plot_edges(importance, sub):
    ch_names_renamed =  ['Fp1', 'Fp2', 'F7', 'F3','Fz','F4','F8','T3','C3','Cz','C4','T4','T5','P3','Pz','P4','T6','O1','O2']

    #fig, ax = plt.subplots(ncols=5, figsize=(20, 4), gridspec_kw=dict(top=0.9),
    #                           sharex=False, sharey=False)
    fig = plt.figure(figsize=(20, 4))
    ax = ImageGrid(fig, 111,
                nrows_ncols = (1,5),
                axes_pad = 0.15,
                share_all=False,
                cbar_location = "right",
                cbar_mode="single",
                cbar_size="5%",
                cbar_pad=0.05
                )
    
    #fig.colorbar(ax, orientation='vertical', fraction=0.046, pad=0.04)
    # create an index for each tick position
    my_xticks = ch_names_renamed
    x = range(19)
    
    vmin = np.min(importance)
    vmax = np.max(importance)
    
    im = ax[0].imshow(importance[0], cmap='bwr', interpolation='nearest', vmin=vmin, vmax=vmax)
    ax[0].set_xticks(x, my_xticks,  rotation=90, fontsize=13)
    ax[0].set_yticks(x, my_xticks, fontsize=13)
    
    im2 = ax[1].imshow(importance[1], cmap='bwr', interpolation='nearest', vmin=vmin, vmax=vmax)
    ax[1].set_xticks(x, my_xticks,  rotation=90, fontsize=13)
    ax[1].set_yticks(x, my_xticks, fontsize=13)

    im3 = ax[2].imshow(importance[2], cmap='bwr', interpolation='nearest', vmin=vmin, vmax=vmax)
    ax[2].set_xticks(x, my_xticks,  rotation=90, fontsize=13)
    ax[2].set_yticks(x, my_xticks, fontsize=13)
    
    im4 = ax[3].imshow(importance[3], cmap='bwr', interpolation='nearest', vmin=vmin, vmax=vmax) 
    #fig.colorbar(im4, orientation='vertical', fraction=0.046, pad=0.04)
    ax[3].set_xticks(x, my_xticks,  rotation=90, fontsize=13)
    ax[3].set_yticks(x, my_xticks, fontsize=13)
    
    im5 = ax[4].imshow(importance[4], cmap='bwr', interpolation='nearest', vmin=vmin, vmax=vmax)
    #fig.colorbar(im5, orientation='vertical', fraction=0.046, pad=0.04)
    #fig.colorbar(im5, orientation='vertical', fraction=0.046, pad=0.04)
    #fig.colorbar(im5, ax=(ax[1], ax[2]), orientation='vertical')
    ax[4].set_xticks(x, my_xticks,  rotation=90, fontsize=13)
    ax[4].set_yticks(x, my_xticks, fontsize=13)
    
    ax[0].set_title('Delta', fontsize=15)
    ax[1].set_title('Theta', fontsize=15)
    ax[2].set_title('Alpha', fontsize=15)
    ax[3].set_title('Beta', fontsize=15)
    ax[4].set_title('Gamma', fontsize=15)
    #fig.colorbar(im)
    #fig.subplots_adjust(wspace=0.15)
    plt.colorbar(im, cax=ax.cbar_axes[0], format='%0.1f')
    fig.savefig("figs/edge_importance_AD"+str(sub)+".pdf", bbox_inches='tight')
    plt.show()
    
    
    
def plot_frequency_bands_graphs(data_feat, sub):
    # electrode positions for the graph nodes
    # based on the international 10-20 system
    #FP1, FP2, F7, F3, Fz, F4, T7, C3, C4, T8, P7, #P3, P4, P8, O1, and O2
    #ch_names_renamed =  ['FP1', 'FP2', 'F7', 'F3','FZ','F4','F8','T3','C3','CZ','C4','T4','T5','P3','PZ','P4','T6','O1','O2']
    ch_names_renamed =  ['Fp1', 'Fp2', 'F7', 'F3','Fz','F4','F8','T3','C3','Cz','C4','T4','T5','P3','Pz','P4','T6','O1','O2']

    node_positions = {'FP1': (-0.4, 0.45), 'FP2': (0.4, 0.45),
                      'F7': (-1.25, 0.33), 'F3': (-0.65, 0.255),
                      'FZ': (0.0, 0.225), 'F4': (0.65, 0.255),
                      'F8': (1.25, 0.33), 'T3': (-1.5, 0.0),
                      'C3': (-0.75, 0.0), 'CZ': (0.0, 0.0),
                      'C4': (0.75, 0.0), 'T4': (1.5, 0.0),
                      'T5': (-1.25, -0.33), 'P3': (-0.65, -0.255),
                      'PZ': (0.0, -0.225), 'P4': (0.65, -0.255),
                      'T6': (1.25, -0.33), 'O1': (-0.4, -0.45),
                      'O2': (0.4, -0.45)}

    nodelist = node_positions.keys()

    # Form the 10-20 montage
    mont1020 = mne.channels.make_standard_montage('standard_1020')
    # Choose what channels you want to keep 
    # Make sure that these channels exist e.g. T1 does not exist in the standard 10-20 EEG system!
    kept_channels = ch_names_renamed
    ind = []
    kept_channels = [ch.lower() for ch in kept_channels]
    for i, channel in enumerate(mont1020.ch_names):
        if channel.lower() in kept_channels:
            ind.append(i)
    #ind = [i for (i, channel) in enumerate(mont1020.ch_names) if channel in kept_channels]
    mont1020_new = mont1020.copy()
    # Keep only the desired channels
    mont1020_new.ch_names = [mont1020.ch_names[x] for x in ind]
    kept_channel_info = [mont1020.dig[x+3] for x in ind]
    # Keep the first three rows as they are the fiducial points information
    mont1020_new.dig = mont1020.dig[0:3]+kept_channel_info
    #mont1020.plot()

    #fig, ax = plt.subplots(ncols=5, figsize=(20, 4), gridspec_kw=dict(top=0.9),
    #                       sharex=True, sharey=True)
    
    fig = plt.figure(figsize=(20, 4))
    ax = ImageGrid(fig, 111,
                nrows_ncols = (1,5),
                axes_pad = 0.01,
                share_all=False,
                cbar_location = "right",
                cbar_mode="single",
                cbar_size="5%",
                cbar_pad=0.1
                )
    
    data = data_feat.reshape(19, 5)
    #fake_info = mne.create_info(ch_names=mont1020_new.ch_names, sfreq=250., ch_types='eeg')
    #ch_names_renamed =  ['Fp1', 'Fp2', 'F7', 'F3','Fz','F4','F8','T3','C3','Cz','C4','T4','T5','P3','Pz','P4','T6','O1','O2']
    fake_info = mne.create_info(ch_names=ch_names_renamed, sfreq=250., ch_types='eeg')
    fake_evoked = mne.EvokedArray(data, fake_info)
    fake_evoked.set_montage(mont1020_new)
    #print(fake_evoked.ch_names)
    vmin = np.min(data)
    vmax = np.max(data)

    img, _ = mne.viz.plot_topomap(fake_evoked.data[:, 0], fake_evoked.info, show=False, axes=ax[0], 
                                  cmap="bwr",vlim=(vmin, vmax), names=ch_names_renamed)
    #cbar = plt.colorbar(ax=ax, shrink=0.75, orientation='vertical', mappable=img)
    img, _ =mne.viz.plot_topomap(fake_evoked.data[:, 1], fake_evoked.info, show=False, axes=ax[1], 
                                 cmap="bwr",vlim=(vmin, vmax), names=ch_names_renamed)
    #cbar = plt.colorbar(ax=ax, shrink=0.75, orientation='vertical', mappable=img)
    img, _ =mne.viz.plot_topomap(fake_evoked.data[:, 2], fake_evoked.info, show=False, axes=ax[2], 
                                 cmap="bwr",vlim=(vmin, vmax), names=ch_names_renamed)
    #cbar = plt.colorbar(ax=ax, shrink=0.75, orientation='vertical', mappable=img)
    img, _ =mne.viz.plot_topomap(fake_evoked.data[:, 3], fake_evoked.info, show=False, axes=ax[3], 
                                 cmap="bwr",vlim=(vmin,vmax), names=ch_names_renamed)
    #cbar = plt.colorbar(ax=ax, shrink=0.75, orientation='vertical', mappable=img)
    img, _ =mne.viz.plot_topomap(fake_evoked.data[:, 4], fake_evoked.info, show=False, axes=ax[4], 
                                 cmap="bwr", vlim=(vmin, vmax), names=ch_names_renamed) 
    #cbar = plt.colorbar(ax=ax, shrink=0.75, orientation='vertical', mappable=img)
    ax[0].set_title('Delta', fontsize=15)
    ax[1].set_title('Theta', fontsize=15)        
    ax[2].set_title('Alpha', fontsize=15)
    ax[3].set_title('Beta', fontsize=15)
    ax[4].set_title('Gamma', fontsize=15)
    plt.colorbar(img, cax=ax.cbar_axes[0], format='%0.1f')

    for tt in plt.findobj(fig, matplotlib.text.Text):
        if tt.get_text() in fake_evoked.ch_names:
            tt.set_fontsize(13)
        
    fig.savefig("figs/node_importance_AD"+str(sub)+".pdf", bbox_inches='tight')
    
    
def plot_bands_per_regions(zeros, bands):
    # electrode positions for the graph nodes
    # based on the international 10-20 system
    ch_names_renamed =  ['Fp1', 'Fp2', 'F7', 'F3','Fz','F4','F8','T3','C3','Cz',
                         'C4','T4','T5','P3','Pz','P4','T6','O1','O2']
    
    node_positions = {'FP1': (-0.4, 0.45), 'FP2': (0.4, 0.45),
                      'F7': (-1.25, 0.33), 'F3': (-0.65, 0.255),
                      'FZ': (0.0, 0.225), 'F4': (0.65, 0.255),
                      'F8': (1.25, 0.33), 'T3': (-1.5, 0.0),
                      'C3': (-0.75, 0.0), 'CZ': (0.0, 0.0),
                      'C4': (0.75, 0.0), 'T4': (1.5, 0.0),
                      'T5': (-1.25, -0.33), 'P3': (-0.65, -0.255),
                      'PZ': (0.0, -0.225), 'P4': (0.65, -0.255),
                      'T6': (1.25, -0.33), 'O1': (-0.4, -0.45),
                      'O2': (0.4, -0.45)}

    nodelist = node_positions.keys()

    # Form the 10-20 montage
    mont1020 = mne.channels.make_standard_montage('standard_1020')
    # Choose what channels you want to keep 
    kept_channels = ch_names_renamed
    ind = []
    kept_channels = [ch.lower() for ch in kept_channels]
    for i, channel in enumerate(mont1020.ch_names):
        if channel.lower() in kept_channels:
            ind.append(i)
    mont1020_new = mont1020.copy()
    # Keep only the desired channels
    mont1020_new.ch_names = [mont1020.ch_names[x] for x in ind]
    kept_channel_info = [mont1020.dig[x+3] for x in ind]
    # Keep the first three rows as they are the fiducial points information
    mont1020_new.dig = mont1020.dig[0:3]+kept_channel_info
    
    fake_info = mne.create_info(ch_names=ch_names_renamed, sfreq=250., ch_types='eeg')
    fake_evoked = mne.EvokedArray(zeros, fake_info)
    fake_evoked.set_montage(mont1020_new)
    zeros = np.mean(zeros, -1)
    mask_params = dict(marker='x', markerfacecolor='w', markeredgecolor='k', linewidth=0, markersize='10%')
        
    img, _ = mne.viz.plot_topomap(zeros, fake_evoked.info, show=False, cmap="bwr", 
                                  names=bands, size=6, mask_params=mask_params)