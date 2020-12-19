import os
import torch
import pandas as pd
import plotly.express as px
import matplotlib.animation as animation

# Our modules
from vae import configs


def load_pretrained_model(model, trained_epochs):
    model.load_state_dict(torch.load(configs.ParamsConfig.TRAINED_MODELS_PATH + '/saved_model_' + str(trained_epochs) + "epochs.bin"))
    model.eval()
    return


def data_to_pandas_dataframe(model, trained_epochs, dataset, latent_dims):

    load_pretrained_model(model, trained_epochs)

    df_total = pd.DataFrame(columns=['epochs', 'x', 'y', 'z'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    instruments = []
    for i, x in enumerate(dataset):
        z_mu, z_sigma = model.encoder(x['input'].to(device))  #  model.encoder(x['input'].to(device))
        z = z_mu.to('cpu').detach().numpy()
        instrument = x['file'][0].split('_')[0]

        if latent_dims == 2:
            df_total = df_total.append({"epochs": trained_epochs,
                                        "x" : z[:, 0][0], 
                                        "y" : z[:, 1][0],
                                        "instrument": instrument}, ignore_index=True)

        elif latent_dims == 3:
            df_total = df_total.append({"epochs": trained_epochs,
                                        "x" : z[:, 0][0],
                                        "y" : z[:, 1][0],
                                        "z" : z[:, 2][0],
                                        "instrument": instrument}, ignore_index=True)
        else:
            raise ValueError("Too many dimensions.")

    return df_total


def plot_latent_space(model, trained_epochs, dataset, latent_dims=2, save_html=True, save_png=True):

    df_total = data_to_pandas_dataframe(model, trained_epochs, dataset, latent_dims)

    title = 'Latent space of VAE trained {} epochs'.format(trained_epochs)

    if latent_dims == 2:
        fig = px.scatter(df_total, x="x", y="y", color="instrument", title=title)
        fig.update_traces(marker=dict(opacity=0.5))
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
        name_fig = 'html_2d_latent_space_'

    elif latent_dims == 3:
        fig = px.scatter_3d(df_total, x="x", y="y", z="z", color="instrument", title=title)
        fig.update_traces(marker=dict(size=2, opacity=0.7))
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
        name_fig = 'html_3d_latent_space_'


    if save_html or save_png:
        if not os.path.exists(configs.PlotsConfig.PLOTS_PATH):
            os.mkdir(configs.PlotsConfig.PLOTS_PATH) 

        if save_html:
            fig.write_html(configs.PlotsConfig.PLOTS_PATH + '/' + name_fig + str(trained_epochs) + 'epochs.html')

        if save_png:
            fig.write_image(configs.PlotsConfig.PLOTS_PATH + '/' + name_fig + str(trained_epochs) + 'epochs.png')

    fig.show()

    return

