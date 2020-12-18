<div>
<img style="float:left; border-radius:50%" src="https://avatars2.githubusercontent.com/u/58553327?s=460&u=3276252f07fb379c248bc8c9ce344bfdcaed7c45&v=4" width="40px">
</div>
<br>


A python library for audio features extraction.

beta.0 (December 2020) version

## Documentation

See documentation [here]()


## Dependencies

* [Numpy](https://numpy.org/)
* [torch]()
* [plotly]()
* [Librosa]()
* [Flask]()
* [Flask cors]()
* [Coverage]()


## Repository Structure

Directories:

* <strong>vae</strong>: module containing VAE architecture, training and data uitilities.
    * .py: main module.
    * configs.py
    * data_utils.py
    * loss.py
    * model.py
    * train.py
    * plot_utils.py


* <strong>plots</strong>: latent space and reconstruction plots.


* <strong>tests</strong>:
    * .py


* <strong>notebooks</strong>:
    * 1-.ipynb
    * 2-.ipynb
    * 3-.ipynb
  
  
* <strong>webserver</strong>: integration of audioanalysis module in Flask.



## Installation

```
cd .path/to/timbre-vae
python setup.py install
```

## References

[1] 

## Authors

[**Carlos Hernández**](https://carlosholivan.github.io/index.html) - carloshero@unizar.es

Department of Electronic Engineering and Communications, Universidad de Zaragoza, Calle María de Luna 3, 50018 Zaragoza