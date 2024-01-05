# DATA 562 Sample Project - Social License to Operate (SLO)

This sample analytics project is distributed for use by students of DATA 562 at Calvin University. It is based on the SLO application built at [CSIRO/Data61](https://research.csiro.au/data61/) between 2015-2020 ([Xu et al, COLING-2020, industry track](https://aclanthology.org/2020.coling-industry.14/)).

The project deployed a public-stance monitoring application what watched the Twitter feed for comments on Australian mining companies. The goal was to help the companies know what their public stakeholders were saying about them so that they could take appropriate actions in response.

For details, see the project:

- [Product/System Design](notebooks/product.ipynb)
- [Time Tracker](https://app.clockify.me/workspaces/649f2bd325d4ee229253c053)
- [Data Storage](https://drive.google.com/drive/u/1/folders/1te6TeLQ7Uq0-JzBbiBINytPkH7yCqy1b)
- [EDA](notebooks/analysis.ipynb)

To start the development container specified in `.devcontainer`, either create a Codespace on GitHub or run VSCode Command&rarr;&ldquo;Dev Container: Rebuild Container&rdquo; locally.

The sample dataset is a set of raw tweets on Australian mining companies collected from 2010-Jan-1 through 2018-May-10 (see [dataset.json](data/dataset.json)).
