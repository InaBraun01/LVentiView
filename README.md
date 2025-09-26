<div align="center">


  <img src="assets/logo.png" alt="logo" width="200" height="auto" />
  <h1>LVentiView</h1>
  
  <p>
    Read Me 
  </p>
  
  
<!-- Badges -->
<p>
  <a href="https://github.com/InaBraun01/LVentiView/graphs/contributors">
    <img src="https://img.shields.io/github/contributors/InaBraun01/LVentiView" alt="contributors" />
  </a>
  <a href="">
    <img src="https://img.shields.io/github/last-commit/InaBraun01/LVentiView" alt="last update" />
  </a>
  <a href="https://github.com/InaBraun01/LVentiView/network/members">
    <img src="https://img.shields.io/github/forks/InaBraun01/LVentiView" alt="forks" />
  </a>
  <a href="https://github.com/InaBraun01/LVentiView/stargazers">
    <img src="https://img.shields.io/github/stars/InaBraun01/LVentiView" alt="stars" />
  </a>
  <a href="https://github.com/InaBraun01/LVentiView/issues/">
    <img src="https://img.shields.io/github/issues/InaBraun01/LVentiView" alt="open issues" />
  </a>
  <a href="https://github.com/InaBraun01/LVentiView/blob/master/LICENSE">
    <img src="https://img.shields.io/github/license/InaBraun01/LVentiView.svg" alt="license" />
  </a>
</p>
   
<h4>
    <a href="https://github.com/InaBraun01/LVentiView/">View Demo</a>
  <span> · </span>
    <a href="https://github.com/InaBraun01/LVentiView">Documentation</a>
  <span> · </span>
    <a href="https://github.com/InaBraun01/LVentiView/issues/">Report Bug</a>
  <span> · </span>
    <a href="https://github.com/InaBraun01/LVentiView/issues/">Request Feature</a>
  </h4>
</div>

<br />

<!-- Table of Contents -->
# Table of Contents

- [About the Project](#about-the-project)
- [About the Software](#about-the-software)
- [Graphical User Interphase](#graphical-user-interphase)
  * [Installation](#installation)
  * [Linux & macOS](#linux-&-macOS)
  * [Windows](#windows)
  * [Color Reference](#color-reference)
  * [Environment Variables](#environment-variables)
- [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
  * [Running Tests](#running-tests)
  * [Run Locally](#run-locally)
  * [Deployment](#deployment)
- [Usage](#usage)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
  * [Code of Conduct](#code-of-conduct)
- [FAQ](#faq)
- [License](#license)
- [Contact](#contact)
- [Acknowledgements](#acknowledgements)
  

<!-- About the Project -->
## About the Project
Accurate quantification of left ventricular cavity volume (LVV) and ejection fraction (EF) from cardiac magnetic resonance imaging (MRI) is essential for diagnosis and prognosis in cardiovascular care. However, current clinical methods for calculating EF are subject to substantial measurement uncertainty, primarily due to variability in manual segmentation and the use of basic geometric assumptions for interpolation between acquired 2D MRI slices. To address these challenges, LVentiView, an open-source Python platform with a graphical user interface, was developed. The platform integrates automated MRI segmentation, three-dimensional (3D) mesh generation, volumetric analysis, and regional myocardial thickness calculation. Accuracy of the volume calculation was assessed using an idealized left ventricular geometry with known ground-truth volumes. The software was further evaluated on clinical cardiac MRI datasets. LVentiView successfully performed automated segmentation and mesh-based 3D reconstruction, enabling accurate quantification of LVV and regional wall thickness for both an idealized geometry and clinical MRI datasets. By combining automation with an intuitive interface, LVentiView provides a user-friendly tool for quantifying LVV, EF, and regional myocardial thickness. It is designed with potential for integration into clinical practice.

<!-- About the Software -->
## About the Software
LVentiView is organized into two main modules: the Segmentation module and the Mesh Generation module. The software can be used either through a graphical user interface (GUI) or directly from the terminal. Regardless of the interface, the inputs and outputs remain identical.

<!-- Graphical User Interphase -->
## Graphical User Interphase

### Installation  
To install the graphical user interface (GUI), simply download the **.zip file** for your operating system and follow the instructions below.  

---

#### Linux & macOS  
1. Download the app (.zip file).  
2. Unzip the app to extract the `.app` file.  
3. Move the extracted `.app` file to your `/Applications` folder.  
4. You can now open **LVentiView** like any other application.  


#### Windows  
1. Download the installer.  
2. Run the installer and follow the on-screen instructions.  
3. Once installed, open **LVentiView** from the Start Menu or Desktop shortcut. 

---
### Start Page
On the **Start Page**, you’ll find an overview of the user interface along with descriptions of all parameters that can be configured manually.  
From here, you can also navigate directly to the two main modules of **LVentiView**:  
- **Segmentation Module**  
- **Mesh Generation Module**  

<div align="center"> 
  <img src="https://placehold.co/600x400?text=Your+Screenshot+here" alt="screenshot" />
</div>


### Segmentation Module
The **Segmentation Module** processes cardiac MRI series with a single click.  
Users simply choose an input folder and an output directory, then start segmentation by pressing the **Run Segmentation** button in the GUI.

By default, two post-processing steps are applied:  
- **Data cleaning**: removes incomplete time points and slices outside the mitral valve–apex range.  
- **Volume analysis**: calculates myocardial and blood pool volumes, identifies ED/ES states, and plots volume curves.  

The workflow is fully automated but can be customized by adjusting thresholds, enabling/disabling steps, or performing manual cleaning in the GUI.  

<div align="center"> 
  <img src="https://placehold.co/600x400?text=Your+Screenshot+here" alt="screenshot" />
</div>

### Mesh Generation Module
The **Mesh Generation Module** constructs a 3D LV model from segmented MRI data and provides LVV, EF, and myocardial thickness quantification.  

In the default workflow, users select a folder with segmentation data (.pkl) and an output directory. The fitting process is started by pressing the **Run Mesh Fitting** button in the GUI (or via the terminal). The software then builds one volumetric mesh per MRI time step, saved in `.vtk` format, along with visualizations and Dice scores assessing mesh quality.  

Two post-processing steps are included:  
- **Volume analysis**: computes myocardial and blood pool volumes, identifies ED/ES states, and plots them over the cardiac cycle.  
- **Thickness mapping**: generates local myocardial thickness maps for each time point.  

The workflow is automated but fully configurable—users can adjust fitting parameters, restrict fitting to selected time steps, and enable/disable analysis options.  

<div align="center"> 
  <img src="https://placehold.co/600x400?text=Your+Screenshot+here" alt="screenshot" />
</div>

---


<!-- Env Variables -->
### Environment Variables

To run this project, you will need to add the following environment variables to your .env file

`API_KEY`

`ANOTHER_API_KEY`

<!-- Getting Started -->
## Getting Started

<!-- Prerequisites -->
### Prerequisites

This project uses Yarn as package manager

```bash
 npm install --global yarn
```

<!-- Installation -->
### Installation

Install my-project with npm

```bash
  yarn install my-project
  cd my-project
```
   
<!-- Running Tests -->
### Running Tests

To run tests, run the following command

```bash
  yarn test test
```

<!-- Run Locally -->
### Run Locally

Clone the project

```bash
  git clone https://github.com/Louis3797/awesome-readme-template.git
```

Go to the project directory

```bash
  cd my-project
```

Install dependencies

```bash
  yarn install
```

Start the server

```bash
  yarn start
```


<!-- Deployment -->
### Deployment

To deploy this project run

```bash
  yarn deploy
```


<!-- Usage -->
## Usage

Use this space to tell a little more about your project and how it can be used. Show additional screenshots, code samples, demos or link to other resources.


```javascript
import Component from 'my-project'

function App() {
  return <Component />
}
```

<!-- Roadmap -->
## Roadmap

* [x] Todo 1
* [ ] Todo 2


<!-- Contributing -->
## Contributing

<a href="https://github.com/Louis3797/awesome-readme-template/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Louis3797/awesome-readme-template" />
</a>


Contributions are always welcome!

See `contributing.md` for ways to get started.


<!-- Code of Conduct -->
### Code of Conduct

Please read the [Code of Conduct](https://github.com/Louis3797/awesome-readme-template/blob/master/CODE_OF_CONDUCT.md)

<!-- FAQ -->
## FAQ

- Question 1

  + Answer 1

- Question 2

  + Answer 2


<!-- License -->
## License

Distributed under the no License. See LICENSE.txt for more information.


<!-- Contact -->
## Contact

Your Name - [@twitter_handle](https://twitter.com/twitter_handle) - email@email_client.com

Project Link: [https://github.com/Louis3797/awesome-readme-template](https://github.com/Louis3797/awesome-readme-template)


<!-- Acknowledgments -->
## Acknowledgements

Use this section to mention useful resources and libraries that you have used in your projects.

 - [Shields.io](https://shields.io/)
 - [Awesome README](https://github.com/matiassingers/awesome-readme)
 - [Emoji Cheat Sheet](https://github.com/ikatyang/emoji-cheat-sheet/blob/master/README.md#travel--places)
 - [Readme Template](https://github.com/othneildrew/Best-README-Template)
