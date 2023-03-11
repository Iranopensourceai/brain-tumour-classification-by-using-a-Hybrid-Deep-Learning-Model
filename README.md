<a name="readme-top"></a>
<!--
# brain-tumour-classification-by-using-a-Hybrid-Deep-Learning-Model
brain tumour classification by using a Hybrid Deep Learning Model
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

[![Product Name Screen Shot][product-screenshot]](https://example.com)

### brain tumor classification

Tumors are a large mass of cells which grow abnormally in any part of human anatomy. Based on their spread to other parts, tumors can be malignant (cancerous) and benign (noncancerous). Malignant tumors must be identified in early stages and treated immediately, which if neglected can lead to death of the person. Though several techniques based on deep learning and computer vision have been proposed by employing different types of CNN architectures, they did not focus on learning cross-channel and spatial correlations with temporal dependency present in features. 
Accomplishing the objective of this work, we use pre-trained Xception as a backbone model for initial representation learning; further, we design spatial and cross-channel attention blocks to learn channel-wise correlations and spatiotemporal features.This work is limited to identify the type of tumor and type of glioma present in given MR image.

Use the `BLANK_README.md` to get started.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Built With

* [![tensorflow][tensorflow.js]][tensorflow-url]
* [![keras][keras.io]][keras-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

#### inception 
[![Product Name Screen Shot][product-screenshot]](https://cloud.google.com/static/tpu/docs/images/inceptionv3onc--oview.png)

#### Model
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_6 (InputLayer)            [(None, 255, 255, 3) 0                                            
__________________________________________________________________________________________________
conv2d_470 (Conv2D)             (None, 127, 127, 32) 864         input_6[0][0]                    
__________________________________________________________________________________________________
batch_normalization_470 (BatchN (None, 127, 127, 32) 96          conv2d_470[0][0]                 
__________________________________________________________________________________________________
activation_470 (Activation)     (None, 127, 127, 32) 0           batch_normalization_470[0][0]    
__________________________________________________________________________________________________
conv2d_471 (Conv2D)             (None, 125, 125, 32) 9216        activation_470[0][0]             
__________________________________________________________________________________________________
batch_normalization_471 (BatchN (None, 125, 125, 32) 96          conv2d_471[0][0]                 
__________________________________________________________________________________________________
activation_471 (Activation)     (None, 125, 125, 32) 0           batch_normalization_471[0][0]    
__________________________________________________________________________________________________
conv2d_472 (Conv2D)             (None, 125, 125, 64) 18432       activation_471[0][0]             
__________________________________________________________________________________________________
batch_normalization_472 (BatchN (None, 125, 125, 64) 192         conv2d_472[0][0]                 
__________________________________________________________________________________________________
activation_472 (Activation)     (None, 125, 125, 64) 0           batch_normalization_472[0][0]    
__________________________________________________________________________________________________
max_pooling2d_20 (MaxPooling2D) (None, 62, 62, 64)   0           activation_472[0][0]             
...
Total params: 21,810,980
Trainable params: 19,634,564
Non-trainable params: 2,176,416
__________________________________________________________________________________________________

### Prerequisites


  ```

### Installation

_Below is an example of how you can instruct your audience on installing and setting up your app. This template doesn't rely on any external dependencies or services._

1. Get a free API Key at [https://example.com](https://example.com)
2. Clone the repo
   ```sh
   git clone https://github.com/your_username_/Project-Name.git
   ```
3. Install NPM packages
   ```sh
   npm install
   ```
4. Enter your API in `config.js`
   ```js
   const API_KEY = 'ENTER YOUR API';
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ROADMAP -->
## Roadmap
[![Product Name Screen Shot][product-screenshot]](https://cloud.google.com/static/tpu/docs/images/inceptionv3onc--oview.png)


<p align="right">(<a href="#readme-top">back to top</a>)</p>






<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->


[keras.io]: https://img.shields.io/badge/keras-FF0000?style=for-the-badge&logo=nextsdotjs&logoColor=white
[keras-url]: https://keras.io/
[tensorflow.js]: https://img.shields.io/badge/tensorflow-FFA500?style=for-the-badge&logo=nextsdotjs&logoColor=white
[tensorflow-url]: https://www.tensorflow.org/
