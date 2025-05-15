<a href="https://fertmeneses.github.io/" target="_blank"> <img src="assets/website_icon_FM.jpg" alt="Logo FM" style="width: 3%;" /> </a> <a href="https://fertmeneses.github.io/" target="_blank"> Go to Home Page </a> 

# Machine Learning and Quantum Diamond Magnetometry applied to Object Monitoring

My flagstaff project as a research fellow at The University of Melbourne combined **Machine Learning** architectures with experimental **Quantum Sensing** to solve a real-life problem: **monitor the position of an object using magnetic fields**. 

Magnetic sensing is a powerful technique that complements other methods such as Global Positioning System (GPS) for object monitoring applications. It is particularly interesting in scenarios where GPS signals are denied, such as **indoors, underground or underwater**, because **magnetic signals can propagate in these environments**. Also, **magnetic tracking does not require that the target is equipped with a receiver**, which broadens the application scope for this technology.

The **major challenge for magnetic sensing** comes with the **data analysis and interpretation**. Usually, magnetic information has a complex signature convoluted with the environmental noise, requiring sophiticated physical models and calculations that allow a precise interpretation to correlate the position of an object with its magnetic signal. **In this project**, we solve this problem by developing a **Machine Learning algorithm that learns to predict the position of an object based on magnetic measurements without the need of any physical models**.

'''Figure: Quantum Sensing + Machine Learning = Object monitoring'''

## Quantum Sensing Technology

Our **sensing platform** is a solid-state **nitrogen-vacancy defect** within a **diamond** sample, which leverages the quantum properties of the defect and can precisely measure magnetic field variations down to less than 1 nT. In our application, we are **tracking an elevator which can travel up to 8 floors**, making a good example of an indoor environment where GPS signals are denied. The elevator generates **magnetic field variations in the order of few hundreds of nT**, then our sensitivity is more than enough to detect those little variations. Just to give a reference to the reader, the magnetic field of the Earth is approximately 50,000 nT!

Briefly, the **quantum diamond magnetometer** operates with a **green laser** which excites the electrons within the nitrogen-vacancy defect, pushing them to high energy levels. Since high energy levels naturally decay into low energy levels, we employ a **photodetector** to read the light emmited in that quantum process. Additionally, we can manipulate the quantum system by using **microwaves** and moving the electrons into "bright" or "dark" levels, which have different light emission intensities. As the energy difference between these dark and bright levels depend on the **external magnetic field**, we can use a protocol that detects variations in the light emission and precisely identify the external magnetic field.

'''Figure: Quantum Diamond Magnetometer'''

## Machine Learning Architecture

**How do we correlate the elevator position $Z$ with the magnetic signal $B$?** As the elevator moves across the building, the magnetic field varies in the three spatial directions X,Y,Z, and we measure those signals as $(B_X,B_Y,B_Z)$. Then, our objective is to **build a ML algorithm that receives the input $(B_X,B_Y,B_Z)$ and predicts the output $Z$ position of the elevator**. However, magnetic data is noisy and it fluctuates over time, so instead of just using the input $(B_X,B_Y,B_Z)$ for each position $Z$, we use the **magnetic information from the latest few seconds**, namely a time window $\Delta t$, to predict a single position. 

This approach may misleadingly appear as a forecast application, but the truth is that the elevator can have a completely arbitrary schedule for its travels, and we are not trying to find a pattern for the users' behavior. What we are trying to do is to **correlate the elevator's position with the magnetic field, for any travelling scheme**!

Our ML algorithm works very similar to a **computer vision application**, but instead of analizying a 2D image, it studies a **1D timeseries**. As shown in the image below, **the input data are three short timeseries** (you can think of them as colors in an image), each one belonging to a spatial direction of the magnetic field $(B_X,B_Y,B_Z)$. The algorithm uses a **set of convolutional layers**, correlating close data points and extracting features, which are then feed to **fully connected layers**, finally producing a **single output: the Z position of the elevator**. As this calculation can be made really fast (much less than 1 ms), the ML algorithm can predict position after position as we measure the magnetic fields (with a frequency of 0.1 seconds), working as a **real-time object monitoring application**.

'''Figure: ML architecture'''

## Object Monitoring Application

Xxxx

'''Figure: Prediction results simplified'''

<center><figure>
  <img src="assets/Image_Conceptual_Bottles.jpg" alt="Conceptual image"> 
  <figcaption><sup>Conceptual image, AI-generated using the prompt: "Two shelves with crystal bottles of many sizes and shapes, with an alchemist theme".</sup></figcaption>
</figure></center>

## Perspectives

Other scenarios, possible ML improvements.

## Further information

[arXiv](https://arxiv.org/abs/2502.14683)

[Virtual lecture](https://www.youtube.com/watch?v=5ZBcUqQFWfI)

Data is available at [GitHub project](https://github.com/Fertmeneses/ML_QDM_Meneses_et_al).

-----

[🔼 Back to top](#machine-learning-and-quantum-diamond-magnetometry-applied-to-object-monitoring)

<a href="https://fertmeneses.github.io/" target="_blank"> <img src="assets/website_icon_FM.jpg" alt="Logo FM" style="width: 3%;" /> </a> <a href="https://fertmeneses.github.io/" target="_blank"> Go to Home Page </a> 

![Banner](https://github.com/Fertmeneses/coding_challenge_bottle_sets/blob/main/assets/Banner.png?raw=true)

