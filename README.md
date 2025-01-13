# LaplacianHeatFlow
---

This experiment uses the USroads-48 data set (https://sparse.tamu.edu/Gleich/usroads-48), a sparse suite matrix graph with 126k vertices and 323k edges representing the continental US road network. 

The dataset was loaded into memory using a scipy sparse COO matrix, and for future reference occupies 6.4 GB of RAM when the entire graph is used. An efficient sparse matrix exponential function is used (Awad H. Al-Mohy and Nicholas J. Higham 2011) to compute each time step. For 25 uniform time steps in the range [0,140] this process took 16 seconds to compute and about 2 minutes to render as an animation.

For the experiment below I place a “heat mass” of 1 distributed uniformly in a small radius around Caltech. Then I allow the heat time evolution to take place for $v(t)= e^{-tL}v(0)$ . The following plot shows the time step difference. This is the graph given by 

 $$G(v(t)-v(t-1),E)$$
 
during time evolution. 

![incremental_heat_diffusionCT](https://github.com/user-attachments/assets/49c35457-6c6b-45f8-99af-c3eab29aab06)

![output](https://github.com/user-attachments/assets/d0b8c735-540b-4ad5-998c-abf35d846aa3)
