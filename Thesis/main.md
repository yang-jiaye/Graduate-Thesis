$$
\begin{equation}
    \Delta\mathbf{R}_i=\mathbf{R}_{i-1}^T\mathbf{R}_i =\exp\left(\left(\widetilde{\omega}_{i-1}-\mathbf{b}_{i-1}^g-\eta_{i-1}^{gd}\right)\Delta t_i\right)
\end{equation}
$$

$$
\Delta\mathbf{v}_i=\mathbf{R}_{i-1}^T(\mathbf{v}_i-\mathbf{v}_{i-1}-\mathbf{g}\Delta t_i)\\
=\Delta\mathbf{R}_i(\widetilde{\mathbf{a}}_{i-1}-\mathbf{b}_{i-1}^a-\eta_{i-1}^{ad})\Delta t_i
$$

$$
\Delta\mathbf{p}_i=\mathbf{R}_i^T\left(\mathbf{p}_i-\mathbf{p}_{i-1}-\mathbf{v}_{i-1}\Delta t_i-\frac{1}{2}\mathbf{g}\Delta t_i^2 \right)\\
=\frac{1}{2}(\widetilde{\mathbf{a}}_{i-1}-\mathbf{b}_{i-1}^a-\eta_{i-1}^{ad})\Delta t_i^2
$$