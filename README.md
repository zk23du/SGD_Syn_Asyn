# SGD_Syn_Asyn
## Synchronous Training in Distributed Systems
In a distributed setting, synchronous training typically involves multiple workers (which could be processes or machines) that handle different subsets of the dataset. The key characteristic of synchronous training is that all workers synchronize their updates at specific points in the training process. Here’s how it typically works:
1. Gradient Computation: Each worker computes gradients based on its portion of the data.
2. Synchronization Point: All workers send their computed gradients to a central point (like a parameter server or through a collective communication operation such as Allreduce).
3. Averaging Gradients: The gradients from all workers are averaged together. This averaged gradient is then used to update the model parameters.
4. Barrier: There is often a barrier operation, where all workers must wait until every worker has reached this point of synchronization before they can proceed with further computations. This ensures that all workers use the same version of the model for the next iteration of training.

### Advantages and Disadvantages
#### Advantages:
1. Consistency: Because all workers update their models with the same averaged gradients, the model remains consistent across all workers. This typically results in more stable convergence, especially in complex models and tasks.
2. Parallelism: Synchronous training allows for efficient use of distributed resources by parallelizing the computation of gradients.

### Disadvantages:
1. Latency: The slowest worker determines the pace of the training process because all workers must wait for each other at the synchronization point. This can introduce significant delays, especially if the computational capabilities or data loads are unevenly distributed among workers.
2. Communication Overhead: Especially in large-scale setups, the communication needed to average the gradients can become a bottleneck.
   
## Asynchronous Training as an Alternative
Asynchronous training is another approach in distributed systems where workers update the shared model parameters independently, without waiting for other workers. This can alleviate the issues of latency and uneven load but might introduce problems related to convergence and model consistency due to the potential for conflicts and stale gradients.
1. Independent Updates: Each worker computes gradients independently based on its data and sends these updates to a central parameter server (or directly updates the shared parameters) without waiting for the other workers.
2. Immediate Application: The parameter server (or the model parameters storage in case of direct updates) immediately applies these updates to the model as soon as they are received, potentially leading to situations where the model parameters continuously change as new updates are applied.
3. No Synchronization Barrier: There is no global barrier or synchronization point where workers must wait for each other. This can lead to faster iteration times, especially in heterogeneous environments where some workers are faster than others.
   
## Non-Distributed Synchronous SGD
In non-distributed settings, synchronous training typically refers to the regular process of updating model parameters using the full batch or mini-batches of the dataset sequentially. Each update is dependent on the previous update, ensuring that the training process is synchronous in terms of sequential progression through iterations.

