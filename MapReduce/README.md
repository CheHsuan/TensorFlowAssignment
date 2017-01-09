# Abstract
This is a simple example of distributed tensorflow(in-graph and synchronous).Create two workers to compute PI value using monte carlo method.   

# Usage
Use three terminals to run this example. 
### terminal 1
```
python create_worker 0
```
### terminal 2
```
python create_worker 1
```
### terminal 3
```
 python compute-pi.py number
```
