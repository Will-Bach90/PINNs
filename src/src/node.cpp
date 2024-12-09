#include "../include/node.h" 

Node::Node(std::size_t size) 
    : value(size,0.0)
    , grad_(size,0.0) 
{
    size_ = size;
}

void Node::zeroGrad()
{
    for (auto &g : grad_)  
    {
        g = 0.0; 
    }      
}