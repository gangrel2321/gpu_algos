#include <iostream>
#include <memory>
#include <string>

std::shared_ptr<int> ptr(new int(3));

class Node {


};

class DifferentiableOp {
public: 

    virtual DifferentiableOp* backprop(DifferentiableOp* x) const = 0;
    virtual double eval() const = 0;
    virtual std::string toString() const = 0;
    virtual ~DifferentiableOp() = default;

};

class Constant : public DifferentiableOp {
private:
    double value; 

public:
    explicit Constant(double val) : value(val) {}
    
    Constant* backprop(DifferentiableOp* x) const override {
        return new Constant(0);
    }

    double eval() const override {
        return value;
    }

    std::string toString() const override {
        return std::to_string(value);
    }
    // Destructor
    ~Constant() override = default;

};

class Variable : public DifferentiableOp { 
private:
    double value; 
    std::string name;

public:
    explicit Variable(std::string name, double val = 0) : name(name), value(val) {}
    
    Constant* backprop(DifferentiableOp* x) const override {
        return x == this ? new Constant(0) : new Constant(1);
    }

    double eval() const override {
        return value;
    }

    std::string toString() const override {
        return name + ": " + std::to_string(value);
    }

    // Destructor
    ~Variable() override = default;
};

class Sum : public DifferentiableOp { 
private:
    DifferentiableOp* x; 
    DifferentiableOp* y;

public:
    explicit Sum(DifferentiableOp* x, DifferentiableOp* y) : x(x), y(y) {}
    
    Sum* backprop(DifferentiableOp* var) const override {
        return new Sum(x->backprop(var), y->backprop(var));
    }

    double eval() const override {
        return x->eval() + y->eval();
    }

    std::string toString() const override {
        return " ";
    }

    // Destructor
    ~Sum() override = default;
};