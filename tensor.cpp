#include <iostream>
#include <memory>
#include <string>

std::shared_ptr<int> ptr(new int(3));

class Node {


};

class DifferentiableOp {
public: 

    virtual std::shared_ptr<DifferentiableOp> backprop(std::shared_ptr<DifferentiableOp> x) const = 0;
    virtual double eval() const = 0;
    virtual std::string toString() const = 0;
    virtual ~DifferentiableOp() = default;

};

class Constant : public DifferentiableOp {
private:
    double value; 

public:
    explicit Constant(double val) : value(val) {}
    
    std::shared_ptr<DifferentiableOp> backprop(std::shared_ptr<DifferentiableOp> x) const override {
        return std::make_shared<Constant>(0);
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
    
    std::shared_ptr<DifferentiableOp> backprop(std::shared_ptr<DifferentiableOp> x) const override {
        return x.get() == this ? std::make_shared<Constant>(1) : std::make_shared<Constant>(0);
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

class Add : public DifferentiableOp { 
private:
    std::shared_ptr<DifferentiableOp> x; 
    std::shared_ptr<DifferentiableOp> y;

public:
    explicit Add(std::shared_ptr<DifferentiableOp> x, std::shared_ptr<DifferentiableOp> y) : x(x), y(y) {}
    
    std::shared_ptr<DifferentiableOp> backprop(std::shared_ptr<DifferentiableOp> var) const override {
        return std::make_shared<Add>(x->backprop(var), y->backprop(var));
    }

    double eval() const override {
        return x->eval() + y->eval();
    }

    std::string toString() const override {
        return " ";
    }

    // Destructor
    ~Add() override = default;
};