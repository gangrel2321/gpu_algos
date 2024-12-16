#include <iostream>
#include <memory>
#include <string>

class DifferentiableOp;
class Add;
class Prod;
class Constant;

class DifferentiableOp {
public: 

    virtual DifferentiableOp backprop(Variable& x) const {
        return Constant(0); 
    };
    virtual double eval() const {
        return 0;
    };
    virtual std::string toString() const {
        return "GenericDifferentiableOp";
    };
    virtual ~DifferentiableOp() = default;

    Add operator+(const DifferentiableOp& other){
        return Add(*this, other);
    }
    Prod operator*(const DifferentiableOp& other){
        return Prod(*this, other);
    }

};

class Constant : public DifferentiableOp {
private:
    double value; 

public:
    explicit Constant(double val) : value(val) {}
    
    DifferentiableOp backprop(Variable& x) const override {
        return Constant(0);
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
    
    DifferentiableOp backprop(Variable& x) const override {
        return x.name == name ? Constant(1) : Constant(0);
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
    DifferentiableOp x; 
    DifferentiableOp y;

public:
    explicit Add(DifferentiableOp x, DifferentiableOp y) : x(x), y(y) {}
    
    DifferentiableOp backprop(Variable& var) const override {
        return Add(x.backprop(var), y.backprop(var));
    }

    double eval() const override {
        return x.eval() + y.eval();
    }

    std::string toString() const override {
        return x.toString() + " + " + y.toString();
    }

    // Destructor
    ~Add() override = default;
};

class Prod : public DifferentiableOp { 
private:
    DifferentiableOp x; 
    DifferentiableOp y;
public:
    explicit Prod(DifferentiableOp x, DifferentiableOp y) : x(x), y(y) {}
    
    DifferentiableOp backprop(Variable& var) const override {
        return Add(
            Prod(x.backprop(var),y), Prod(x, y.backprop(var))
        );
    }

    double eval() const override {
        return x.eval() * y.eval();
    }

    std::string toString() const override {
        return x.toString() + " * " + y.toString();
    }

    // Destructor
    ~Prod() override = default;
};


int main() {
    Variable x = Variable("x",3);
    Variable y = Variable("y",2);
    DifferentiableOp z = (x*x) + (Constant(3)*x*y) + Constant(1);
    std::cout << z.toString() << std::endl;
}