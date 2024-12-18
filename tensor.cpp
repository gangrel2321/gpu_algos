#include <iostream>
#include <memory>
#include <string>

class Variable;

template <class T>
class Node {
public: 

    virtual double backprop(Node<Variable>& x) {
        return static_cast<T*>(this)->backprop(x); 
    };
    virtual double eval() const {
        return static_cast<const T*>(this)->eval();
    };
    virtual std::string toString() const {
        return static_cast<const T*>(this)->toString();
    };
    T cast() const {
        return *static_cast<const T*>(this); 
    }
    virtual ~Node() = default;

};

class Variable : public Node<Variable> { 

double value; 
std::string name;
public:
    explicit Variable(std::string name, double val = 0) : name(name), value(val) {}
    
    double backprop(Node<Variable>& x) override {
        try {
            Variable& var_ref = dynamic_cast<Variable&>(x);
            return var_ref.name == name ? 1.0 : 0.0;
        }
        catch (const std::bad_cast& e){
            return 0.0;
        }
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

class Constant : public Node<Constant> {
private:
    double value; 

public:
    explicit Constant(double val) : value(val) {}
    
    double backprop(Node<Variable>& x) override {
        return 0.0;
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
template <typename U, typename V>
class Add : public Node<Add<U,V>> { 
private:
    Node<U>& x; 
    Node<V>& y;

public:
    explicit Add(Node<U>& x, Node<V>& y) : x(x), y(y) {}
    
    double backprop(Node<Variable>& var) override {
        return x.cast().backprop(var) + y.cast().backprop(var);
    }

    double eval() const override {
        return x.cast().eval() + y.cast().eval();
    }

    std::string toString() const override {
        return x.cast().toString() + " + " + y.cast().toString();
    }

    // Destructor
    ~Add() override = default;
};

template <typename U, typename V>
class Prod : public Node<Prod<U,V>> { 
private:
    Node<U>& x; 
    Node<V>& y;
public:
    explicit Prod(Node<U>& x, Node<V>& y) : x(x), y(y) {}
    
    double backprop(Node<Variable>& var) override {
        return x.cast().backprop(var)*y.cast().eval() + x.cast().eval()*y.cast().backprop(var);
    }

    double eval() const override {
        return x.cast().eval() * y.cast().eval();
    }

    std::string toString() const override {
        return x.cast().toString() + " * " + y.cast().toString();
    }

    // Destructor
    ~Prod() override = default;
};


int main() {
    Variable x = Variable("x",3);
    Variable y = Variable("y",5);
    Variable p = Variable("p",0);
    Prod a = Prod(x,y);
    Constant b = Constant(22);
    auto z = Add(x,a);
    auto c = Add(z, b);
    
    std::cout << "TEST" << std::endl;
    std::cout << "Reverse Diff: " << c.backprop(x) << std::endl;
    std::cout << "Eval: " << c.eval() << std::endl;
    // Node z = (x*x) + (Constant(3)*x*y) + Constant(1);
    // std::cout << z.toString() << std::endl;
}