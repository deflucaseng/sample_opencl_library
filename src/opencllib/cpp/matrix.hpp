#ifndef MATRIX_HPP
#define MATRIX_HPP

class Matrix{
	public:
		Matrix(float* data, int height, int width): data(data), height(height), width(width){}
		~Matrix(){delete data;}
		float* get_data() const {return data;}
		int get_height() const {return height;}
		int get_width() const {return width;}

	private:
		float* data;
		int height;
		int width;
};


#endif