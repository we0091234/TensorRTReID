#ifndef _SCANF_H_
#define _SCANF_H_
#include <vector>
#include <string>
#include<dirent.h>
#include <sys/types.h>
#include <iostream>
#include<dirent.h>
#include <sys/types.h>
#include <string.h>
int scanFiles(std::vector<std::string> &fileList, std::string inputDirectory);
int readFileList(char *basePath,std::vector<std::string> &fileList,std::vector<std::string> fileType);
std::string getHouZhui(std::string fileName);
// bool is_dir(const char* path);
#endif