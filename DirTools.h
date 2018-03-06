#ifndef DIRTOOLS_H
#define DIRTOOLS_H

#include <sys/stat.h>
#include <string>

namespace DirTools {


    bool check_dir_exists(std::string path);

    bool check_file_exists(std::string path);

    bool create_dir(std::string path);

    char* string_to_char(std::string str);

    bool string_ending_slash(std::string str);

    std::string combine_path_file(std::string path, std::string file);

    std::string get_pwd();

    std::string get_file_from_pwd(std::string path, std::string file);

    std::string get_path_from_pwd(std::string path);
}

#endif