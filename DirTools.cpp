#include <zconf.h>
#include "DirTools.h"

namespace DirTools {

    bool check_dir_exists(std::string path) {
        struct stat info;
        if (stat(path.c_str(), &info) != 0) {
            return false;
        }
        else if (info.st_mode & S_IFDIR) {
            return true;
        }
        else {
            return false;
        }
    }

    bool check_file_exists(std::string path) {
        struct stat buffer;
        return (stat (path.c_str(), &buffer) == 0);
    }

    bool create_dir(std::string path) {
        int status = mkdir(path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        if (status == 0) {
            return true;
        }
        else {
            return false;
        }
    }

    char *string_to_char(std::string str) {
        char *cstr = &str[0u];
        return cstr;
    }

    bool string_ending_slash(std::string str) {
        if (str[str.size()-1] == '/') {
            return true;
        }
        else {
            return false;
        }
    }

    std::string combine_path_file(std::string path, std::string file) {
        if (string_ending_slash(path)) {
            return path + file;
        }
        else {
            return path + "/" + file;
        }
    }

    std::string get_pwd() {
        char cwd[1024];
        getcwd(cwd, sizeof(cwd));
        std::string dir(cwd);
        return dir;
    }

    std::string get_file_from_pwd(std::string path, std::string file) {
        //Returns path to the selected directory and file from the current
        //working directory
        std::string dir = get_pwd();
        if (!string_ending_slash(dir) && path[0] != '/') {
            dir += "/";
        }
        dir += path;
        if (!string_ending_slash(dir) && file[0] != '/') {
            dir += "/";
        }
        dir += file;
        return dir;
    }

    std::string get_path_from_pwd(std::string path) {
        std::string dir = get_pwd();
        if (!string_ending_slash(dir) && path[0] != '/') {
            dir += "/";
        }
        dir += path;
        return dir;
    }
}