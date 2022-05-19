cmake_minimum_required(VERSION 3.8)
# プロジェクト名を指定する
project(CaseClosed)

# 次の行を追加するとMinSizeRelとRelWithDebInfoの選択肢を生成することが抑制できます。
set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "limited configs" FORCE)

add_executable(CaseClosed CaseClosed.cpp)
# C++11
target_compile_features(CaseClosed
  PRIVATE cxx_std_11
  )
# マクロ定数WITH_SOMELIBを追加
target_compile_definitions(CaseClosed
  PRIVATE WITH_SOMELIB
  )

# OpenCVのパッケージを探す
find_package(OpenCV REQUIRED)

# OpenCVが見つかった場合のみ設定を行う
if(OpenCV_FOUND)
  # インクルードパスを指定
  target_include_directories(CaseClosed
    PRIVATE
      ${OpenCV_INCLUDE_DIRS}
      ${PYTHON_INCLUDE_DIRS}
    )

  # リンクするライブラリ指定
  target_link_libraries(CaseClosed
    ${OpenCV_LIBS}
    ${PYTHON_LIBRARIES}
    )
endif(OpenCV_FOUND)
