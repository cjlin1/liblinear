# predict_stream

ADD_EXECUTABLE(predict_stream
    predict_stream.cpp
    )

TARGET_LINK_LIBRARIES(predict_stream
    ${Boost_PROGRAM_OPTIONS_LIBRARY}
    ${Boost_FILESYSTEM_LIBRARY}
    ${Boost_THREAD_LIBRARY}
    ${Boost_SYSTEM_LIBRARY}
	${LINEAR_LIBRARIES}
    )

INSTALL(TARGETS predict_stream RUNTIME DESTINATION bin LIBRARY DESTINATION bin ARCHIVE DESTINATION bin)
