#include "ImageProcessing.hpp"

/*          GPU codes           */    

void convertToGrayscale_GPU(sycl::buffer<uint8_t,3>& inp, sycl::buffer<uint8_t,2>& oup, sycl::queue q){

    q.submit([&](sycl::handler& h) {

        sycl::accessor inp_acc(inp, h, sycl::read_only);

        sycl::accessor oup_acc(oup, h, sycl::write_only);

        h.parallel_for(oup.get_range(), [=](sycl::id<2> i) {

            int tmp = 
                (inp_acc[i[0]][i[1]][0]) +
                (inp_acc[i[0]][i[1]][1]) +
                (inp_acc[i[0]][i[1]][2]);
            oup_acc[i[0]][i[1]] = tmp / 3;// (tmp >> 1) - tmp;
        });
     });
}

/*
void convertToGrayscale_GPU_Better(sycl::buffer<uint8_t, 3>& inp, sycl::buffer<uint8_t, 2>& oup, sycl::queue q) {
    
    size_t height = inp.get_range()[1], width = inp.get_range()[0];

    // M is row N is column.

    sycl::buffer<sycl::ext::oneapi::bfloat16, 2> row_vec(sycl::range<2>(8,16));
    row_vec.get_host_access()[0][0] = 1 / 3;
    row_vec.get_host_access()[1][0] = 1 / 3;
    row_vec.get_host_access()[2][0] = 1 / 3;

    sycl::buffer<sycl::ext::oneapi::bfloat16, 2> column_vec(sycl::range<2>(16,8));
    row_vec.get_host_access()[0][0] = 3;
    row_vec.get_host_access()[0][1] = 3;
    row_vec.get_host_access()[0][2] = 3;

    sycl::buffer<float, 2> output(sycl::range<2>(8,8));
    row_vec.get_host_access()[0][0] = 0;

    //std::vector<sycl::ext::oneapi::experimental::matrix::combination>



    q.submit([&](sycl::handler& h) {

        sycl::accessor inp_acc(inp, h, sycl::read_write);
        sycl::accessor row_acc(row_vec, h, sycl::read_write);
        sycl::accessor col_acc(column_vec, h, sycl::read_write);
        sycl::accessor oup_acc(output, h, sycl::read_write);

        h.parallel_for(sycl::nd_range<2>(oup.get_range(), { 1,1 }), [=](sycl::nd_item<2> workItem) {

            const auto global_idx = workItem.get_global_id(0);
            const auto global_idy = workItem.get_global_id(1);
            const auto sg_startx = global_idx - workItem.get_local_id(0);
            const auto sg_starty = global_idy - workItem.get_local_id(1);

            sycl::sub_group sg = workItem.get_sub_group();

            // 1 Row 3 Columns
            sycl::ext::oneapi::experimental::matrix::joint_matrix<
                sycl::sub_group, sycl::ext::oneapi::bfloat16,
                sycl::ext::oneapi::experimental::matrix::use::a,
                8,
                16,
                sycl::ext::oneapi::experimental::matrix::layout::row_major>     sub_a;

            // 3 Rows 1 Column
            sycl::ext::oneapi::experimental::matrix::joint_matrix<
                sycl::sub_group, sycl::ext::oneapi::bfloat16,
                sycl::ext::oneapi::experimental::matrix::use::b,
                16,
                8,
                sycl::ext::oneapi::experimental::matrix::layout::row_major>     sub_b;

            // 8 Row 8 Column
            sycl::ext::oneapi::experimental::matrix::joint_matrix<
                sycl::sub_group,
                float,
                sycl::ext::oneapi::experimental::matrix::use::accumulator,
                8,
                8,
                sycl::ext::oneapi::experimental::matrix::layout::dynamic>     sub_c;

            sycl::ext::oneapi::experimental::matrix::joint_matrix_fill(sg, sub_c, 0.0);

            sycl::ext::oneapi::experimental::matrix::joint_matrix_load(sg, sub_a, row_acc.get_pointer(), 0);
            
            sycl::ext::oneapi::experimental::matrix::joint_matrix_load(sg, sub_b, col_acc.get_pointer(), 0);
            
            sycl::ext::oneapi::experimental::matrix::joint_matrix_mad(sg, sub_a, sub_b, sub_c);

            sycl::ext::oneapi::experimental::matrix::joint_matrix_store(sg, sub_c, oup_acc.get_pointer(), 0, sycl::ext::oneapi::experimental::matrix::layout::row_major);
                
            });
        });

    try {
        q.wait_and_throw();
    } catch (sycl::exception const& e) {
        std::cout << "Caught synchronous SYCL exception:\n"
            << e.what() << std::endl;
    }
}
*/

void sobel_GPU(sycl::buffer<uint8_t,2>& inp, sycl::buffer<uint8_t,2>& oup, sycl::queue q){

    size_t height = inp.get_range()[1], width = inp.get_range()[0];

    q.submit([&](sycl::handler& h) {

        sycl::accessor inp_acc(inp, h, sycl::read_only);

        sycl::accessor oup_acc(oup, h, sycl::write_only);

        h.parallel_for(inp.get_range(), [=](sycl::id<2> i) {

            // Set the entire image to 255.
            oup_acc[i] = 0;

            if (i[0] > 0 && i[1] > 0 && i[0] < width && i[1] < height) {
                // Calculate the X gradient.
                int Y_Val =
                    (inp_acc[i[0] - 1][i[1] - 1] * -1) + (inp_acc[i[0]][i[1] - 1] * 0) + (inp_acc[i[0] + 1][i[1] - 1] * 1) +
                    (inp_acc[i[0] - 1][i[1]] * -2) + (inp_acc[i[0]][i[1]] * 0) + (inp_acc[i[0] + 1][i[1]] * 2) +
                    (inp_acc[i[0] - 1][i[1] + 1] * -1) + (inp_acc[i[0]][i[1] + 1] * 0) + (inp_acc[i[0] + 1][i[1] + 1] * 1);

                // Calculate the Y gradient.
                int X_Val =
                    (inp_acc[i[0] - 1][i[1] - 1] * -1) + (inp_acc[i[0]][i[1] - 1] * -2) + (inp_acc[i[0] + 1][i[1] - 1] * -1) +
                    (inp_acc[i[0] - 1][i[1]] * 0) + (inp_acc[i[0]][i[1]] * 0) + (inp_acc[i[0] + 1][i[1]] * 0) +
                    (inp_acc[i[0] - 1][i[1] + 1] * 1) + (inp_acc[i[0]][i[1] + 1] * 2) + (inp_acc[i[0] + 1][i[1] + 1] * 1);

                oup_acc[i] = (uint8_t)sycl::sqrt((float) (X_Val * X_Val + Y_Val * Y_Val));
            }
         });
     });
}

void floodFill_GPU(sycl::buffer<uint8_t, 2>& inp, sycl::buffer<uint8_t, 2>& oup, sycl::queue q, std::pair<size_t, size_t> startPoint, std::pair<uint8_t, uint8_t> threashhold) {

    size_t height = inp.get_range()[1], width = inp.get_range()[0];
    size_t lambda = height / 10;

    q.submit([&](sycl::handler& h) {
        sycl::accessor oup_acc(oup, h, sycl::write_only);

        h.parallel_for(oup.get_range(), [=](sycl::id<2> i) {
            oup_acc[i] = 0;

            if (i[0] == startPoint.first && i[1] == startPoint.second) oup_acc[i] = 1;
        });
    });

    sycl::buffer<size_t> areaFilled(sycl::range<1>(1));
    areaFilled.get_host_access()[0] = 0;
    size_t prevAreaFilled = 0;

    do {
        prevAreaFilled = areaFilled.get_host_access()[0];
        areaFilled.get_host_access()[0] = 0;

        for (int iter = 0; iter < lambda; iter++) {

            q.submit([&](sycl::handler& h) {

                sycl::accessor inp_acc(inp, h, sycl::read_only);
                sycl::accessor oup_acc(oup, h, sycl::read_write);

                h.parallel_for(inp.get_range(), [=](sycl::id<2> i) {

                    // If not an edge pixel.
                    if (i[0] > 0 && i[1] > 0 && i[0] < width - 1 && i[1] < height - 1) {
                        // If in the fill.
                        if (oup_acc[i] == 1) {

                            // Test left pixel.
                            if (inp_acc[i[0] - 1][i[1]] >= threashhold.first && inp_acc[i[0] - 1][i[1]] <= threashhold.second) {
                                oup_acc[i[0] - 1][i[1]] = 1;
                            }

                            // Test top pixel.
                            if (inp_acc[i[0]][i[1] + 1] >= threashhold.first && inp_acc[i[0]][i[1] + 1] <= threashhold.second) {
                                oup_acc[i[0]][i[1] + 1] = 1;
                            }

                            // Test right pixel.
                            if (inp_acc[i[0] + 1][i[1]] >= threashhold.first && inp_acc[i[0] + 1][i[1]] <= threashhold.second) {
                                oup_acc[i[0] + 1][i[1]] = 1;
                            }

                            // Test bottom pixel.
                            if (inp_acc[i[0]][i[1] - 1] >= threashhold.first && inp_acc[i[0]][i[1] - 1] <= threashhold.second) {
                                oup_acc[i[0]][i[1] - 1] = 1;
                            }
                        }
                    }
                });
            });
        }

        q.submit([&](sycl::handler& h) {

            sycl::accessor oup_acc(oup, h, sycl::read_only);
            sycl::accessor af_acc(areaFilled, h, sycl::write_only);

            h.parallel_for(inp.get_range()[0], [=](auto x) {
                size_t globalId = x[0];
                size_t filledPixels = 0;
                for (size_t y = 0; y < height; y++) {
                    filledPixels += oup_acc[globalId][y];
                }


                auto v = sycl::atomic_ref<size_t, sycl::memory_order::relaxed,
                    sycl::memory_scope::device,
                    sycl::access::address_space::global_space>(af_acc[0]);

                v.fetch_add(filledPixels);
            });

        });

    } while (prevAreaFilled != areaFilled.get_host_access()[0]);

    q.submit([&](sycl::handler& h) {
        sycl::accessor oup_acc(oup, h, sycl::read_write);

        h.parallel_for(oup.get_range(), [=](sycl::id<2> i) {
            oup_acc[i] *= 255;
        });
    });
}

void floodFill_GPU_Better(sycl::buffer<uint8_t, 2>& inp, sycl::buffer<uint8_t, 2>& oup, sycl::queue q, std::pair<size_t, size_t> startPoint, std::pair<uint8_t, uint8_t> threashhold) {

    size_t height = inp.get_range()[1], width = inp.get_range()[0];
    size_t lambda = height / 10;

    q.submit([&](sycl::handler& h) {
        sycl::accessor oup_acc(oup, h, sycl::write_only);

        h.parallel_for(oup.get_range(), [=](sycl::id<2> i) {
            oup_acc[i] = 0;

            if (i[0] == startPoint.first && i[1] == startPoint.second) oup_acc[i] = 255;
        });
    });

    sycl::buffer<size_t> areaFilled(sycl::range<1>(1));
    areaFilled.get_host_access()[0] = 0;
    size_t prevAreaFilled;

    uint32_t wg_size = std::floor((float)std::sqrt(q.get_device().get_info<sycl::info::device::max_work_group_size>()));
    uint32_t glob_size = wg_size * std::ceil((float)inp.get_range()[0] / wg_size);

    do {
        prevAreaFilled = areaFilled.get_host_access()[0];
        areaFilled.get_host_access()[0] = 0;

        for (int i = 0; i < lambda; i++) {

            q.submit([&](sycl::handler& h) {

                sycl::accessor inp_acc(inp, h, sycl::read_only);
                sycl::accessor oup_acc(oup, h, sycl::read_write);
                sycl::accessor af_acc(areaFilled, h, sycl::read_write);

                // Local scratch pad for each workgroup.
                sycl::accessor<int, 1, sycl::access::mode::read_write, sycl::access::target::local> wgp_reduct_scratch(wg_size * wg_size, h);

                h.parallel_for(sycl::nd_range<2>{{glob_size, glob_size}, { wg_size, wg_size }}, [=](sycl::nd_item<2> workItem) {
                    auto glob_i = workItem.get_global_id();
                    auto loc_i = workItem.get_local_id();
                    auto lin_loc_i = workItem.get_local_linear_id();

                    wgp_reduct_scratch[lin_loc_i] = 0;

                    // If not an edge pixel.
                    if (glob_i[0] > 0 && glob_i[1] > 0 && glob_i[0] < width - 1 && glob_i[1] < height - 1) {

                        // If in the fill.
                        if (oup_acc[glob_i] == 255) {
                            wgp_reduct_scratch[lin_loc_i] = 1;

                            // Test left pixel.
                            if (inp_acc[glob_i[0] - 1][glob_i[1]] >= threashhold.first && inp_acc[glob_i[0] - 1][glob_i[1]] <= threashhold.second) {
                                oup_acc[glob_i[0] - 1][glob_i[1]] = 255;
                            }

                            // Test top pixel.
                            if (inp_acc[glob_i[0]][glob_i[1] + 1] >= threashhold.first && inp_acc[glob_i[0]][glob_i[1] + 1] <= threashhold.second) {
                                oup_acc[glob_i[0]][glob_i[1] + 1] = 255;
                            }

                            // Test right pixel.
                            if (inp_acc[glob_i[0] + 1][glob_i[1]] >= threashhold.first && inp_acc[glob_i[0] + 1][glob_i[1]] <= threashhold.second) {
                                oup_acc[glob_i[0] + 1][glob_i[1]] = 255;
                            }

                            // Test bottom pixel.
                            if (inp_acc[glob_i[0]][glob_i[1] - 1] >= threashhold.first && inp_acc[glob_i[0]][glob_i[1] - 1] <= threashhold.second) {
                                oup_acc[glob_i[0]][glob_i[1] - 1] = 255;
                            }
                        }
                    }

                    // Do sum reduction over wg into the scratchpad.
                    // i is set initially to half wg_size then halved untill 0 in a loop.
                    for (int i = wg_size * wg_size / 2; i > 0; i >>= 1) {

                        // Synchronise workgroup.
                        workItem.barrier(sycl::access::fence_space::local_space);

                        // Perform the summation if in range.
                        if (lin_loc_i < i) wgp_reduct_scratch[lin_loc_i] += wgp_reduct_scratch[lin_loc_i + i];
                    }

                    // If the 0th work item in this workgroup we then submit the local wgp sum to the global value.
                    if (lin_loc_i == 0) {
                        auto v = sycl::atomic_ref<size_t, sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space>(af_acc[0]);

                        v.fetch_add(wgp_reduct_scratch[0]);
                    }
                });
            });
        }
    } while (areaFilled.get_host_access()[0] != prevAreaFilled);

}

void floodFill_GPU_Better_Better(sycl::buffer<uint8_t, 2>& inp, sycl::buffer<uint8_t, 2>& oup, sycl::queue q, std::pair<size_t, size_t> startPoint, std::pair<uint8_t, uint8_t> threashhold) {

    size_t height = inp.get_range()[1], width = inp.get_range()[0];

    q.submit([&](sycl::handler& h) {
        sycl::accessor oup_acc(oup, h, sycl::write_only);

        h.parallel_for(oup.get_range(), [=](sycl::id<2> i) {
            oup_acc[i] = 0;

            if (i[0] == startPoint.first && i[1] == startPoint.second) oup_acc[i] = 255;
            });
        });

    sycl::buffer<size_t> areaFilled(sycl::range<1>(1));
    areaFilled.get_host_access()[0] = 0;
    size_t prevAreaFilled;

    uint32_t wg_size = std::floor((float)std::sqrt(q.get_device().get_info<sycl::info::device::max_work_group_size>()));
    uint32_t glob_size = wg_size * std::ceil((float)inp.get_range()[0] / wg_size);

    size_t lambda = (height / wg_size) / 5;
    if (lambda > 6) lambda = 6;

    do {
        prevAreaFilled = areaFilled.get_host_access()[0];
        areaFilled.get_host_access()[0] = 0;

        for (int i = 0; i < lambda; i++) {

            q.submit([&](sycl::handler& h) {

                sycl::accessor inp_acc(inp, h, sycl::read_only);
                sycl::accessor oup_acc(oup, h, sycl::read_write);
                sycl::accessor af_acc(areaFilled, h, sycl::read_write);

                // Local scratch pad for each workgroup.
                sycl::accessor<int, 1, sycl::access::mode::read_write, sycl::access::target::local> wgp_reduct_scratch(wg_size * wg_size, h);

                h.parallel_for(sycl::nd_range<2>{{glob_size, glob_size}, { wg_size, wg_size }}, [=](sycl::nd_item<2> workItem) {
                    auto glob_i = workItem.get_global_id();
                    auto loc_i = workItem.get_local_id();
                    auto lin_loc_i = workItem.get_local_linear_id();

                    wgp_reduct_scratch[lin_loc_i] = 0;


                    for (int i = 0; i < wg_size; i++) {

                        // If not an edge pixel.
                        if (glob_i[0] > 0 && glob_i[1] > 0 && glob_i[0] < width - 1 && glob_i[1] < height - 1) {

                            // If in the fill.
                            if (oup_acc[glob_i] == 255) {
                                wgp_reduct_scratch[lin_loc_i] = 1;

                                // Test left pixel.
                                if (inp_acc[glob_i[0] - 1][glob_i[1]] >= threashhold.first && inp_acc[glob_i[0] - 1][glob_i[1]] <= threashhold.second) {
                                    oup_acc[glob_i[0] - 1][glob_i[1]] = 255;
                                }

                                // Test top pixel.
                                if (inp_acc[glob_i[0]][glob_i[1] + 1] >= threashhold.first && inp_acc[glob_i[0]][glob_i[1] + 1] <= threashhold.second) {
                                    oup_acc[glob_i[0]][glob_i[1] + 1] = 255;
                                }

                                // Test right pixel.
                                if (inp_acc[glob_i[0] + 1][glob_i[1]] >= threashhold.first && inp_acc[glob_i[0] + 1][glob_i[1]] <= threashhold.second) {
                                    oup_acc[glob_i[0] + 1][glob_i[1]] = 255;
                                }

                                // Test bottom pixel.
                                if (inp_acc[glob_i[0]][glob_i[1] - 1] >= threashhold.first && inp_acc[glob_i[0]][glob_i[1] - 1] <= threashhold.second) {
                                    oup_acc[glob_i[0]][glob_i[1] - 1] = 255;
                                }
                            }
                        }

                        // Synchronise workgroup.
                        workItem.barrier(sycl::access::fence_space::local_space);
                    }

                    // Do sum reduction over wg into the scratchpad.
                    // i is set initially to half wg_size then halved untill 0 in a loop.
                    for (int i = wg_size * wg_size / 2; i > 0; i >>= 1) {

                        // Synchronise workgroup.
                        workItem.barrier(sycl::access::fence_space::local_space);

                        // Perform the summation if in range.
                        if (lin_loc_i < i) wgp_reduct_scratch[lin_loc_i] += wgp_reduct_scratch[lin_loc_i + i];
                    }

                    // If the 0th work item in this workgroup we then submit the local wgp sum to the global value.
                    if (lin_loc_i == 0) {
                        auto v = sycl::atomic_ref<size_t, sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space>(af_acc[0]);

                        v.fetch_add(wgp_reduct_scratch[0]);
                    }
                    });
                });
        }
        q.wait();
    } while (areaFilled.get_host_access()[0] != prevAreaFilled);

}


/*          CPU codes           */


void convertToGrayscale_CPU(cv::Mat& inp, cv::Mat oup) {

    // yes I know I can use cvtColor() but I am not.
    for (size_t i = 0; i < inp.size().width; i++) {
        for (size_t j = 0; j < inp.size().height; j++) {
            oup.at<uint8_t>(j,i) =
                (inp.at<cv::Vec3b>(j, i) / 3)[0] +
                (inp.at<cv::Vec3b>(j, i) / 3)[1] +
                (inp.at<cv::Vec3b>(j, i) / 3)[2];
        }
    }
}

void sobel_CPU(cv::Mat& inp, cv::Mat& oup) {
    for (size_t i = 1; i < inp.size().width - 1; i++) {
        for (size_t j = 1; j < inp.size().height - 1; j++) {

            // Calculate the X gradient.
            int Y_Val =
                (inp.at<uint8_t>(j - 1, i - 1) * -1) + (inp.at<uint8_t>(j - 1, i) * 0) + (inp.at<uint8_t>(j - 1, i + 1) * 1) +
                (inp.at<uint8_t>(j, i - 1) * -2) + (inp.at<uint8_t>(j, i) * 0) + (inp.at<uint8_t>(j, i + 1) * 2) +
                (inp.at<uint8_t>(j + 1, i - 1) * -1) + (inp.at<uint8_t>(j + 1, i) * 0) + (inp.at<uint8_t>(j + 1, i + 1) * 1);

            int X_Val =
                (inp.at<uint8_t>(j - 1, i - 1) * -1) + (inp.at<uint8_t>(j - 1, i) * -2) + (inp.at<uint8_t>(j - 1, i + 1) * -1) +
                (inp.at<uint8_t>(j, i - 1) * 0) + (inp.at<uint8_t>(j, i) * 0) + (inp.at<uint8_t>(j, i + 1) * 0) +
                (inp.at<uint8_t>(j + 1, i - 1) * 1) + (inp.at<uint8_t>(j + 1, i) * 2) + (inp.at<uint8_t>(j + 1, i + 1) * 1);


            oup.at<uint8_t>(j,i) = (uint8_t)std::sqrt((float)(X_Val * X_Val + Y_Val * Y_Val));

        }
    }
}

void floodFill_CPU(cv::Mat& inp, cv::Mat& oup, std::pair<size_t, size_t> centre, std::pair<uint8_t, uint8_t> threashhold) {

    size_t currentFilledPixels = 0;
    size_t prevFilledPixels;

    oup.at<uint8_t>(centre.first, centre.second) = 255;

    do {
        prevFilledPixels = currentFilledPixels;
        currentFilledPixels = 0;
        for (size_t i = 1; i < inp.size().width - 1; i++) {
            for (size_t j = 1; j < inp.size().height - 1; j++) {

                // If in the fill.
                if (oup.at<uint8_t>(i, j) == 255) {
                    currentFilledPixels++;

                    // Test left pixel.
                    if (inp.at<uint8_t>(i - 1, j) >= threashhold.first && inp.at<uint8_t>(i - 1, j) <= threashhold.second) {
                        oup.at<uint8_t>(i - 1, j) = 255;
                    }

                    // Test top pixel.
                    if (inp.at<uint8_t>(i, j + 1) >= threashhold.first && inp.at<uint8_t>(i, j + 1) <= threashhold.second) {
                        oup.at<uint8_t>(i, j + 1) = 255;
                    }

                    // Test right pixel.
                    if (inp.at<uint8_t>(i + 1, j) >= threashhold.first && inp.at<uint8_t>(i + 1, j) <= threashhold.second) {
                        oup.at<uint8_t>(i + 1, j) = 255;
                    }

                    // Test bottom pixel.
                    if (inp.at<uint8_t>(i, j - 1) >= threashhold.first && inp.at<uint8_t>(i, j - 1) <= threashhold.second) {
                        oup.at<uint8_t>(i, j - 1) = 255;
                    }
                }
            }
        }
    } while (currentFilledPixels != prevFilledPixels);
}