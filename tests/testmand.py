# testmand
#
#

__all__ = []
__author__ = ["name"]
__email__ = ["email"]


import pyopencl as cl
import numpy as np
from PIL import Image
from decimal import Decimal

def mandel(ctx, x, y, zoom, max_iter=1000, iter_steps=1, width=500, height=500, use_double=False):
    mf = cl.mem_flags
    cl_queue = cl.CommandQueue(ctx)
    # build program
    code = """
    #if real_t == double
        #pragma OPENCL EXTENSION cl_khr_fp64 : enable
    #endif
    kernel void mandel(
        __global real_t *coords,
        __global uint *output,
        __global real_t *output_coord,
        const uint max_iter,
        const uint start_iter    
    ){
        uint id = get_global_id(0);         
        real_t2 my_coords = vload2(id, coords);           
        real_t2 my_value_coords = vload2(id, output_coord);           
        real_t x = my_value_coords.x;
        real_t y = my_value_coords.y;
        uint iter = 0;
        for(iter=start_iter; iter<max_iter; ++iter){
            if(x*x + y*y > 4.0f){
                break;
            }
            real_t xtemp = x*x - y*y + my_coords.x;
            y = 2*x*y + my_coords.y;
            x = xtemp;
        }
        // copy the current x,y pair back
        real_t2 val = (real_t2){x, y};
        vstore2(val, id, output_coord);
        output[id] = iter;
    }        
    """
    _cltype, _nptype = ("double",np.float64) if use_double else ("float", np.float32)
    prg = cl.Program(ctx, code).build("-cl-opt-disable -D real_t=%s -D real_t2=%s2" % (_cltype, _cltype))

    # Calculate the "viewport".
    x0 = x - ((Decimal(3) * zoom)/Decimal(2.))
    y0 = y - ((Decimal(2) * zoom)/Decimal(2.))
    x1 = x + ((Decimal(3) * zoom)/Decimal(2.))
    y1 = y + ((Decimal(2) * zoom)/Decimal(2.))

    # Create index map in x,y pairs
    xx = np.arange(0, width, 1, dtype=np.uint32)
    yy = np.arange(0, height, 1, dtype=np.uint32)
    index_map = np.dstack(np.meshgrid(xx, yy))
    # and local "coordinates" (real, imaginary parts)
    coord_map = np.ndarray(index_map.shape, dtype=_nptype)
    coord_map[:] = index_map
    coord_map[:] *= (_nptype((x1-x0)/Decimal(width)), _nptype((y1-y0)/Decimal(height)))
    coord_map[:] += (_nptype(x0), _nptype(y0))
    coord_map = coord_map.flatten()
    index_map = index_map.flatten().astype(dtype=np.uint32)
    # Create input and output buffer
    buffer_in_cl = cl.Buffer(ctx, mf.READ_ONLY, size=coord_map.nbytes)
    buffer_out = np.zeros(width*height, dtype=np.uint32) # This will contain the iteration values of that run
    buffer_out_cl = cl.Buffer(ctx, mf.WRITE_ONLY, size=buffer_out.nbytes)
    buffer_out_coords = np.zeros(width*height*2, dtype=_nptype) # This the last x,y values
    buffer_out_coords_cl = cl.Buffer(ctx, mf.READ_WRITE, size=buffer_out_coords.nbytes)
    # 2D Buffer to collect the iterations needed per pixel 
    #iter_map = np.zeros(width*height, dtype=np.uint32).reshape((width, height)) #.reshape((height, width))
    iter_map = np.zeros(width*height, dtype=np.uint32).reshape((height, width))

    start_max_iter = 0
    to_do = coord_map.size / 2
    steps_size = int(max_iter / float(iter_steps))
    while to_do > 0 and start_max_iter < max_iter:
        end_max_iter = min(max_iter, start_max_iter + steps_size )
        print("Iterations from iteration %i to %i for %i numbers" % (start_max_iter, end_max_iter, to_do))

        # copy x/y pairs to device 
        cl.enqueue_copy(cl_queue, buffer_in_cl, coord_map[:int(to_do*2)]).wait()        
        cl.enqueue_copy(cl_queue, buffer_out_coords_cl, buffer_out_coords[:int(to_do*2)]).wait()        
        # and finally call the ocl function
        prg.mandel(cl_queue, (int(to_do),), None,
            buffer_in_cl,                   
            buffer_out_cl,
            buffer_out_coords_cl,
            np.uint32(end_max_iter),
            np.uint32(start_max_iter)
        ).wait()
        # Copy the output back
        cl.enqueue_copy(cl_queue, buffer_out_coords, buffer_out_coords_cl).wait()
        cl.enqueue_copy(cl_queue, buffer_out, buffer_out_cl).wait()

        # Get indices of "found" escapes
        done = np.where(buffer_out[:int(to_do)]<end_max_iter)[0]
        # and write the iterations to the coresponding cell
        index_reshaped = index_map[:int(to_do*2)].reshape((int(to_do), 2))
        tmp = index_reshaped[done]
        iter_map[tmp[:,1], tmp[:,0]] = buffer_out[done]        
        #iter_map[tmp[:,0], tmp[:,1]] = buffer_out[done]        

        # Get the indices of non escapes
        undone = np.where(buffer_out[:int(to_do)]==end_max_iter)[0]
        # and write them back to our "job" maps for the next loop
        tmp = buffer_out_coords[:int(to_do*2)].reshape((int(to_do), 2))
        buffer_out_coords[:undone.size*2] = tmp[undone].flatten()
        tmp = coord_map[:int(to_do*2)].reshape((int(to_do), 2))
        coord_map[:undone.size*2] = tmp[undone].flatten()
        index_map[:undone.size*2] = index_reshaped[undone].flatten()

        to_do = undone.size
        start_max_iter = end_max_iter
        print("%i done. %i unknown" % (done.size, undone.size))

    # simple coloring by modulo 255 on the iter_map
    return (iter_map % 255).astype(np.uint8).reshape((height, width))


if __name__ == '__main__':
    ctx = cl.create_some_context(interactive=True)
    img = mandel(ctx,
          x=Decimal("-0.7546546453361122021732941811"),
          y=Decimal("0.05020518634419688663435986387"),
          zoom=Decimal("0.0002046859427855630601247281079"),
          max_iter=2000,
          iter_steps=1,
          width=500,
          height=400,
          use_double=False
    )
    Image.fromarray(img).show()


# vim: fdm=marker ts=2 sw=2 sts=2 sr et
