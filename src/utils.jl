function get_constraints_from_box_bounds(bounds)
    function (y)
        mapreduce(vcat, [(bounds.lb, 1), (bounds.ub, -1)]) do (bound, sign)
            # drop constraints for unbounded variables
            mask = (!isinf).(bound)
            sign * (y[mask] - bound[mask])
        end
    end
end
