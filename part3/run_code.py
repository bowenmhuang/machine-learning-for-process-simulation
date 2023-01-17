pts = 100
data, objs, is_successes = generate_data(pts, ops)

data, objs, is_successes = 1, 1, 1
RF, a = train_RF_obj(data, objs)
plot_corr_obj(pts, RF)
RF = train_RF_successes(data, is_successes)
RF = train_RF_both(data, objs, is_successes)

test_pts = 1000
data, objs, is_successes = generate_test_data(pts)
b = test_RF_obj(RF, test_pts)
test_RF_successes(RF, test_pts)
print(a, b)


run_list = []
runs = 2
for i in range(runs):
    data, objs, is_successes = generate_data(pts)
    RF, a = train_RF_obj(data, objs)
    b = test_RF_obj(RF, test_pts)
    RF, c = train_RF_successes(data, is_successes)
    d = test_RF_successes(RF, test_pts)
    run_list.append([a, b, c, d])
with open("run_list_{}.pkl".format(pts), "wb") as f:
    pickle.dump(run_list, f)
t_list = map(list, zip(*run_list))
print(sum(t_list[0]) / runs)
print(sum(t_list[1]) / runs)
print(sum(t_list[2]) / runs)
print(sum(t_list[3]) / runs)
