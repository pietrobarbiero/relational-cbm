import os.path
import torch
import numpy as np


def hanoi_toy_dataset(mode, n_samples=2, random_seed = 42, n_disks = 3, n_boards = 1, n_positions = 10, n_sizes = 5, load = True):
    np.random.seed(random_seed)

    folder = os.path.dirname(__file__)
    dataset_name = f'hanoi_{mode}_{n_samples}_{random_seed}_{n_disks}_{n_boards}_{n_positions}_{n_sizes}.pt'
    dataset_name = os.path.join(folder, dataset_name)

    if mode == 'flat':
        if os.path.exists(dataset_name) and load:
            X, c, y = torch.load(dataset_name)
            return X, c, y

        X, c, y = [], [], []
        positions = np.arange(n_positions)
        sizes = np.arange(n_sizes)

        for board_id in range(n_boards):
            for sample_id in range(n_samples):
                # disk_positions = np.random.permutation(positions)[:n_disks]
                i = np.random.randint(0, len(positions) - n_disks)
                disk_positions = positions[i:i + n_disks]
                disk_sizes = np.random.permutation(sizes)[:n_disks]

                # sort disks from lowest to highest position
                disk_positions_ascending = np.argsort(disk_positions)
                disk_positions = disk_positions[disk_positions_ascending]
                disk_sizes = disk_sizes[disk_positions_ascending]

                disk_positions_noise = np.random.rand(n_disks) + disk_positions
                disk_sizes_noise = np.random.rand(n_disks) + disk_sizes

                for first_disk_local_id in range(n_disks):
                    disk_position = disk_positions_noise[first_disk_local_id]
                    disk_size = disk_sizes_noise[first_disk_local_id]
                    correct_i = True

                    # disks to check
                    if first_disk_local_id == 0:
                        check_disk_ids = [first_disk_local_id+1]
                    elif first_disk_local_id == n_disks-1:
                        check_disk_ids = [first_disk_local_id-1]
                    else:
                        check_disk_ids = [first_disk_local_id-1, first_disk_local_id+1]

                    for check_disk_id in check_disk_ids:
                        check_disk_position = disk_positions_noise[check_disk_id]
                        check_disk_size = disk_sizes_noise[check_disk_id]

                        first_over_second = int(disk_position > check_disk_position)
                        first_larger_second = int(disk_size > check_disk_size)
                        if (first_over_second and first_larger_second) or (not first_over_second and not first_larger_second):
                            correct_i = False
                            break

                    X.append([disk_position, disk_size])
                    one_hot_pos = np.zeros(n_positions)
                    one_hot_pos[int(disk_positions[first_disk_local_id])] = 1
                    one_hot_size = np.zeros(n_sizes)
                    one_hot_size[int(disk_sizes[first_disk_local_id])] = 1
                    c.append([*one_hot_pos, *one_hot_size])
                    y.append(int(correct_i))

        X = torch.FloatTensor(np.array(X))
        c_labels = torch.FloatTensor(np.array(c))
        y_labels = torch.FloatTensor(np.array(y)).unsqueeze(1)
        torch.save((X, c_labels, y_labels), dataset_name)

        return X, c_labels, y_labels

    else:
        if os.path.exists(dataset_name) and load:
            X, label_concepts, label_tasks, q_names, tower_ids = torch.load(dataset_name)
            return X, label_concepts, label_tasks, q_names, tower_ids

        X, q_concepts_top, q_concepts_larger, q_tasks = [], [], [], []
        tasks = []
        concepts_top = []
        concepts_larger = []
        tower_ids = []
        positions = np.arange(n_positions)
        sizes = np.arange(n_sizes)

        for board_id in range(n_boards):
            unique_disk_id = 0
            sample_id = 0
            while sample_id < n_samples:
                unique_disk_id_temp = unique_disk_id
                # disk_positions = np.random.permutation(positions)[:n_disks]
                i = np.random.randint(0, len(positions) - n_disks)
                disk_positions = positions[i:i + n_disks]
                disk_sizes = np.random.permutation(sizes)[:n_disks]

                # sort disks from lowest to highest position
                disk_positions_ascending = np.argsort(disk_positions)
                disk_positions = disk_positions[disk_positions_ascending]
                disk_sizes = disk_sizes[disk_positions_ascending]

                disk_positions_noise = 0.1*np.random.rand(n_disks) + disk_positions
                disk_sizes_noise = 0.1*np.random.rand(n_disks) + disk_sizes

                # disk_positions_noise =  disk_positions
                # disk_sizes_noise =  disk_sizes

                global_disk_ids = np.arange(n_disks) + unique_disk_id

                x, q_c_top, q_c_larger, q_t, t, c_top, c_larger = [], [], [], [], [], [], []

                for first_disk_local_id in range(n_disks):
                    first_disk_global_id = unique_disk_id
                    disk_position = disk_positions_noise[first_disk_local_id]
                    disk_size = disk_sizes_noise[first_disk_local_id]

                    # append individual features of disk
                    x.append([disk_position, disk_size])
                    # one_hot_pos = np.zeros(n_positions)
                    # one_hot_pos[int(disk_positions[first_disk_local_id])] = 1
                    # one_hot_size = np.zeros(n_sizes)
                    # one_hot_size[int(disk_sizes[first_disk_local_id])] = 1
                    # q.extend([*one_hot_pos, *one_hot_size])
                    # c.extend([f'level{p}({unique_disk_id})' for p in positions])
                    # c.extend([f'size{s}({unique_disk_id})' for s in sizes])

                    # disks to check
                    if first_disk_local_id == 0:
                        check_disk_ids = [first_disk_local_id+1]
                    elif first_disk_local_id == n_disks-1:
                        check_disk_ids = [first_disk_local_id-1]
                    else:
                        check_disk_ids = [first_disk_local_id-1, first_disk_local_id+1]

                    # now check relational features
                    # all pairs
                    for second_disk_local_id, second_disk_global_id in enumerate(global_disk_ids):
                        if second_disk_global_id != first_disk_global_id:
                            # print(unique_disk_id, f'top({unique_disk_id},{second_disk_global_id})', f'larger({unique_disk_id},{second_disk_global_id})')
                            check_disk_position = disk_positions_noise[second_disk_local_id]
                            check_disk_size = disk_sizes_noise[second_disk_local_id]

                            if second_disk_local_id in check_disk_ids:
                                first_over_second = int(disk_position > check_disk_position)
                            else:
                                first_over_second = 0

                            first_larger_second = int(disk_size > check_disk_size)

                            q_c_top.append(int(first_over_second))
                            c_top.append(f'top({first_disk_global_id},{second_disk_global_id})')
                            q_c_larger.append(int(first_larger_second))
                            c_larger.append(f'larger({first_disk_global_id},{second_disk_global_id})')


                    if len(check_disk_ids)==2:
                        correct_i = True
                        for check_disk_id in check_disk_ids: #two maximum
                            check_disk_size = disk_sizes_noise[check_disk_id]

                            if (first_disk_local_id > check_disk_id) and (disk_size > check_disk_size): # if you are on top and larger
                                correct_i = False
                                break

                            if (first_disk_local_id < check_disk_id) and (disk_size < check_disk_size): # if you are below and smaller
                                correct_i = False
                                break
                    # else:
                    #     correct_i = False

                        q_t.append(int(correct_i))
                        t.append(f'correct({unique_disk_id})')



                    unique_disk_id += 1

                num_ones = sum(q_tasks)
                num_zeros = len(q_tasks) - sum(q_tasks)

                num_new_ones = sum(q_t)
                num_new_zeros = len(q_t) - sum(q_t)

                if np.abs((num_zeros + num_new_zeros)  - (num_ones + num_new_ones)) > 2:
                    unique_disk_id = unique_disk_id_temp
                else:
                    # tp = np.zeros([n_positions, (n_sizes+1) * 2-1])
                    # for p, s in x:
                    #     tp[p][n_sizes - (s+1): n_sizes + (s+2)] = 1
                    # print(tp[::-1])
                    # all_concepts = np.array(c_top + c_larger)
                    # all_concepts_labels = np.array(q_c_top + q_c_larger)
                    # all_tasks = np.array(t)
                    # all_tasks_labels = np.array(q_t)
                    # print(all_concepts[all_concepts_labels>0])
                    # print(all_tasks[all_tasks_labels>0])
                    # input()
                    sample_id+=1
                    tower_ids.append([str(uid) for uid in global_disk_ids])
                    X.extend(x)
                    q_tasks.extend(q_t)
                    q_concepts_top.extend(q_c_top)
                    q_concepts_larger.extend(q_c_larger)
                    tasks.extend(t)
                    concepts_top.extend(c_top)
                    concepts_larger.extend(c_larger)

        q_names = {
            'tasks': tasks,
            'concepts': concepts_top + concepts_larger,
        }
        X = torch.FloatTensor(np.array(X)).unsqueeze(0)
        label_concepts = torch.FloatTensor(np.array(q_concepts_top + q_concepts_larger)).unsqueeze(0)
        label_tasks = torch.FloatTensor(np.array(q_tasks)).unsqueeze(0)
        torch.save((X, label_concepts, label_tasks, q_names, tower_ids), dataset_name)

        return X, label_concepts, label_tasks, q_names, tower_ids



if __name__ == '__main__':
    # flat
    X, c, y = hanoi_toy_dataset(mode="flat", n_samples=1000, random_seed=1, load=True)
    # X_test, c_test, y_test = hanoi_toy_dataset(mode="flat", n_samples=1000, random_seed=2, load=True)
    # n_features = X.shape[1]
    # n_concepts = c.shape[1]
    # n_classes = y.shape[1]
    # X_rel = X.view(-1, 3 * n_features)
    # X_rel_test = X_test.view(-1, 3 * n_features)
    # c_rel = c.view(-1, 3 * n_concepts)
    # c_rel_test = c_test.view(-1, 3 * n_concepts)
    # y_rel = y.view(-1, 3 * n_classes)
    # y_rel_test = y_test.view(-1, 3 * n_classes)
    #
    # emb_size = 100
    #
    # model = torch.nn.Sequential(
    #     torch.nn.Linear(n_features, emb_size),
    #     torch.nn.LeakyReLU(),
    #     torch.nn.Linear(emb_size, emb_size),
    #     torch.nn.LeakyReLU(),
    #     torch.nn.Linear(emb_size, n_concepts)
    # )
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # criterion = torch.nn.BCEWithLogitsLoss()
    # for i in range(1000):
    #     c_pred = model(X)
    #     loss = criterion(c_pred, c)
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    #     accuracy = f1_score(c_pred > 0., c > 0.5, average="micro")
    #
    #     c_pred_test = model(X_test)
    #     test_accuracy = f1_score(c_pred_test > 0., c_test > 0.5, average="micro")
    #     print(f"Flat Concept Train {i}) loss {loss.item():.2f}, acc{accuracy:.2f} - test acc{test_accuracy:.2f}")
    #
    # print()
    #
    # # relational
    # model_rel = torch.nn.Sequential(
    #     torch.nn.Linear(n_features * 3, emb_size),
    #     torch.nn.LeakyReLU(),
    #     torch.nn.Linear(emb_size, emb_size),
    #     torch.nn.LeakyReLU(),
    #     torch.nn.Linear(emb_size, n_concepts * 3)
    # )
    # optimizer_rel = torch.optim.Adam(model_rel.parameters(), lr=0.001)
    # criterion_rel = torch.nn.BCEWithLogitsLoss()
    # for i in range(1000):
    #     c_pred_rel = model_rel(X_rel)
    #     loss_rel = criterion_rel(c_pred_rel, c_rel)
    #     optimizer_rel.zero_grad()
    #     loss_rel.backward()
    #     optimizer_rel.step()
    #     accuracy_rel = f1_score(c_pred_rel > 0., c_rel > 0.5, average="micro")
    #
    #     c_pred_rel_test = model_rel(X_rel_test)
    #     test_accuracy_rel = f1_score(c_pred_rel_test > 0., c_rel_test > 0.5, average="micro")
    #     print(
    #         f"Relational Concept Train {i}) loss {loss_rel.item():.2f}, acc{accuracy_rel:.2f} - test acc{test_accuracy_rel:.2f}")
    #
    # print()
    #
    # model = torch.nn.Sequential(
    #     torch.nn.Linear(n_features, emb_size),
    #     torch.nn.LeakyReLU(),
    #     torch.nn.Linear(emb_size, emb_size),
    #     torch.nn.LeakyReLU(),
    #     torch.nn.Linear(emb_size, n_classes)
    # )
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # criterion = torch.nn.BCEWithLogitsLoss()
    # for i in range(1000):
    #     y_pred = model(X)
    #     loss = criterion(y_pred, y)
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    #     accuracy = f1_score(y_pred > 0., y > 0.5, average="micro")
    #
    #     y_pred_test = model(X_test)
    #     test_accuracy = f1_score(y_pred_test > 0., y_test > 0.5, average="micro")
    #     print(f"Flat Task Train {i}) loss {loss.item():.2f}, task acc{accuracy:.2f} - task test acc{test_accuracy:.2f}")
    #
    # print()
    #
    # # relational
    # model_rel = torch.nn.Sequential(
    #     torch.nn.Linear(n_features * 3, emb_size),
    #     torch.nn.LeakyReLU(),
    #     torch.nn.Linear(emb_size, emb_size),
    #     torch.nn.LeakyReLU(),
    #     torch.nn.Linear(emb_size, n_classes * 3)
    # )
    # optimizer_rel = torch.optim.Adam(model_rel.parameters(), lr=0.001)
    # criterion_rel = torch.nn.BCEWithLogitsLoss()
    # for i in range(1000):
    #     y_pred_rel = model_rel(X_rel)
    #     loss_rel = criterion_rel(y_pred_rel, y_rel)
    #     optimizer_rel.zero_grad()
    #     loss_rel.backward()
    #     optimizer_rel.step()
    #     accuracy_rel = f1_score(y_pred_rel > 0., y_rel > 0.5, average="micro")
    #
    #     y_pred_rel_test = model_rel(X_rel_test)
    #     test_accuracy_rel = f1_score(y_pred_rel_test > 0., y_rel_test > 0.5, average="micro")
    #     print(
    #         f"Relational Task Train {i}) loss {loss_rel.item():.2f}, task rel acc{accuracy_rel:.2f} - task rel test acc{test_accuracy_rel:.2f}")
    #
    # print()
    #
    # # relational concept -> task
    # model = torch.nn.Sequential(
    #     torch.nn.Linear(n_concepts, emb_size),
    #     torch.nn.LeakyReLU(),
    #     torch.nn.Linear(emb_size, emb_size),
    #     torch.nn.LeakyReLU(),
    #     torch.nn.Linear(emb_size, n_classes)
    # )
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # criterion = torch.nn.BCEWithLogitsLoss()
    # for i in range(1000):
    #     y_pred = model(c)
    #     loss = criterion(y_pred, y)
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    #     accuracy = f1_score(y_pred > 0., y > 0.5, average="micro")
    #
    #     y_pred_test = model(c_test)
    #     test_accuracy = f1_score(y_pred_test > 0., y_test > 0.5, average="micro")
    #     print(
    #         f"Flat Concept - Task Train {i}) loss {loss.item():.2f}, task acc{accuracy:.2f} - task test acc{test_accuracy:.2f}")
    #
    # model_rel = torch.nn.Sequential(
    #     torch.nn.Linear(n_concepts * 3, emb_size),
    #     torch.nn.LeakyReLU(),
    #     torch.nn.Linear(emb_size, emb_size),
    #     torch.nn.LeakyReLU(),
    #     torch.nn.Linear(emb_size, n_classes * 3)
    # )
    # optimizer_rel = torch.optim.Adam(model_rel.parameters(), lr=0.001)
    # criterion_rel = torch.nn.BCEWithLogitsLoss()
    # for i in range(1000):
    #     y_pred_rel = model_rel(c_rel)
    #     loss_rel = criterion_rel(y_pred_rel, y_rel)
    #     optimizer_rel.zero_grad()
    #     loss_rel.backward()
    #     optimizer_rel.step()
    #     accuracy_rel = f1_score(y_pred_rel > 0., y_rel > 0.5, average="micro")
    #
    #     y_pred_rel_test = model_rel(c_rel_test)
    #     test_accuracy_rel = f1_score(y_pred_rel_test > 0., y_rel_test > 0.5, average="micro")
    #     print(
    #         f"Relational Concept - Task Train {i}) loss {loss_rel.item():.2f}, task rel acc{accuracy_rel:.2f} - task rel test acc{test_accuracy_rel:.2f}")
    #
    # print()