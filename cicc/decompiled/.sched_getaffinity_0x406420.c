// Function: .sched_getaffinity
// Address: 0x406420
//
// attributes: thunk
int sched_getaffinity(__pid_t pid, size_t cpusetsize, cpu_set_t *cpuset)
{
  return sched_getaffinity(pid, cpusetsize, cpuset);
}
