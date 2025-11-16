// Function: .sched_setaffinity
// Address: 0x406840
//
// attributes: thunk
int sched_setaffinity(__pid_t pid, size_t cpusetsize, const cpu_set_t *cpuset)
{
  return sched_setaffinity(pid, cpusetsize, cpuset);
}
