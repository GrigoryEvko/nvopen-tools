// Function: .__sched_cpucount
// Address: 0x406e30
//
// attributes: thunk
int __sched_cpucount(size_t setsize, const cpu_set_t *setp)
{
  return _sched_cpucount(setsize, setp);
}
