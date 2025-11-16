// Function: .sem_init
// Address: 0x4067a0
//
// attributes: thunk
int sem_init(sem_t *sem, int pshared, unsigned int value)
{
  return sem_init(sem, pshared, value);
}
