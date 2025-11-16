// Function: sub_16C9080
// Address: 0x16c9080
//
bool __fastcall sub_16C9080(pthread_rwlock_t **a1)
{
  return pthread_rwlock_wrlock(*a1) == 0;
}
