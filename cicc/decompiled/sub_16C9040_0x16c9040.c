// Function: sub_16C9040
// Address: 0x16c9040
//
bool __fastcall sub_16C9040(pthread_rwlock_t **a1)
{
  return pthread_rwlock_rdlock(*a1) == 0;
}
