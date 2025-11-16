// Function: sub_16C90A0
// Address: 0x16c90a0
//
bool __fastcall sub_16C90A0(pthread_rwlock_t **a1)
{
  return pthread_rwlock_unlock(*a1) == 0;
}
