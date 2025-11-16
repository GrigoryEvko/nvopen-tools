// Function: sub_16C9060
// Address: 0x16c9060
//
bool __fastcall sub_16C9060(pthread_rwlock_t **a1)
{
  return pthread_rwlock_unlock(*a1) == 0;
}
