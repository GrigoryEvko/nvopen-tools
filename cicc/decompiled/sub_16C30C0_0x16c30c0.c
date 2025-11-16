// Function: sub_16C30C0
// Address: 0x16c30c0
//
bool __fastcall sub_16C30C0(pthread_mutex_t **a1)
{
  return pthread_mutex_lock(*a1) == 0;
}
