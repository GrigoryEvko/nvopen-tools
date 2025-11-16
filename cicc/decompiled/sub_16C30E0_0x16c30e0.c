// Function: sub_16C30E0
// Address: 0x16c30e0
//
bool __fastcall sub_16C30E0(pthread_mutex_t **a1)
{
  return pthread_mutex_unlock(*a1) == 0;
}
