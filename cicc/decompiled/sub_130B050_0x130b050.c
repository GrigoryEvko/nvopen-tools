// Function: sub_130B050
// Address: 0x130b050
//
int __fastcall sub_130B050(__int64 a1, __int64 a2)
{
  *(_BYTE *)(a2 + 104) = 0;
  return pthread_mutex_unlock((pthread_mutex_t *)(a2 + 64));
}
