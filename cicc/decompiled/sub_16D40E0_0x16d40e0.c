// Function: sub_16D40E0
// Address: 0x16d40e0
//
int __fastcall sub_16D40E0(__int64 a1, const void *a2)
{
  return pthread_setspecific(*(_DWORD *)(a1 + 8), a2);
}
