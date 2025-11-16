// Function: sub_22091A0
// Address: 0x22091a0
//
__int64 __fastcall sub_22091A0(volatile signed __int64 *a1)
{
  volatile signed __int64 v1; // rax
  signed __int32 v3; // eax

  v1 = *a1;
  if ( *a1 )
    return v1 - 1;
  if ( &_pthread_key_create )
  {
    _InterlockedCompareExchange64(a1, _InterlockedIncrement(&dword_4FD4F38), 0);
    v1 = *a1;
    return v1 - 1;
  }
  v3 = ++dword_4FD4F38;
  *a1 = dword_4FD4F38;
  return v3 - 1LL;
}
