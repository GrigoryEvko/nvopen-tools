// Function: sub_2211A20
// Address: 0x2211a20
//
void __fastcall sub_2211A20(__int64 *a1)
{
  __int64 v1; // rax
  void *v2; // rdi
  int v3; // edx

  v1 = *a1;
  v2 = (void *)(*a1 - 24);
  if ( v2 != &unk_4FD67E0 )
  {
    if ( &_pthread_key_create )
    {
      v3 = _InterlockedExchangeAdd((volatile signed __int32 *)(v1 - 8), 0xFFFFFFFF);
    }
    else
    {
      v3 = *(_DWORD *)(v1 - 8);
      *(_DWORD *)(v1 - 8) = v3 - 1;
    }
    if ( v3 <= 0 )
      j_j___libc_free_0_2((unsigned __int64)v2);
  }
}
