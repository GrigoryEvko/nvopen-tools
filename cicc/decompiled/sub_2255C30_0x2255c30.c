// Function: sub_2255C30
// Address: 0x2255c30
//
void __fastcall sub_2255C30(_QWORD *a1)
{
  __int64 v1; // rax
  int v2; // edx

  *a1 = off_4A08040;
  v1 = a1[1];
  if ( (_UNKNOWN *)(v1 - 24) != &unk_4FD67C0 )
  {
    if ( &_pthread_key_create )
    {
      v2 = _InterlockedExchangeAdd((volatile signed __int32 *)(v1 - 8), 0xFFFFFFFF);
    }
    else
    {
      v2 = *(_DWORD *)(v1 - 8);
      *(_DWORD *)(v1 - 8) = v2 - 1;
    }
    if ( v2 <= 0 )
      j_j___libc_free_0_1(v1 - 24);
  }
  nullsub_806();
}
