// Function: sub_2257850
// Address: 0x2257850
//
void __fastcall sub_2257850(__int64 a1)
{
  void (*v1)(void); // rax
  __int64 v2; // rax
  int v3; // edx

  v1 = **(void (***)(void))a1;
  if ( (char *)v1 == (char *)sub_2257680 )
  {
    *(_QWORD *)a1 = off_4A082C8;
    v2 = *(_QWORD *)(a1 + 8);
    if ( (_UNKNOWN *)(v2 - 24) != &unk_4FD67C0 )
    {
      if ( &_pthread_key_create )
      {
        v3 = _InterlockedExchangeAdd((volatile signed __int32 *)(v2 - 8), 0xFFFFFFFF);
      }
      else
      {
        v3 = *(_DWORD *)(v2 - 8);
        *(_DWORD *)(v2 - 8) = v3 - 1;
      }
      if ( v3 <= 0 )
        j_j___libc_free_0_1(v2 - 24);
    }
    nullsub_806();
  }
  else
  {
    v1();
  }
}
