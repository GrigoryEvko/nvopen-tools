// Function: sub_1253F60
// Address: 0x1253f60
//
__int64 __fastcall sub_1253F60(_QWORD *a1)
{
  volatile signed __int32 *v1; // r13
  signed __int32 v2; // eax
  void (*v4)(); // rax
  signed __int32 v5; // eax
  __int64 (__fastcall *v6)(__int64); // rdx

  v1 = (volatile signed __int32 *)a1[2];
  *a1 = &unk_49E6870;
  if ( v1 )
  {
    if ( &_pthread_key_create )
    {
      v2 = _InterlockedExchangeAdd(v1 + 2, 0xFFFFFFFF);
    }
    else
    {
      v2 = *((_DWORD *)v1 + 2);
      *((_DWORD *)v1 + 2) = v2 - 1;
    }
    if ( v2 == 1 )
    {
      v4 = *(void (**)())(*(_QWORD *)v1 + 16LL);
      if ( v4 != nullsub_25 )
        ((void (__fastcall *)(volatile signed __int32 *))v4)(v1);
      if ( &_pthread_key_create )
      {
        v5 = _InterlockedExchangeAdd(v1 + 3, 0xFFFFFFFF);
      }
      else
      {
        v5 = *((_DWORD *)v1 + 3);
        *((_DWORD *)v1 + 3) = v5 - 1;
      }
      if ( v5 == 1 )
      {
        v6 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v1 + 24LL);
        if ( v6 == sub_9C26E0 )
          (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v1 + 8LL))(v1);
        else
          v6((__int64)v1);
      }
    }
  }
  return j_j___libc_free_0(a1, 64);
}
