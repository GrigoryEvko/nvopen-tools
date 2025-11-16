// Function: sub_3702CE0
// Address: 0x3702ce0
//
void __fastcall sub_3702CE0(unsigned __int64 *a1)
{
  unsigned __int64 *v2; // rax
  unsigned __int64 v3; // rdi
  volatile signed __int32 *v4; // r12
  signed __int32 v5; // eax
  unsigned __int64 v6; // rdi
  void (*v7)(); // rax
  signed __int32 v8; // eax
  __int64 (__fastcall *v9)(__int64); // rdx

  v2 = a1 + 22;
  v3 = a1[20];
  if ( (unsigned __int64 *)v3 != v2 )
    _libc_free(v3);
  v4 = (volatile signed __int32 *)a1[12];
  a1[10] = (unsigned __int64)&unk_4A352E0;
  if ( v4 )
  {
    if ( &_pthread_key_create )
    {
      v5 = _InterlockedExchangeAdd(v4 + 2, 0xFFFFFFFF);
    }
    else
    {
      v5 = *((_DWORD *)v4 + 2);
      *((_DWORD *)v4 + 2) = v5 - 1;
    }
    if ( v5 == 1 )
    {
      v7 = *(void (**)())(*(_QWORD *)v4 + 16LL);
      if ( v7 != nullsub_25 )
        ((void (__fastcall *)(volatile signed __int32 *))v7)(v4);
      if ( &_pthread_key_create )
      {
        v8 = _InterlockedExchangeAdd(v4 + 3, 0xFFFFFFFF);
      }
      else
      {
        v8 = *((_DWORD *)v4 + 3);
        *((_DWORD *)v4 + 3) = v8 - 1;
      }
      if ( v8 == 1 )
      {
        v9 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v4 + 24LL);
        if ( v9 == sub_9C26E0 )
          (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v4 + 8LL))(v4);
        else
          v9((__int64)v4);
      }
    }
  }
  v6 = a1[6];
  a1[5] = (unsigned __int64)&unk_4A3C600;
  if ( v6 )
    j_j___libc_free_0(v6);
  if ( (unsigned __int64 *)*a1 != a1 + 2 )
    _libc_free(*a1);
}
