// Function: sub_222D6A0
// Address: 0x222d6a0
//
void __fastcall sub_222D6A0(unsigned __int64 a1)
{
  unsigned __int64 v1; // rbp
  volatile signed __int32 *v2; // r12
  signed __int32 v3; // eax
  volatile signed __int32 *v4; // rbx
  signed __int32 v5; // ett
  __int64 v6; // rdi
  volatile __int32 *v7; // rdi
  signed __int32 v8; // eax
  volatile signed __int32 *v9; // rdi
  signed __int32 v10; // eax
  void (*v11)(void); // rdx
  void (*v12)(); // rax
  signed __int32 v13; // eax
  __int64 (__fastcall *v14)(__int64); // rdx

  v1 = a1;
  v2 = *(volatile signed __int32 **)(a1 + 24);
  if ( v2 )
  {
    v3 = *((_DWORD *)v2 + 2);
    v4 = v2 + 2;
    do
    {
      if ( !v3 )
        goto LABEL_11;
      v5 = v3;
      v3 = _InterlockedCompareExchange(v4, v3 + 1, v3);
    }
    while ( v5 != v3 );
    if ( *((_DWORD *)v2 + 2)
      && (v6 = *(_QWORD *)(a1 + 16)) != 0
      && (v7 = (volatile __int32 *)(v6 + 16), _InterlockedExchange(v7, 1) < 0) )
    {
      sub_222D1B0((__int64)v7);
      if ( !&_pthread_key_create )
        goto LABEL_9;
    }
    else if ( !&_pthread_key_create )
    {
LABEL_9:
      v8 = *((_DWORD *)v2 + 2);
      *((_DWORD *)v2 + 2) = v8 - 1;
LABEL_10:
      if ( v8 == 1 )
      {
        v12 = *(void (**)())(*(_QWORD *)v2 + 16LL);
        if ( v12 != nullsub_25 )
          ((void (__fastcall *)(volatile signed __int32 *))v12)(v2);
        if ( &_pthread_key_create )
        {
          v13 = _InterlockedExchangeAdd(v2 + 3, 0xFFFFFFFF);
        }
        else
        {
          v13 = *((_DWORD *)v2 + 3);
          *((_DWORD *)v2 + 3) = v13 - 1;
        }
        if ( v13 == 1 )
        {
          v14 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v2 + 24LL);
          if ( v14 == sub_9C26E0 )
          {
            (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v2 + 8LL))(v2);
            v1 = a1;
            goto LABEL_12;
          }
          v14((__int64)v2);
        }
      }
LABEL_11:
      v1 = a1;
      goto LABEL_12;
    }
    v8 = _InterlockedExchangeAdd(v4, 0xFFFFFFFF);
    goto LABEL_10;
  }
LABEL_12:
  if ( v1 )
  {
    v9 = *(volatile signed __int32 **)(v1 + 24);
    if ( v9 )
    {
      if ( &_pthread_key_create )
      {
        v10 = _InterlockedExchangeAdd(v9 + 3, 0xFFFFFFFF);
      }
      else
      {
        v10 = *((_DWORD *)v9 + 3);
        *((_DWORD *)v9 + 3) = v10 - 1;
      }
      if ( v10 == 1 )
      {
        v11 = *(void (**)(void))(*(_QWORD *)v9 + 24LL);
        if ( (char *)v11 == (char *)sub_9C26E0 )
          (*(void (**)(void))(*(_QWORD *)v9 + 8LL))();
        else
          v11();
      }
    }
    j___libc_free_0(v1);
  }
}
