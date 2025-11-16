// Function: sub_2208EC0
// Address: 0x2208ec0
//
void __fastcall sub_2208EC0(_QWORD *a1)
{
  unsigned __int64 v2; // rdi
  unsigned __int64 v3; // rbx
  volatile signed __int32 *v4; // r12
  signed __int32 v5; // eax
  void (__fastcall *v6)(unsigned __int64); // rax
  unsigned __int64 v7; // rdi
  unsigned __int64 v8; // rbx
  volatile signed __int32 *v9; // r12
  signed __int32 v10; // eax
  void (__fastcall *v11)(unsigned __int64); // rax
  unsigned __int64 v12; // r8
  __int64 v13; // rbx
  unsigned __int64 v14; // rdi

  v2 = a1[1];
  if ( v2 )
  {
    if ( !a1[2] )
      goto LABEL_13;
    v3 = 0;
    do
    {
      v4 = *(volatile signed __int32 **)(v2 + 8 * v3);
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
          v6 = *(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v4 + 8LL);
          if ( v6 == sub_2208CE0 )
          {
            nullsub_801();
            j___libc_free_0((unsigned __int64)v4);
          }
          else
          {
            v6((unsigned __int64)v4);
          }
        }
        v2 = a1[1];
      }
      ++v3;
    }
    while ( a1[2] > v3 );
    if ( v2 )
LABEL_13:
      j_j___libc_free_0_0(v2);
  }
  v7 = a1[3];
  if ( v7 )
  {
    if ( !a1[2] )
      goto LABEL_26;
    v8 = 0;
    do
    {
      v9 = *(volatile signed __int32 **)(v7 + 8 * v8);
      if ( v9 )
      {
        if ( &_pthread_key_create )
        {
          v10 = _InterlockedExchangeAdd(v9 + 2, 0xFFFFFFFF);
        }
        else
        {
          v10 = *((_DWORD *)v9 + 2);
          *((_DWORD *)v9 + 2) = v10 - 1;
        }
        if ( v10 == 1 )
        {
          v11 = *(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v9 + 8LL);
          if ( v11 == sub_2208CE0 )
          {
            nullsub_801();
            j___libc_free_0((unsigned __int64)v9);
          }
          else
          {
            v11((unsigned __int64)v9);
          }
        }
        v7 = a1[3];
      }
      ++v8;
    }
    while ( a1[2] > v8 );
    if ( v7 )
LABEL_26:
      j_j___libc_free_0_0(v7);
  }
  v12 = a1[4];
  v13 = 0;
  if ( v12 )
  {
    do
    {
      v14 = *(_QWORD *)(v12 + v13);
      if ( v14 )
      {
        j_j___libc_free_0_0(v14);
        v12 = a1[4];
      }
      v13 += 8;
    }
    while ( v13 != 96 );
    if ( v12 )
      j_j___libc_free_0_0(v12);
  }
}
