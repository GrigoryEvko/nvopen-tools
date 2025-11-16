// Function: sub_9C90D0
// Address: 0x9c90d0
//
void __fastcall sub_9C90D0(_QWORD *a1)
{
  _QWORD *v1; // rbx
  _QWORD *v2; // r15
  __int64 v3; // r13
  __int64 v4; // r12
  __int64 v5; // rdi
  _QWORD *v6; // rdi
  __int64 v7; // r14
  __int64 v8; // r12
  volatile signed __int32 *v9; // r13
  signed __int32 v10; // eax
  void (*v11)(); // rax
  signed __int32 v12; // eax
  __int64 (__fastcall *v13)(__int64); // rdx

  v1 = (_QWORD *)a1[1];
  v2 = (_QWORD *)*a1;
  if ( v1 == (_QWORD *)*a1 )
    goto LABEL_30;
  do
  {
    v3 = v2[9];
    v4 = v2[8];
    if ( v3 != v4 )
    {
      do
      {
        v5 = *(_QWORD *)(v4 + 8);
        if ( v5 != v4 + 24 )
          j_j___libc_free_0(v5, *(_QWORD *)(v4 + 24) + 1LL);
        v4 += 40;
      }
      while ( v3 != v4 );
      v4 = v2[8];
    }
    if ( v4 )
      j_j___libc_free_0(v4, v2[10] - v4);
    v6 = (_QWORD *)v2[4];
    if ( v6 != v2 + 6 )
      j_j___libc_free_0(v6, v2[6] + 1LL);
    v7 = v2[2];
    v8 = v2[1];
    if ( v7 != v8 )
    {
      while ( 1 )
      {
        v9 = *(volatile signed __int32 **)(v8 + 8);
        if ( !v9 )
          goto LABEL_13;
        if ( &_pthread_key_create )
        {
          v10 = _InterlockedExchangeAdd(v9 + 2, 0xFFFFFFFF);
        }
        else
        {
          v10 = *((_DWORD *)v9 + 2);
          *((_DWORD *)v9 + 2) = v10 - 1;
        }
        if ( v10 != 1 )
          goto LABEL_13;
        v11 = *(void (**)())(*(_QWORD *)v9 + 16LL);
        if ( v11 != nullsub_25 )
          ((void (__fastcall *)(volatile signed __int32 *))v11)(v9);
        if ( &_pthread_key_create )
        {
          v12 = _InterlockedExchangeAdd(v9 + 3, 0xFFFFFFFF);
        }
        else
        {
          v12 = *((_DWORD *)v9 + 3);
          *((_DWORD *)v9 + 3) = v12 - 1;
        }
        if ( v12 != 1 )
          goto LABEL_13;
        v13 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v9 + 24LL);
        if ( v13 == sub_9C26E0 )
        {
          (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v9 + 8LL))(v9);
          v8 += 16;
          if ( v7 == v8 )
          {
LABEL_25:
            v8 = v2[1];
            break;
          }
        }
        else
        {
          v13((__int64)v9);
LABEL_13:
          v8 += 16;
          if ( v7 == v8 )
            goto LABEL_25;
        }
      }
    }
    if ( v8 )
      j_j___libc_free_0(v8, v2[3] - v8);
    v2 += 11;
  }
  while ( v1 != v2 );
  v2 = (_QWORD *)*a1;
LABEL_30:
  if ( v2 )
    j_j___libc_free_0(v2, a1[2] - (_QWORD)v2);
}
