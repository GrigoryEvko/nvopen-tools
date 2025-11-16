// Function: sub_9D04A0
// Address: 0x9d04a0
//
__int64 __fastcall sub_9D04A0(__int64 a1, __int64 a2)
{
  __int64 v3; // rsi
  __int64 v4; // rax
  __int64 v5; // rcx
  __int64 v6; // r15
  __int64 v7; // rcx
  __int64 v8; // rdx
  __int64 v9; // r13
  __int64 v10; // rdi
  __int64 v11; // r14
  __int64 v12; // rbx
  volatile signed __int32 *v13; // rdi
  signed __int32 v14; // eax
  void (*v15)(void); // rax
  signed __int32 v16; // eax
  void (*v17)(void); // rdx
  int v18; // ebx
  __int64 v20; // [rsp+8h] [rbp-58h]
  __int64 v21; // [rsp+10h] [rbp-50h]
  __int64 v22[7]; // [rsp+28h] [rbp-38h] BYREF

  v21 = a1 + 16;
  v20 = sub_C8D7D0(a1, a1 + 16, a2, 32, v22);
  v3 = v20;
  v4 = *(_QWORD *)a1;
  v5 = 32LL * *(unsigned int *)(a1 + 8);
  v6 = *(_QWORD *)a1 + v5;
  if ( *(_QWORD *)a1 != v6 )
  {
    v7 = v20 + v5;
    v8 = v20;
    do
    {
      if ( v8 )
      {
        *(_DWORD *)v8 = *(_DWORD *)v4;
        *(_QWORD *)(v8 + 8) = *(_QWORD *)(v4 + 8);
        *(_QWORD *)(v8 + 16) = *(_QWORD *)(v4 + 16);
        v3 = *(_QWORD *)(v4 + 24);
        *(_QWORD *)(v8 + 24) = v3;
        *(_QWORD *)(v4 + 24) = 0;
        *(_QWORD *)(v4 + 16) = 0;
        *(_QWORD *)(v4 + 8) = 0;
      }
      v8 += 32;
      v4 += 32;
    }
    while ( v8 != v7 );
    v9 = *(_QWORD *)a1;
    v6 = *(_QWORD *)a1 + 32LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v6 )
    {
      while ( 1 )
      {
        v10 = *(_QWORD *)(v6 - 24);
        v11 = *(_QWORD *)(v6 - 16);
        v6 -= 32;
        v12 = v10;
        if ( v11 != v10 )
          break;
LABEL_22:
        if ( v10 )
        {
          v3 = *(_QWORD *)(v6 + 24) - v10;
          j_j___libc_free_0(v10, v3);
        }
        if ( v6 == v9 )
        {
          v6 = *(_QWORD *)a1;
          goto LABEL_26;
        }
      }
      while ( 1 )
      {
        v13 = *(volatile signed __int32 **)(v12 + 8);
        if ( !v13 )
          goto LABEL_9;
        if ( &_pthread_key_create )
        {
          v14 = _InterlockedExchangeAdd(v13 + 2, 0xFFFFFFFF);
        }
        else
        {
          v14 = *((_DWORD *)v13 + 2);
          v3 = (unsigned int)(v14 - 1);
          *((_DWORD *)v13 + 2) = v3;
        }
        if ( v14 != 1 )
          goto LABEL_9;
        v15 = *(void (**)(void))(*(_QWORD *)v13 + 16LL);
        if ( v15 != nullsub_25 )
          v15();
        if ( &_pthread_key_create )
        {
          v16 = _InterlockedExchangeAdd(v13 + 3, 0xFFFFFFFF);
        }
        else
        {
          v16 = *((_DWORD *)v13 + 3);
          *((_DWORD *)v13 + 3) = v16 - 1;
        }
        if ( v16 != 1 )
          goto LABEL_9;
        v17 = *(void (**)(void))(*(_QWORD *)v13 + 24LL);
        if ( (char *)v17 == (char *)sub_9C26E0 )
        {
          (*(void (**)(void))(*(_QWORD *)v13 + 8LL))();
          v12 += 16;
          if ( v11 == v12 )
          {
LABEL_21:
            v10 = *(_QWORD *)(v6 + 8);
            goto LABEL_22;
          }
        }
        else
        {
          v17();
LABEL_9:
          v12 += 16;
          if ( v11 == v12 )
            goto LABEL_21;
        }
      }
    }
  }
LABEL_26:
  v18 = v22[0];
  if ( v21 != v6 )
    _libc_free(v6, v3);
  *(_DWORD *)(a1 + 12) = v18;
  *(_QWORD *)a1 = v20;
  return v20;
}
