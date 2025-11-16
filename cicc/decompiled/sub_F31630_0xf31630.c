// Function: sub_F31630
// Address: 0xf31630
//
__int64 __fastcall sub_F31630(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  char **v7; // rsi
  __int64 v9; // rcx
  __int64 v10; // r9
  __int64 v11; // r13
  __int64 v12; // rdx
  __int64 v13; // r12
  __int64 v14; // r14
  __int64 v15; // r15
  __int64 v16; // r8
  __int64 v17; // r13
  __int64 v18; // rdi
  __int64 v19; // r14
  __int64 v20; // r15
  __int64 v21; // rdi
  int v22; // r13d
  __int64 v24; // [rsp+8h] [rbp-58h]
  __int64 v25; // [rsp+10h] [rbp-50h]
  unsigned __int64 v26[7]; // [rsp+28h] [rbp-38h] BYREF

  v7 = (char **)(a1 + 16);
  v25 = a1 + 16;
  v24 = sub_C8D7D0(a1, a1 + 16, a2, 0x28u, v26, a6);
  v11 = v24;
  v12 = 5LL * *(unsigned int *)(a1 + 8);
  v13 = *(_QWORD *)a1 + 40LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v13 )
  {
    v14 = *(_QWORD *)a1;
    do
    {
      while ( 1 )
      {
        v15 = 40;
        if ( v11 )
        {
          v16 = v11 + 16;
          *(_DWORD *)(v11 + 8) = 0;
          *(_QWORD *)v11 = v11 + 16;
          *(_DWORD *)(v11 + 12) = 0;
          if ( *(_DWORD *)(v14 + 8) )
          {
            v7 = (char **)v14;
            sub_F312A0(v11, v14, v12, v9, v16, v10);
            v16 = v11 + 16;
          }
          v15 = v11 + 40;
          *(_QWORD *)(v11 + 24) = 0;
          *(_QWORD *)(v11 + 16) = v11 + 40;
          *(_QWORD *)(v11 + 32) = 0;
          if ( *(_QWORD *)(v14 + 24) )
            break;
        }
        v14 += 40;
        v11 = v15;
        if ( v13 == v14 )
          goto LABEL_9;
      }
      v7 = (char **)(v14 + 16);
      v14 += 40;
      v11 += 40;
      sub_F2F930(v16, v7, v12, v9, v16, v10);
    }
    while ( v13 != v14 );
LABEL_9:
    v17 = *(_QWORD *)a1;
    v13 = *(_QWORD *)a1 + 40LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v13 )
    {
      do
      {
        v13 -= 40;
        v18 = *(_QWORD *)(v13 + 16);
        if ( v18 != v13 + 40 )
          _libc_free(v18, v7);
        v19 = *(_QWORD *)v13;
        v20 = *(_QWORD *)v13 + 80LL * *(unsigned int *)(v13 + 8);
        if ( *(_QWORD *)v13 != v20 )
        {
          do
          {
            v20 -= 80;
            v21 = *(_QWORD *)(v20 + 8);
            if ( v21 != v20 + 24 )
              _libc_free(v21, v7);
          }
          while ( v19 != v20 );
          v19 = *(_QWORD *)v13;
        }
        if ( v19 != v13 + 16 )
          _libc_free(v19, v7);
      }
      while ( v13 != v17 );
      v13 = *(_QWORD *)a1;
    }
  }
  v22 = v26[0];
  if ( v25 != v13 )
    _libc_free(v13, v7);
  *(_DWORD *)(a1 + 12) = v22;
  *(_QWORD *)a1 = v24;
  return v24;
}
