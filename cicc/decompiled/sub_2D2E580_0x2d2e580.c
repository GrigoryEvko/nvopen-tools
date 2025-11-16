// Function: sub_2D2E580
// Address: 0x2d2e580
//
__int64 __fastcall sub_2D2E580(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rax
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // rcx
  __int64 v11; // r13
  unsigned __int64 v12; // rax
  unsigned __int64 v13; // r12
  __int64 v14; // rdx
  __int64 v15; // rdx
  unsigned int *v16; // rdi
  unsigned __int64 v17; // r13
  __int64 v18; // rax
  unsigned __int64 v19; // r14
  __int64 v20; // r15
  __int64 v21; // rsi
  int v22; // r13d
  __int64 v24; // [rsp+8h] [rbp-58h]
  __int64 v25; // [rsp+10h] [rbp-50h]
  unsigned __int64 v26; // [rsp+18h] [rbp-48h]
  unsigned __int64 v27[7]; // [rsp+28h] [rbp-38h] BYREF

  v25 = a1 + 16;
  v7 = sub_C8D7D0(a1, a1 + 16, a2, 0x38u, v27, a6);
  v10 = *(unsigned int *)(a1 + 8);
  v24 = v7;
  v11 = v7;
  v12 = *(_QWORD *)a1;
  v13 = *(_QWORD *)a1 + 56 * v10;
  if ( *(_QWORD *)a1 != v13 )
  {
    do
    {
      while ( 1 )
      {
        if ( v11 )
        {
          v14 = *(_QWORD *)v12;
          *(_DWORD *)(v11 + 16) = 0;
          *(_DWORD *)(v11 + 20) = 1;
          *(_QWORD *)v11 = v14;
          *(_QWORD *)(v11 + 8) = v11 + 24;
          v15 = *(unsigned int *)(v12 + 16);
          if ( (_DWORD)v15 )
            break;
        }
        v12 += 56LL;
        v11 += 56;
        if ( v13 == v12 )
          goto LABEL_7;
      }
      v16 = (unsigned int *)(v11 + 8);
      v26 = v12;
      v11 += 56;
      sub_2D29780(v16, v12 + 8, v15, v10, v8, v9);
      v12 = v26 + 56;
    }
    while ( v13 != v26 + 56 );
LABEL_7:
    v17 = *(_QWORD *)a1;
    v13 = *(_QWORD *)a1 + 56LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v13 )
    {
      do
      {
        v18 = *(unsigned int *)(v13 - 40);
        v19 = *(_QWORD *)(v13 - 48);
        v13 -= 56LL;
        v18 *= 32;
        v20 = v19 + v18;
        if ( v19 != v19 + v18 )
        {
          do
          {
            v21 = *(_QWORD *)(v20 - 16);
            v20 -= 32;
            if ( v21 )
              sub_B91220(v20 + 16, v21);
          }
          while ( v19 != v20 );
          v19 = *(_QWORD *)(v13 + 8);
        }
        if ( v19 != v13 + 24 )
          _libc_free(v19);
      }
      while ( v13 != v17 );
      v13 = *(_QWORD *)a1;
    }
  }
  v22 = v27[0];
  if ( v25 != v13 )
    _libc_free(v13);
  *(_DWORD *)(a1 + 12) = v22;
  *(_QWORD *)a1 = v24;
  return v24;
}
