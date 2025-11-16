// Function: sub_1B32780
// Address: 0x1b32780
//
__int64 __fastcall sub_1B32780(_QWORD *a1, __int64 a2, __int64 *a3, __int64 a4, __int64 **a5)
{
  __int64 v5; // rax
  __int64 v7; // rdx
  bool v10; // cf
  unsigned __int64 v11; // rax
  _QWORD *v12; // r15
  __int64 v13; // rsi
  __int64 *v14; // r13
  __int64 *v15; // rbx
  __int64 *v16; // rax
  __int64 *v17; // r14
  _QWORD *v18; // rdi
  __int64 v20; // rbx
  __int64 v21; // rax
  __int64 v22; // r10
  _QWORD *v23; // rdi
  __int64 v24; // r9
  signed __int64 v25; // [rsp+0h] [rbp-50h]
  __int64 v26; // [rsp+8h] [rbp-48h]
  __int64 v28; // [rsp+10h] [rbp-40h]
  __int64 v29; // [rsp+18h] [rbp-38h]

  v5 = (__int64)(a1[1] - *a1) >> 6;
  if ( v5 == 0x1FFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v7 = 1;
  if ( v5 )
    v7 = (__int64)(a1[1] - *a1) >> 6;
  v10 = __CFADD__(v7, v5);
  v11 = v7 + v5;
  if ( v10 )
  {
    v20 = 0x7FFFFFFFFFFFFFC0LL;
  }
  else
  {
    if ( !v11 )
    {
      v12 = *(_QWORD **)a4;
      *(_QWORD *)(a4 + 8) = 0;
      *(_QWORD *)a4 = 0;
      v13 = *(_QWORD *)(a4 + 16);
      v28 = 64;
      *(_QWORD *)(a4 + 16) = 0;
      v14 = *a5;
      v15 = a5[1];
      v16 = a5[2];
      a5[1] = 0;
      a5[2] = 0;
      *a5 = 0;
      v29 = 0;
      goto LABEL_7;
    }
    if ( v11 > 0x1FFFFFFFFFFFFFFLL )
      v11 = 0x1FFFFFFFFFFFFFFLL;
    v20 = v11 << 6;
  }
  v21 = sub_22077B0(v20);
  v22 = *a3;
  v12 = (_QWORD *)v21;
  v23 = *(_QWORD **)a4;
  v24 = *(_QWORD *)(a4 + 8);
  v28 = v21 + 64;
  *(_QWORD *)(a4 + 8) = 0;
  v13 = *(_QWORD *)(a4 + 16);
  *(_QWORD *)a4 = 0;
  *(_QWORD *)(a4 + 16) = 0;
  v14 = *a5;
  v29 = v21 + v20;
  v15 = a5[1];
  v16 = a5[2];
  a5[1] = 0;
  a5[2] = 0;
  *a5 = 0;
  if ( v12 )
  {
    *v12 = v22;
    v12[1] = 0;
    v12[2] = v23;
    v12[3] = v24;
    v12[4] = v13;
    v12[5] = v14;
    v12[6] = v15;
    v12[7] = v16;
    goto LABEL_16;
  }
  v12 = v23;
LABEL_7:
  v26 = v13 - (_QWORD)v12;
  v25 = (char *)v16 - (char *)v14;
  if ( v15 != v14 )
  {
    v17 = v14;
    do
    {
      if ( *v17 )
        sub_161E7C0((__int64)v17, *v17);
      ++v17;
    }
    while ( v15 != v17 );
  }
  if ( v14 )
    j_j___libc_free_0(v14, v25);
  if ( v12 )
  {
    v18 = v12;
    v12 = 0;
    j_j___libc_free_0(v18, v26);
  }
LABEL_16:
  *a1 = v12;
  a1[1] = v28;
  a1[2] = v29;
  return v29;
}
