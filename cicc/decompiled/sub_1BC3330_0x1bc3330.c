// Function: sub_1BC3330
// Address: 0x1bc3330
//
__int64 __fastcall sub_1BC3330(__int64 *a1, char *a2, __int64 a3, __int64 a4)
{
  __int64 v5; // r15
  __int64 *v6; // r12
  unsigned __int64 v7; // rax
  unsigned __int64 v8; // rdi
  __int64 *v9; // r8
  __int64 *v10; // r14
  bool v11; // cf
  unsigned __int64 v12; // rax
  __int64 v13; // r9
  __int64 v14; // rbx
  _QWORD *v15; // rdi
  __int64 v16; // rbx
  __int64 *v17; // rax
  __int64 v18; // rsi
  __int64 v19; // rsi
  __int64 v20; // rax
  __int64 v21; // rsi
  __int64 v22; // rdi
  __int64 *i; // r14
  unsigned __int64 v24; // rdi
  __int64 v26; // rbx
  __int64 v27; // rax
  __int64 v28; // [rsp+8h] [rbp-58h]
  __int64 v29; // [rsp+10h] [rbp-50h]
  __int64 *v30; // [rsp+18h] [rbp-48h]
  __int64 *v31; // [rsp+20h] [rbp-40h]
  __int64 *v32; // [rsp+20h] [rbp-40h]
  __int64 v33; // [rsp+28h] [rbp-38h]

  v5 = a1[1];
  v6 = (__int64 *)*a1;
  v7 = 0xCCCCCCCCCCCCCCCDLL * ((v5 - *a1) >> 3);
  if ( v7 == 0x333333333333333LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v8 = 1;
  v9 = (__int64 *)a2;
  if ( v7 )
    v8 = 0xCCCCCCCCCCCCCCCDLL * ((v5 - (__int64)v6) >> 3);
  v10 = (__int64 *)a2;
  v11 = __CFADD__(v8, v7);
  v12 = v8 - 0x3333333333333333LL * ((v5 - (__int64)v6) >> 3);
  v13 = a2 - (char *)v6;
  if ( v11 )
  {
    v26 = 0x7FFFFFFFFFFFFFF8LL;
LABEL_31:
    v28 = a3;
    v27 = sub_22077B0(v26);
    v13 = a2 - (char *)v6;
    v9 = (__int64 *)a2;
    v33 = v27;
    a3 = v28;
    v29 = v27 + v26;
    v14 = v27 + 40;
    goto LABEL_7;
  }
  if ( v12 )
  {
    if ( v12 > 0x333333333333333LL )
      v12 = 0x333333333333333LL;
    v26 = 40 * v12;
    goto LABEL_31;
  }
  v29 = 0;
  v14 = 40;
  v33 = 0;
LABEL_7:
  v15 = (_QWORD *)(v33 + v13);
  if ( v33 + v13 )
  {
    a4 = *(unsigned int *)(a3 + 16);
    *v15 = *(_QWORD *)a3;
    v15[1] = v15 + 3;
    v15[2] = 0x200000000LL;
    if ( (_DWORD)a4 )
    {
      v32 = v9;
      sub_1BB9C60((__int64)(v15 + 1), (char **)(a3 + 8), a3, a4, (int)v9, v13);
      v9 = v32;
    }
  }
  if ( v9 != v6 )
  {
    v16 = v33;
    v17 = v6;
    while ( 1 )
    {
      if ( v16
        && (v19 = *v17,
            *(_DWORD *)(v16 + 16) = 0,
            *(_DWORD *)(v16 + 20) = 2,
            *(_QWORD *)v16 = v19,
            *(_QWORD *)(v16 + 8) = v16 + 24,
            a3 = *((unsigned int *)v17 + 4),
            (_DWORD)a3) )
      {
        v30 = v9;
        v31 = v17;
        sub_1BB9960(v16 + 8, (__int64)(v17 + 1), a3, a4, (int)v9, v13);
        v9 = v30;
        v18 = v16 + 40;
        v17 = v31 + 5;
        if ( v30 == v31 + 5 )
        {
LABEL_17:
          v14 = v16 + 80;
          break;
        }
      }
      else
      {
        v17 += 5;
        v18 = v16 + 40;
        if ( v9 == v17 )
          goto LABEL_17;
      }
      v16 = v18;
    }
  }
  if ( v9 != (__int64 *)v5 )
  {
    do
    {
      while ( 1 )
      {
        v20 = *v10;
        *(_DWORD *)(v14 + 16) = 0;
        *(_DWORD *)(v14 + 20) = 2;
        *(_QWORD *)v14 = v20;
        *(_QWORD *)(v14 + 8) = v14 + 24;
        if ( *((_DWORD *)v10 + 4) )
          break;
        v10 += 5;
        v14 += 40;
        if ( (__int64 *)v5 == v10 )
          goto LABEL_23;
      }
      v21 = (__int64)(v10 + 1);
      v22 = v14 + 8;
      v10 += 5;
      v14 += 40;
      sub_1BB9960(v22, v21, a3, a4, (int)v9, v13);
    }
    while ( (__int64 *)v5 != v10 );
  }
LABEL_23:
  for ( i = v6; (__int64 *)v5 != i; i += 5 )
  {
    v24 = i[1];
    if ( (__int64 *)v24 != i + 3 )
      _libc_free(v24);
  }
  if ( v6 )
    j_j___libc_free_0(v6, a1[2] - (_QWORD)v6);
  a1[1] = v14;
  *a1 = v33;
  a1[2] = v29;
  return v29;
}
