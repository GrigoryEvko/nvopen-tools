// Function: sub_39A5160
// Address: 0x39a5160
//
unsigned __int64 __fastcall sub_39A5160(unsigned __int64 *a1, __int64 *a2, __int64 a3, __int64 a4)
{
  unsigned __int64 v5; // r15
  unsigned __int64 v6; // r12
  __int64 v7; // rax
  __int64 v8; // rdi
  __int64 *v9; // r8
  __int64 *v10; // r14
  bool v11; // cf
  unsigned __int64 v12; // rax
  __int64 v13; // r9
  __int64 v14; // rbx
  _QWORD *v15; // rdi
  unsigned __int64 v16; // rbx
  unsigned __int64 v17; // rax
  unsigned __int64 v18; // rsi
  __int64 v19; // rsi
  __int64 v20; // rax
  __int64 v21; // rsi
  __int64 v22; // rdi
  unsigned __int64 i; // r14
  unsigned __int64 v24; // rdi
  unsigned __int64 v26; // rbx
  __int64 v27; // rax
  __int64 v28; // [rsp+8h] [rbp-58h]
  unsigned __int64 v29; // [rsp+10h] [rbp-50h]
  __int64 *v30; // [rsp+18h] [rbp-48h]
  unsigned __int64 v31; // [rsp+20h] [rbp-40h]
  __int64 *v32; // [rsp+20h] [rbp-40h]
  unsigned __int64 v33; // [rsp+28h] [rbp-38h]

  v5 = a1[1];
  v6 = *a1;
  v7 = 0x2E8BA2E8BA2E8BA3LL * ((__int64)(v5 - *a1) >> 3);
  if ( v7 == 0x1745D1745D1745DLL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v8 = 1;
  v9 = a2;
  if ( v7 )
    v8 = 0x2E8BA2E8BA2E8BA3LL * ((__int64)(v5 - v6) >> 3);
  v10 = a2;
  v11 = __CFADD__(v8, v7);
  v12 = v8 + v7;
  v13 = (__int64)a2 - v6;
  if ( v11 )
  {
    v26 = 0x7FFFFFFFFFFFFFF8LL;
LABEL_31:
    v28 = a3;
    v27 = sub_22077B0(v26);
    v13 = (__int64)a2 - v6;
    v9 = a2;
    v33 = v27;
    a3 = v28;
    v29 = v27 + v26;
    v14 = v27 + 88;
    goto LABEL_7;
  }
  if ( v12 )
  {
    if ( v12 > 0x1745D1745D1745DLL )
      v12 = 0x1745D1745D1745DLL;
    v26 = 88 * v12;
    goto LABEL_31;
  }
  v29 = 0;
  v14 = 88;
  v33 = 0;
LABEL_7:
  v15 = (_QWORD *)(v33 + v13);
  if ( v33 + v13 )
  {
    a4 = *(unsigned int *)(a3 + 16);
    *v15 = *(_QWORD *)a3;
    v15[1] = v15 + 3;
    v15[2] = 0x800000000LL;
    if ( (_DWORD)a4 )
    {
      v32 = v9;
      sub_39A1C20((__int64)(v15 + 1), (char **)(a3 + 8), a3, a4, (int)v9, v13);
      v9 = v32;
    }
  }
  if ( v9 != (__int64 *)v6 )
  {
    v16 = v33;
    v17 = v6;
    while ( 1 )
    {
      if ( v16
        && (v19 = *(_QWORD *)v17,
            *(_DWORD *)(v16 + 16) = 0,
            *(_DWORD *)(v16 + 20) = 8,
            *(_QWORD *)v16 = v19,
            *(_QWORD *)(v16 + 8) = v16 + 24,
            a3 = *(unsigned int *)(v17 + 16),
            (_DWORD)a3) )
      {
        v30 = v9;
        v31 = v17;
        sub_39A1B40(v16 + 8, v17 + 8, a3, a4, (int)v9, v13);
        v9 = v30;
        v18 = v16 + 88;
        v17 = v31 + 88;
        if ( v30 == (__int64 *)(v31 + 88) )
        {
LABEL_17:
          v14 = v16 + 176;
          break;
        }
      }
      else
      {
        v17 += 88LL;
        v18 = v16 + 88;
        if ( v9 == (__int64 *)v17 )
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
        *(_DWORD *)(v14 + 20) = 8;
        *(_QWORD *)v14 = v20;
        *(_QWORD *)(v14 + 8) = v14 + 24;
        if ( *((_DWORD *)v10 + 4) )
          break;
        v10 += 11;
        v14 += 88;
        if ( (__int64 *)v5 == v10 )
          goto LABEL_23;
      }
      v21 = (__int64)(v10 + 1);
      v22 = v14 + 8;
      v10 += 11;
      v14 += 88;
      sub_39A1B40(v22, v21, a3, a4, (int)v9, v13);
    }
    while ( (__int64 *)v5 != v10 );
  }
LABEL_23:
  for ( i = v6; v5 != i; i += 88LL )
  {
    v24 = *(_QWORD *)(i + 8);
    if ( v24 != i + 24 )
      _libc_free(v24);
  }
  if ( v6 )
    j_j___libc_free_0(v6);
  a1[1] = v14;
  *a1 = v33;
  a1[2] = v29;
  return v29;
}
