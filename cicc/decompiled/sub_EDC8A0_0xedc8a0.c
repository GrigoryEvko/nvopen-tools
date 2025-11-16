// Function: sub_EDC8A0
// Address: 0xedc8a0
//
__int64 __fastcall sub_EDC8A0(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4)
{
  _QWORD *v5; // r15
  _QWORD *v6; // r12
  unsigned __int64 v7; // rax
  unsigned __int64 v8; // rdi
  __int64 v9; // r8
  __int64 v10; // r14
  bool v11; // cf
  unsigned __int64 v12; // rax
  __int64 v13; // r9
  __int64 v14; // rbx
  _QWORD *v15; // rdi
  bool v16; // zf
  _QWORD *v17; // rbx
  __int64 v18; // rax
  __int64 v19; // rdi
  _QWORD *i; // r14
  __int64 v22; // rbx
  __int64 v23; // rax
  __int64 v24; // [rsp+8h] [rbp-58h]
  __int64 v25; // [rsp+10h] [rbp-50h]
  __int64 v26; // [rsp+18h] [rbp-48h]
  __int64 v27; // [rsp+20h] [rbp-40h]
  __int64 v28; // [rsp+20h] [rbp-40h]
  _QWORD *v29; // [rsp+28h] [rbp-38h]

  v5 = (_QWORD *)a1[1];
  v6 = (_QWORD *)*a1;
  v7 = 0xCCCCCCCCCCCCCCCDLL * (((__int64)v5 - *a1) >> 3);
  if ( v7 == 0x333333333333333LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v8 = 1;
  v9 = a2;
  if ( v7 )
    v8 = 0xCCCCCCCCCCCCCCCDLL * (v5 - v6);
  v10 = a2;
  v11 = __CFADD__(v8, v7);
  v12 = v8 - 0x3333333333333333LL * (v5 - v6);
  v13 = a2 - (_QWORD)v6;
  if ( v11 )
  {
    v22 = 0x7FFFFFFFFFFFFFF8LL;
LABEL_31:
    v24 = a3;
    v23 = sub_22077B0(v22);
    v13 = a2 - (_QWORD)v6;
    v9 = a2;
    v29 = (_QWORD *)v23;
    a3 = v24;
    v25 = v23 + v22;
    v14 = v23 + 40;
    goto LABEL_7;
  }
  if ( v12 )
  {
    if ( v12 > 0x333333333333333LL )
      v12 = 0x333333333333333LL;
    v22 = 40 * v12;
    goto LABEL_31;
  }
  v25 = 0;
  v14 = 40;
  v29 = 0;
LABEL_7:
  v15 = (_QWORD *)((char *)v29 + v13);
  if ( (_QWORD *)((char *)v29 + v13) )
  {
    v16 = *(_QWORD *)(a3 + 8) == 0;
    v15[1] = 0;
    *v15 = v15 + 3;
    v15[2] = 10;
    if ( !v16 )
    {
      a2 = a3;
      v28 = v9;
      sub_ED6290((__int64)v15, (char **)a3, a3, a4, v9, v13);
      v9 = v28;
    }
  }
  if ( (_QWORD *)v9 != v6 )
  {
    v17 = v29;
    v18 = (__int64)v6;
    while ( 1 )
    {
      if ( v17 && (v17[1] = 0, *v17 = v17 + 3, v17[2] = 10, *(_QWORD *)(v18 + 8)) )
      {
        v26 = v9;
        v27 = v18;
        sub_ED61C0((__int64)v17, v18, a3, a4, v9, v13);
        v9 = v26;
        a2 = (__int64)(v17 + 5);
        v18 = v27 + 40;
        if ( v26 == v27 + 40 )
        {
LABEL_17:
          v14 = (__int64)(v17 + 10);
          break;
        }
      }
      else
      {
        v18 += 40;
        a2 = (__int64)(v17 + 5);
        if ( v9 == v18 )
          goto LABEL_17;
      }
      v17 = (_QWORD *)a2;
    }
  }
  if ( (_QWORD *)v9 != v5 )
  {
    do
    {
      while ( 1 )
      {
        v16 = *(_QWORD *)(v10 + 8) == 0;
        *(_QWORD *)(v14 + 8) = 0;
        *(_QWORD *)v14 = v14 + 24;
        *(_QWORD *)(v14 + 16) = 10;
        if ( !v16 )
          break;
        v10 += 40;
        v14 += 40;
        if ( v5 == (_QWORD *)v10 )
          goto LABEL_23;
      }
      a2 = v10;
      v19 = v14;
      v10 += 40;
      v14 += 40;
      sub_ED61C0(v19, a2, a3, a4, v9, v13);
    }
    while ( v5 != (_QWORD *)v10 );
  }
LABEL_23:
  for ( i = v6; v5 != i; i += 5 )
  {
    if ( (_QWORD *)*i != i + 3 )
      _libc_free(*i, a2);
  }
  if ( v6 )
    j_j___libc_free_0(v6, a1[2] - (_QWORD)v6);
  a1[1] = v14;
  *a1 = v29;
  a1[2] = v25;
  return v25;
}
