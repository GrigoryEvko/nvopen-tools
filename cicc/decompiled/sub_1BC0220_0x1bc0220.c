// Function: sub_1BC0220
// Address: 0x1bc0220
//
unsigned __int64 **__fastcall sub_1BC0220(unsigned __int64 **a1, unsigned __int64 *a2, __int64 a3, __int64 a4, int a5)
{
  unsigned __int64 *v5; // r12
  unsigned __int64 *v6; // r13
  __int64 v7; // rax
  __int64 v8; // rcx
  __int64 v9; // r15
  bool v10; // cf
  unsigned __int64 v11; // rax
  int v12; // r9d
  __int64 v13; // rcx
  __int64 v14; // rbx
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rbx
  __int64 v18; // rsi
  __int64 v19; // rdi
  unsigned __int64 *i; // r14
  unsigned __int64 v21; // rdi
  unsigned __int64 v22; // rdi
  __int64 v24; // rbx
  __int64 v25; // rax
  __int64 v26; // [rsp+8h] [rbp-58h]
  __int64 v27; // [rsp+10h] [rbp-50h]
  __int64 v29; // [rsp+20h] [rbp-40h]
  __int64 v30; // [rsp+28h] [rbp-38h]

  v5 = a1[1];
  v6 = *a1;
  v7 = 0x2E8BA2E8BA2E8BA3LL * (((char *)v5 - (char *)*a1) >> 4);
  if ( v7 == 0xBA2E8BA2E8BA2ELL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v8 = 1;
  v9 = (__int64)a2;
  if ( v7 )
    v8 = 0x2E8BA2E8BA2E8BA3LL * (((char *)v5 - (char *)v6) >> 4);
  v10 = __CFADD__(v8, v7);
  v11 = v8 + v7;
  v12 = v10;
  v13 = (char *)a2 - (char *)v6;
  if ( v10 )
  {
    v24 = 0x7FFFFFFFFFFFFFA0LL;
  }
  else
  {
    if ( !v11 )
    {
      v27 = 0;
      v14 = 176;
      v29 = 0;
      goto LABEL_7;
    }
    if ( v11 > 0xBA2E8BA2E8BA2ELL )
      v11 = 0xBA2E8BA2E8BA2ELL;
    v24 = 176 * v11;
  }
  v26 = a3;
  v25 = sub_22077B0(v24);
  v13 = (char *)a2 - (char *)v6;
  a3 = v26;
  v29 = v25;
  v27 = v25 + v24;
  v14 = v25 + 176;
LABEL_7:
  v15 = v29 + v13;
  if ( v29 + v13 )
  {
    *(_QWORD *)(v15 + 80) = 0;
    *(_QWORD *)v15 = v15 + 16;
    *(_QWORD *)(v15 + 8) = 0x800000000LL;
    *(_QWORD *)(v15 + 96) = v15 + 112;
    v13 = v15 + 168;
    *(_QWORD *)(v15 + 104) = 0x400000000LL;
    *(_BYTE *)(v15 + 88) = 0;
    *(_QWORD *)(v15 + 128) = 0;
    *(_QWORD *)(v15 + 136) = 0;
    *(_QWORD *)(v15 + 144) = a3;
    *(_QWORD *)(v15 + 152) = v15 + 168;
    *(_QWORD *)(v15 + 160) = 0x100000000LL;
  }
  if ( a2 != v6 )
  {
    v16 = v29;
    v17 = (__int64)v6;
    while ( 1 )
    {
      if ( v16 )
      {
        v30 = v16;
        sub_1BBD870(v16, v17, a3, v13, a5, v12);
        v16 = v30;
      }
      v17 += 176;
      v13 = v16 + 176;
      if ( a2 == (unsigned __int64 *)v17 )
        break;
      v16 += 176;
    }
    v14 = v16 + 352;
  }
  if ( a2 != v5 )
  {
    do
    {
      v18 = v9;
      v19 = v14;
      v9 += 176;
      v14 += 176;
      sub_1BBD870(v19, v18, a3, v13, a5, v12);
    }
    while ( v5 != (unsigned __int64 *)v9 );
  }
  for ( i = v6; i != v5; i += 22 )
  {
    v21 = i[19];
    if ( (unsigned __int64 *)v21 != i + 21 )
      _libc_free(v21);
    v22 = i[12];
    if ( (unsigned __int64 *)v22 != i + 14 )
      _libc_free(v22);
    if ( (unsigned __int64 *)*i != i + 2 )
      _libc_free(*i);
  }
  if ( v6 )
    j_j___libc_free_0(v6, (char *)a1[2] - (char *)v6);
  *a1 = (unsigned __int64 *)v29;
  a1[1] = (unsigned __int64 *)v14;
  a1[2] = (unsigned __int64 *)v27;
  return a1;
}
