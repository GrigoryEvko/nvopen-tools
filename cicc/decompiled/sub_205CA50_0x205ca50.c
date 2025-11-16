// Function: sub_205CA50
// Address: 0x205ca50
//
unsigned __int64 **__fastcall sub_205CA50(
        unsigned __int64 **a1,
        unsigned __int64 *a2,
        __int64 a3,
        __int64 a4,
        int a5,
        int a6)
{
  __int64 v6; // r15
  unsigned __int64 *v8; // r12
  unsigned __int64 *v9; // r14
  unsigned __int64 v10; // rax
  unsigned __int64 v11; // rsi
  bool v12; // cf
  unsigned __int64 v13; // rax
  signed __int64 v14; // rsi
  __int64 v15; // rbx
  __int64 v16; // rax
  int v17; // edi
  int v18; // esi
  __int64 v19; // rcx
  char v20; // si
  __int64 v21; // rax
  __int64 v22; // rbx
  __int64 v23; // rsi
  __int64 v24; // rdi
  unsigned __int64 *i; // r13
  unsigned __int64 v26; // rdi
  unsigned __int64 v27; // rdi
  unsigned __int64 v28; // rdi
  __int64 v30; // rbx
  __int64 v31; // rax
  __int64 v32; // [rsp+8h] [rbp-58h]
  __int64 v33; // [rsp+8h] [rbp-58h]
  __int64 v34; // [rsp+8h] [rbp-58h]
  __int64 v35; // [rsp+8h] [rbp-58h]
  __int64 v36; // [rsp+8h] [rbp-58h]
  __int64 v37; // [rsp+10h] [rbp-50h]
  __int64 v39; // [rsp+20h] [rbp-40h]
  __int64 v40; // [rsp+28h] [rbp-38h]
  __int64 v41; // [rsp+28h] [rbp-38h]
  __int64 v42; // [rsp+28h] [rbp-38h]
  __int64 v43; // [rsp+28h] [rbp-38h]

  v6 = (__int64)a2;
  v8 = a1[1];
  v9 = *a1;
  v10 = 0xD37A6F4DE9BD37A7LL * (v8 - *a1);
  if ( v10 == 0xB21642C8590B21LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v11 = 1;
  if ( v10 )
    v11 = 0xD37A6F4DE9BD37A7LL * (v8 - v9);
  v12 = __CFADD__(v11, v10);
  v13 = v11 - 0x2C8590B21642C859LL * (v8 - v9);
  v14 = (char *)a2 - (char *)v9;
  if ( v12 )
  {
    v30 = 0x7FFFFFFFFFFFFFB8LL;
  }
  else
  {
    if ( !v13 )
    {
      v37 = 0;
      v15 = 184;
      v39 = 0;
      goto LABEL_7;
    }
    if ( v13 > 0xB21642C8590B21LL )
      v13 = 0xB21642C8590B21LL;
    v30 = 184 * v13;
  }
  v32 = a3;
  v31 = sub_22077B0(v30);
  v14 = (char *)a2 - (char *)v9;
  a3 = v32;
  v39 = v31;
  v37 = v31 + v30;
  v15 = v31 + 184;
LABEL_7:
  v16 = v39 + v14;
  if ( v39 + v14 )
  {
    a5 = *(_DWORD *)(a3 + 8);
    *(_QWORD *)v16 = v16 + 16;
    *(_QWORD *)(v16 + 8) = 0x400000000LL;
    if ( a5 )
    {
      v34 = a3;
      sub_20449C0(v16, (char **)a3, a3, 0x400000000LL, a5, a6);
      a3 = v34;
      v16 = v39 + v14;
    }
    v17 = *(_DWORD *)(a3 + 88);
    *(_QWORD *)(v16 + 80) = v16 + 96;
    *(_QWORD *)(v16 + 88) = 0x400000000LL;
    if ( v17 )
    {
      v35 = a3;
      v42 = v16;
      sub_2044890(v16 + 80, (char **)(a3 + 80), a3, 0x400000000LL, a5, a6);
      a3 = v35;
      v16 = v42;
    }
    *(_QWORD *)(v16 + 104) = v16 + 120;
    v18 = *(_DWORD *)(a3 + 112);
    *(_QWORD *)(v16 + 112) = 0x400000000LL;
    if ( v18 )
    {
      v36 = a3;
      v43 = v16;
      sub_2044C40(v16 + 104, (char **)(a3 + 104), a3, 0x400000000LL, a5, a6);
      a3 = v36;
      v16 = v43;
    }
    *(_QWORD *)(v16 + 144) = 0x400000000LL;
    v19 = *(unsigned int *)(a3 + 144);
    *(_QWORD *)(v16 + 136) = v16 + 152;
    if ( (_DWORD)v19 )
    {
      v33 = a3;
      v41 = v16;
      sub_2044C40(v16 + 136, (char **)(a3 + 136), a3, v19, a5, a6);
      a3 = v33;
      v16 = v41;
    }
    v20 = *(_BYTE *)(a3 + 172);
    *(_BYTE *)(v16 + 172) = v20;
    if ( v20 )
      *(_DWORD *)(v16 + 168) = *(_DWORD *)(a3 + 168);
    a4 = *(_QWORD *)(a3 + 176);
    *(_QWORD *)(v16 + 176) = a4;
  }
  if ( a2 != v9 )
  {
    v21 = v39;
    v22 = (__int64)v9;
    while ( 1 )
    {
      if ( v21 )
      {
        v40 = v21;
        sub_205AAF0(v21, v22, a3, a4, a5, a6);
        v21 = v40;
        *(_QWORD *)(v40 + 176) = *(_QWORD *)(v22 + 176);
      }
      v22 += 184;
      a4 = v21 + 184;
      if ( a2 == (unsigned __int64 *)v22 )
        break;
      v21 += 184;
    }
    v15 = v21 + 368;
  }
  if ( a2 != v8 )
  {
    do
    {
      v23 = v6;
      v24 = v15;
      v6 += 184;
      v15 += 184;
      sub_205AAF0(v24, v23, a3, a4, a5, a6);
      *(_QWORD *)(v15 - 8) = *(_QWORD *)(v6 - 8);
    }
    while ( v8 != (unsigned __int64 *)v6 );
  }
  for ( i = v9; i != v8; i += 23 )
  {
    v26 = i[17];
    if ( (unsigned __int64 *)v26 != i + 19 )
      _libc_free(v26);
    v27 = i[13];
    if ( (unsigned __int64 *)v27 != i + 15 )
      _libc_free(v27);
    v28 = i[10];
    if ( (unsigned __int64 *)v28 != i + 12 )
      _libc_free(v28);
    if ( (unsigned __int64 *)*i != i + 2 )
      _libc_free(*i);
  }
  if ( v9 )
    j_j___libc_free_0(v9, (char *)a1[2] - (char *)v9);
  *a1 = (unsigned __int64 *)v39;
  a1[1] = (unsigned __int64 *)v15;
  a1[2] = (unsigned __int64 *)v37;
  return a1;
}
