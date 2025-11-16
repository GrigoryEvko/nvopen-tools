// Function: sub_2712560
// Address: 0x2712560
//
unsigned __int64 *__fastcall sub_2712560(
        unsigned __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6)
{
  unsigned __int64 v6; // r12
  __int64 v7; // rax
  __int64 v9; // rcx
  bool v10; // zf
  __int64 v11; // rax
  __int64 v12; // r15
  bool v13; // cf
  unsigned __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // rbx
  __int64 v18; // r13
  unsigned __int64 v19; // r13
  unsigned __int64 i; // rbx
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  __int64 v24; // rdx
  __int64 v25; // rsi
  __int64 v26; // rdi
  __int64 v27; // rcx
  __int64 v28; // r8
  __int64 v29; // r9
  unsigned __int64 v30; // r14
  unsigned __int64 v31; // rdi
  unsigned __int64 v33; // rbx
  __int64 v34; // rax
  __int64 v35; // [rsp+0h] [rbp-60h]
  __int64 v36; // [rsp+8h] [rbp-58h]
  unsigned __int64 v37; // [rsp+10h] [rbp-50h]
  unsigned __int64 v39; // [rsp+20h] [rbp-40h]
  unsigned __int64 v40; // [rsp+28h] [rbp-38h]

  v6 = a1[1];
  v40 = *a1;
  v7 = (__int64)(v6 - *a1) >> 7;
  if ( v7 == 0xFFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v9 = (__int64)(v6 - *a1) >> 7;
  v10 = v7 == 0;
  v11 = 1;
  v12 = a2;
  if ( !v10 )
    v11 = (__int64)(v6 - *a1) >> 7;
  v13 = __CFADD__(v9, v11);
  v14 = v9 + v11;
  v15 = a2 - v40;
  v16 = v13;
  if ( v13 )
  {
    v33 = 0x7FFFFFFFFFFFFF80LL;
  }
  else
  {
    if ( !v14 )
    {
      v37 = 0;
      v17 = 128;
      v39 = 0;
      goto LABEL_7;
    }
    if ( v14 > 0xFFFFFFFFFFFFFFLL )
      v14 = 0xFFFFFFFFFFFFFFLL;
    v33 = v14 << 7;
  }
  v35 = a3;
  v34 = sub_22077B0(v33);
  v15 = a2 - v40;
  a3 = v35;
  v39 = v34;
  v37 = v34 + v33;
  v17 = v34 + 128;
LABEL_7:
  v18 = v39 + v15;
  if ( v39 + v15 )
  {
    v36 = a3;
    *(_QWORD *)v18 = *(_QWORD *)a3;
    *(_WORD *)(v18 + 8) = *(_WORD *)(a3 + 8);
    *(_QWORD *)(v18 + 16) = *(_QWORD *)(a3 + 16);
    sub_C8CF70(v18 + 24, (void *)(v18 + 56), 2, a3 + 56, a3 + 24);
    sub_C8CF70(v18 + 72, (void *)(v18 + 104), 2, v36 + 104, v36 + 72);
    *(_BYTE *)(v18 + 120) = *(_BYTE *)(v36 + 120);
  }
  v19 = v40;
  if ( a2 != v40 )
  {
    for ( i = v39; ; i += 128LL )
    {
      if ( i )
      {
        *(_QWORD *)i = *(_QWORD *)v19;
        *(_BYTE *)(i + 8) = *(_BYTE *)(v19 + 8);
        *(_BYTE *)(i + 9) = *(_BYTE *)(v19 + 9);
        *(_QWORD *)(i + 16) = *(_QWORD *)(v19 + 16);
        sub_C8CD80(i + 24, i + 56, v19 + 24, v16, a5, a6);
        sub_C8CD80(i + 72, i + 104, v19 + 72, v21, v22, v23);
        *(_BYTE *)(i + 120) = *(_BYTE *)(v19 + 120);
      }
      v19 += 128LL;
      if ( a2 == v19 )
        break;
    }
    v17 = i + 256;
  }
  if ( a2 != v6 )
  {
    do
    {
      *(_QWORD *)v17 = *(_QWORD *)v12;
      *(_BYTE *)(v17 + 8) = *(_BYTE *)(v12 + 8);
      *(_BYTE *)(v17 + 9) = *(_BYTE *)(v12 + 9);
      *(_QWORD *)(v17 + 16) = *(_QWORD *)(v12 + 16);
      sub_C8CD80(v17 + 24, v17 + 56, v12 + 24, v16, a5, a6);
      v24 = v12 + 72;
      v25 = v17 + 104;
      v12 += 128;
      v26 = v17 + 72;
      v17 += 128;
      sub_C8CD80(v26, v25, v24, v27, v28, v29);
      *(_BYTE *)(v17 - 8) = *(_BYTE *)(v12 - 8);
    }
    while ( v6 != v12 );
  }
  v30 = v40;
  if ( v40 != v6 )
  {
    while ( 1 )
    {
      if ( *(_BYTE *)(v30 + 100) )
      {
        if ( *(_BYTE *)(v30 + 52) )
          goto LABEL_21;
LABEL_24:
        v31 = *(_QWORD *)(v30 + 32);
        v30 += 128LL;
        _libc_free(v31);
        if ( v30 == v6 )
          break;
      }
      else
      {
        _libc_free(*(_QWORD *)(v30 + 80));
        if ( !*(_BYTE *)(v30 + 52) )
          goto LABEL_24;
LABEL_21:
        v30 += 128LL;
        if ( v30 == v6 )
          break;
      }
    }
  }
  if ( v40 )
    j_j___libc_free_0(v40);
  *a1 = v39;
  a1[1] = v17;
  a1[2] = v37;
  return a1;
}
