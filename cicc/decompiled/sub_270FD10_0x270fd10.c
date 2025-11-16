// Function: sub_270FD10
// Address: 0x270fd10
//
__int64 *__fastcall sub_270FD10(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r8
  __int64 v7; // r14
  __int64 v9; // rbx
  __int64 v10; // rcx
  unsigned __int64 v11; // rax
  unsigned __int64 v12; // rdx
  bool v13; // cf
  unsigned __int64 v14; // rax
  __int64 v15; // r10
  __int64 v16; // r15
  bool v17; // zf
  __int64 v18; // r10
  __int64 v19; // r12
  __int64 v20; // rsi
  __int16 v21; // dx
  __int64 v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // r9
  unsigned __int64 v25; // r12
  __int64 i; // r15
  __int64 v27; // rcx
  __int64 v28; // r8
  __int64 v29; // r9
  __int64 v30; // rdx
  __int64 v31; // rsi
  __int64 v32; // rdi
  __int64 v33; // rcx
  __int64 v34; // r8
  __int64 v35; // r9
  unsigned __int64 v36; // r13
  unsigned __int64 v37; // rdi
  unsigned __int64 v39; // r15
  __int64 v40; // rax
  __int64 v41; // [rsp+0h] [rbp-60h]
  __int64 v42; // [rsp+8h] [rbp-58h]
  unsigned __int64 v43; // [rsp+10h] [rbp-50h]
  __int64 v45; // [rsp+20h] [rbp-40h]
  unsigned __int64 v46; // [rsp+28h] [rbp-38h]

  v6 = a3;
  v7 = a2;
  v9 = a1[1];
  v10 = *a1;
  v46 = *a1;
  v11 = 0xF0F0F0F0F0F0F0F1LL * ((v9 - *a1) >> 3);
  if ( v11 == 0xF0F0F0F0F0F0F0LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v12 = 1;
  if ( v11 )
    v12 = 0xF0F0F0F0F0F0F0F1LL * ((v9 - v10) >> 3);
  v13 = __CFADD__(v12, v11);
  v14 = v12 - 0xF0F0F0F0F0F0F0FLL * ((v9 - v10) >> 3);
  v15 = a2 - v46;
  if ( v13 )
  {
    v39 = 0x7FFFFFFFFFFFFF80LL;
  }
  else
  {
    if ( !v14 )
    {
      v43 = 0;
      v16 = 136;
      v45 = 0;
      goto LABEL_7;
    }
    if ( v14 > 0xF0F0F0F0F0F0F0LL )
      v14 = 0xF0F0F0F0F0F0F0LL;
    v39 = 136 * v14;
  }
  v41 = v6;
  v40 = sub_22077B0(v39);
  v15 = a2 - v46;
  v6 = v41;
  v45 = v40;
  v43 = v40 + v39;
  v16 = v40 + 136;
LABEL_7:
  v17 = v45 + v15 == 0;
  v18 = v45 + v15;
  v19 = v18;
  if ( !v17 )
  {
    v20 = *(_QWORD *)v6;
    v42 = v6;
    *(_BYTE *)(v18 + 10) = *(_BYTE *)(v6 + 10);
    v21 = *(_WORD *)(v6 + 16);
    *(_QWORD *)v18 = v20;
    LOWORD(v20) = *(_WORD *)(v6 + 8);
    *(_WORD *)(v18 + 16) = v21;
    v22 = *(_QWORD *)(v6 + 24);
    *(_WORD *)(v18 + 8) = v20;
    *(_QWORD *)(v18 + 24) = v22;
    sub_C8CD80(v18 + 32, v18 + 64, v6 + 32, v10, v6, a6);
    sub_C8CD80(v19 + 80, v19 + 112, v42 + 80, v23, v42, v24);
    v6 = v42;
    *(_BYTE *)(v19 + 128) = *(_BYTE *)(v42 + 128);
  }
  v25 = v46;
  if ( a2 != v46 )
  {
    for ( i = v45; ; i += 136 )
    {
      if ( i )
      {
        *(_QWORD *)i = *(_QWORD *)v25;
        *(_BYTE *)(i + 8) = *(_BYTE *)(v25 + 8);
        *(_BYTE *)(i + 9) = *(_BYTE *)(v25 + 9);
        *(_BYTE *)(i + 10) = *(_BYTE *)(v25 + 10);
        *(_BYTE *)(i + 16) = *(_BYTE *)(v25 + 16);
        *(_BYTE *)(i + 17) = *(_BYTE *)(v25 + 17);
        *(_QWORD *)(i + 24) = *(_QWORD *)(v25 + 24);
        sub_C8CD80(i + 32, i + 64, v25 + 32, v10, v6, a6);
        sub_C8CD80(i + 80, i + 112, v25 + 80, v27, v28, v29);
        *(_BYTE *)(i + 128) = *(_BYTE *)(v25 + 128);
      }
      v25 += 136LL;
      if ( a2 == v25 )
        break;
    }
    v16 = i + 272;
  }
  if ( a2 != v9 )
  {
    do
    {
      *(_QWORD *)v16 = *(_QWORD *)v7;
      *(_BYTE *)(v16 + 8) = *(_BYTE *)(v7 + 8);
      *(_BYTE *)(v16 + 9) = *(_BYTE *)(v7 + 9);
      *(_BYTE *)(v16 + 10) = *(_BYTE *)(v7 + 10);
      *(_BYTE *)(v16 + 16) = *(_BYTE *)(v7 + 16);
      *(_BYTE *)(v16 + 17) = *(_BYTE *)(v7 + 17);
      *(_QWORD *)(v16 + 24) = *(_QWORD *)(v7 + 24);
      sub_C8CD80(v16 + 32, v16 + 64, v7 + 32, v10, v6, a6);
      v30 = v7 + 80;
      v31 = v16 + 112;
      v7 += 136;
      v32 = v16 + 80;
      v16 += 136;
      sub_C8CD80(v32, v31, v30, v33, v34, v35);
      *(_BYTE *)(v16 - 8) = *(_BYTE *)(v7 - 8);
    }
    while ( v9 != v7 );
  }
  v36 = v46;
  if ( v46 != v9 )
  {
    while ( 1 )
    {
      if ( *(_BYTE *)(v36 + 108) )
      {
        if ( *(_BYTE *)(v36 + 60) )
          goto LABEL_21;
LABEL_24:
        v37 = *(_QWORD *)(v36 + 40);
        v36 += 136LL;
        _libc_free(v37);
        if ( v36 == v9 )
          break;
      }
      else
      {
        _libc_free(*(_QWORD *)(v36 + 88));
        if ( !*(_BYTE *)(v36 + 60) )
          goto LABEL_24;
LABEL_21:
        v36 += 136LL;
        if ( v36 == v9 )
          break;
      }
    }
  }
  if ( v46 )
    j_j___libc_free_0(v46);
  *a1 = v45;
  a1[1] = v16;
  a1[2] = v43;
  return a1;
}
