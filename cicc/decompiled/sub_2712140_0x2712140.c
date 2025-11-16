// Function: sub_2712140
// Address: 0x2712140
//
unsigned __int64 *__fastcall sub_2712140(
        unsigned __int64 *a1,
        __int64 a2,
        __int64 *a3,
        __int64 a4,
        __int64 a5,
        __int64 a6)
{
  unsigned __int64 v7; // r12
  __int64 v8; // rcx
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // rdx
  __int64 v11; // r15
  bool v12; // cf
  unsigned __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rbx
  __int64 v16; // r13
  __int64 v17; // rcx
  __int16 v18; // dx
  __int64 v19; // rdx
  unsigned __int64 v20; // r13
  unsigned __int64 i; // rbx
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // r9
  __int64 v25; // rdx
  __int64 v26; // rsi
  __int64 v27; // rdi
  __int64 v28; // rcx
  __int64 v29; // r8
  __int64 v30; // r9
  unsigned __int64 v31; // r14
  unsigned __int64 v32; // rdi
  unsigned __int64 v34; // rbx
  __int64 v35; // rax
  __int64 *v36; // [rsp+0h] [rbp-60h]
  __int64 *v37; // [rsp+8h] [rbp-58h]
  unsigned __int64 v38; // [rsp+10h] [rbp-50h]
  unsigned __int64 v40; // [rsp+20h] [rbp-40h]
  unsigned __int64 v41; // [rsp+28h] [rbp-38h]

  v7 = a1[1];
  v41 = *a1;
  v8 = 0xF0F0F0F0F0F0F0LL;
  v9 = 0xF0F0F0F0F0F0F0F1LL * ((__int64)(v7 - *a1) >> 3);
  if ( v9 == 0xF0F0F0F0F0F0F0LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v10 = 1;
  v11 = a2;
  if ( v9 )
    v10 = 0xF0F0F0F0F0F0F0F1LL * ((__int64)(v7 - *a1) >> 3);
  v12 = __CFADD__(v10, v9);
  v13 = v10 - 0xF0F0F0F0F0F0F0FLL * ((__int64)(v7 - *a1) >> 3);
  v14 = a2 - v41;
  if ( v12 )
  {
    v34 = 0x7FFFFFFFFFFFFF80LL;
  }
  else
  {
    if ( !v13 )
    {
      v38 = 0;
      v15 = 136;
      v40 = 0;
      goto LABEL_7;
    }
    if ( v13 > 0xF0F0F0F0F0F0F0LL )
      v13 = 0xF0F0F0F0F0F0F0LL;
    v34 = 136 * v13;
  }
  v36 = a3;
  v35 = sub_22077B0(v34);
  v14 = a2 - v41;
  a3 = v36;
  v40 = v35;
  v38 = v35 + v34;
  v15 = v35 + 136;
LABEL_7:
  v16 = v40 + v14;
  if ( v40 + v14 )
  {
    v17 = *a3;
    v37 = a3;
    *(_BYTE *)(v16 + 10) = *((_BYTE *)a3 + 10);
    v18 = *((_WORD *)a3 + 8);
    *(_QWORD *)v16 = v17;
    LOWORD(v17) = *((_WORD *)a3 + 4);
    *(_WORD *)(v16 + 16) = v18;
    v19 = a3[3];
    *(_WORD *)(v16 + 8) = v17;
    *(_QWORD *)(v16 + 24) = v19;
    sub_C8CF70(v16 + 32, (void *)(v16 + 64), 2, (__int64)(a3 + 8), (__int64)(a3 + 4));
    sub_C8CF70(v16 + 80, (void *)(v16 + 112), 2, (__int64)(v37 + 14), (__int64)(v37 + 10));
    *(_BYTE *)(v16 + 128) = *((_BYTE *)v37 + 128);
  }
  v20 = v41;
  if ( a2 != v41 )
  {
    for ( i = v40; ; i += 136LL )
    {
      if ( i )
      {
        *(_QWORD *)i = *(_QWORD *)v20;
        *(_BYTE *)(i + 8) = *(_BYTE *)(v20 + 8);
        *(_BYTE *)(i + 9) = *(_BYTE *)(v20 + 9);
        *(_BYTE *)(i + 10) = *(_BYTE *)(v20 + 10);
        *(_BYTE *)(i + 16) = *(_BYTE *)(v20 + 16);
        *(_BYTE *)(i + 17) = *(_BYTE *)(v20 + 17);
        *(_QWORD *)(i + 24) = *(_QWORD *)(v20 + 24);
        sub_C8CD80(i + 32, i + 64, v20 + 32, v8, a5, a6);
        sub_C8CD80(i + 80, i + 112, v20 + 80, v22, v23, v24);
        *(_BYTE *)(i + 128) = *(_BYTE *)(v20 + 128);
      }
      v20 += 136LL;
      if ( a2 == v20 )
        break;
    }
    v15 = i + 272;
  }
  if ( a2 != v7 )
  {
    do
    {
      *(_QWORD *)v15 = *(_QWORD *)v11;
      *(_BYTE *)(v15 + 8) = *(_BYTE *)(v11 + 8);
      *(_BYTE *)(v15 + 9) = *(_BYTE *)(v11 + 9);
      *(_BYTE *)(v15 + 10) = *(_BYTE *)(v11 + 10);
      *(_BYTE *)(v15 + 16) = *(_BYTE *)(v11 + 16);
      *(_BYTE *)(v15 + 17) = *(_BYTE *)(v11 + 17);
      *(_QWORD *)(v15 + 24) = *(_QWORD *)(v11 + 24);
      sub_C8CD80(v15 + 32, v15 + 64, v11 + 32, v8, a5, a6);
      v25 = v11 + 80;
      v26 = v15 + 112;
      v11 += 136;
      v27 = v15 + 80;
      v15 += 136;
      sub_C8CD80(v27, v26, v25, v28, v29, v30);
      *(_BYTE *)(v15 - 8) = *(_BYTE *)(v11 - 8);
    }
    while ( v7 != v11 );
  }
  v31 = v41;
  if ( v41 != v7 )
  {
    while ( 1 )
    {
      if ( *(_BYTE *)(v31 + 108) )
      {
        if ( *(_BYTE *)(v31 + 60) )
          goto LABEL_21;
LABEL_24:
        v32 = *(_QWORD *)(v31 + 40);
        v31 += 136LL;
        _libc_free(v32);
        if ( v31 == v7 )
          break;
      }
      else
      {
        _libc_free(*(_QWORD *)(v31 + 88));
        if ( !*(_BYTE *)(v31 + 60) )
          goto LABEL_24;
LABEL_21:
        v31 += 136LL;
        if ( v31 == v7 )
          break;
      }
    }
  }
  if ( v41 )
    j_j___libc_free_0(v41);
  *a1 = v40;
  a1[1] = v15;
  a1[2] = v38;
  return a1;
}
