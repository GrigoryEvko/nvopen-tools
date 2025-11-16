// Function: sub_26D33A0
// Address: 0x26d33a0
//
__int64 __fastcall sub_26D33A0(unsigned __int64 *a1, _QWORD *a2, __int64 a3)
{
  __int64 v6; // rsi
  _QWORD *v7; // r12
  signed __int64 v8; // rdi
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // rdx
  char *v11; // rcx
  bool v12; // zf
  char *v13; // rcx
  char *v14; // r14
  __int64 v15; // rax
  unsigned __int64 v16; // rdx
  _BYTE *v17; // rsi
  __int64 v18; // rax
  _BYTE *v19; // rax
  unsigned __int64 v20; // r9
  __int64 v21; // rax
  char *v22; // rdi
  size_t v23; // r15
  char *v24; // rax
  _BYTE *v25; // rax
  __int64 v26; // rax
  char *v27; // rdi
  size_t v28; // r15
  unsigned __int64 v29; // r15
  __int64 i; // r14
  __int64 v31; // rcx
  unsigned __int64 v32; // rdi
  unsigned __int64 v33; // rdi
  unsigned __int64 v34; // rcx
  _QWORD *v35; // rax
  __int64 v36; // rdx
  __int64 v37; // rsi
  unsigned __int64 v38; // rdi
  __int64 v40; // rax
  __int64 v41; // [rsp+8h] [rbp-58h]
  unsigned __int64 v42; // [rsp+8h] [rbp-58h]
  __int64 v43; // [rsp+8h] [rbp-58h]
  unsigned __int64 v44; // [rsp+10h] [rbp-50h]
  __int64 v45; // [rsp+10h] [rbp-50h]
  __int64 v46; // [rsp+10h] [rbp-50h]
  unsigned __int64 v47; // [rsp+18h] [rbp-48h]
  unsigned __int64 v48; // [rsp+20h] [rbp-40h]
  unsigned __int64 v49; // [rsp+20h] [rbp-40h]
  __int64 v50; // [rsp+28h] [rbp-38h]

  v6 = 0x199999999999999LL;
  v7 = (_QWORD *)a1[1];
  v48 = *a1;
  v8 = (signed __int64)v7 - *a1;
  v9 = 0xCCCCCCCCCCCCCCCDLL * (v8 >> 4);
  if ( v9 == 0x199999999999999LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v10 = 1;
  if ( v9 )
    v10 = 0xCCCCCCCCCCCCCCCDLL * (v8 >> 4);
  v47 = v10 - 0x3333333333333333LL * (v8 >> 4);
  v11 = (char *)a2 - v48;
  if ( __CFADD__(v10, v9) )
  {
    v8 = 0x7FFFFFFFFFFFFFD0LL;
    v47 = 0x199999999999999LL;
  }
  else
  {
    if ( v10 == 0x3333333333333333LL * (v8 >> 4) )
    {
      v50 = 0;
      goto LABEL_7;
    }
    if ( v47 <= 0x199999999999999LL )
      v6 = v10 - 0x3333333333333333LL * (v8 >> 4);
    v47 = v6;
    v8 = 80 * v6;
  }
  v43 = a3;
  v40 = sub_22077B0(v8);
  v11 = (char *)a2 - v48;
  a3 = v43;
  v50 = v40;
LABEL_7:
  v12 = &v11[v50] == 0;
  v13 = &v11[v50];
  v14 = v13;
  if ( v12 )
    goto LABEL_19;
  v15 = *(_QWORD *)(a3 + 8);
  v16 = *(_QWORD *)a3;
  *((_QWORD *)v13 + 4) = 0;
  v17 = *(_BYTE **)(a3 + 32);
  *((_QWORD *)v13 + 5) = 0;
  *((_QWORD *)v13 + 1) = v15;
  LOWORD(v15) = *(_WORD *)(a3 + 16);
  *(_QWORD *)v13 = v16;
  *((_WORD *)v13 + 8) = v15;
  v18 = *(_QWORD *)(a3 + 24);
  *((_QWORD *)v13 + 6) = 0;
  *((_QWORD *)v13 + 3) = v18;
  v19 = *(_BYTE **)(a3 + 40);
  v20 = v19 - v17;
  if ( v19 == v17 )
  {
    v23 = 0;
    v22 = 0;
  }
  else
  {
    if ( v20 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_43;
    v41 = a3;
    v44 = *(_QWORD *)(a3 + 40) - (_QWORD)v17;
    v21 = sub_22077B0(v20);
    a3 = v41;
    v20 = v44;
    v22 = (char *)v21;
    v19 = *(_BYTE **)(v41 + 40);
    v17 = *(_BYTE **)(v41 + 32);
    v23 = v19 - v17;
  }
  *((_QWORD *)v14 + 4) = v22;
  *((_QWORD *)v14 + 5) = v22;
  *((_QWORD *)v14 + 6) = &v22[v20];
  if ( v17 != v19 )
  {
    v45 = a3;
    v24 = (char *)memmove(v22, v17, v23);
    a3 = v45;
    v22 = v24;
  }
  v25 = *(_BYTE **)(a3 + 64);
  v17 = *(_BYTE **)(a3 + 56);
  v8 = (signed __int64)&v22[v23];
  v46 = a3;
  *((_QWORD *)v14 + 5) = v8;
  *((_QWORD *)v14 + 7) = 0;
  *((_QWORD *)v14 + 8) = 0;
  *((_QWORD *)v14 + 9) = 0;
  v16 = v25 - v17;
  if ( v25 == v17 )
  {
    v28 = 0;
    v27 = 0;
    goto LABEL_16;
  }
  if ( v16 > 0x7FFFFFFFFFFFFFF8LL )
LABEL_43:
    sub_4261EA(v8, v17, v16);
  v42 = v25 - v17;
  v26 = sub_22077B0(v16);
  v16 = v42;
  v27 = (char *)v26;
  v25 = *(_BYTE **)(v46 + 64);
  v17 = *(_BYTE **)(v46 + 56);
  v28 = v25 - v17;
LABEL_16:
  *((_QWORD *)v14 + 7) = v27;
  *((_QWORD *)v14 + 8) = v27;
  *((_QWORD *)v14 + 9) = &v27[v16];
  if ( v25 != v17 )
    v27 = (char *)memmove(v27, v17, v28);
  *((_QWORD *)v14 + 8) = &v27[v28];
LABEL_19:
  v29 = v48;
  for ( i = v50; (_QWORD *)v29 != a2; i += 80 )
  {
    if ( i )
    {
      *(_QWORD *)i = *(_QWORD *)v29;
      *(_QWORD *)(i + 8) = *(_QWORD *)(v29 + 8);
      *(_BYTE *)(i + 16) = *(_BYTE *)(v29 + 16);
      *(_BYTE *)(i + 17) = *(_BYTE *)(v29 + 17);
      *(_QWORD *)(i + 24) = *(_QWORD *)(v29 + 24);
      *(_QWORD *)(i + 32) = *(_QWORD *)(v29 + 32);
      *(_QWORD *)(i + 40) = *(_QWORD *)(v29 + 40);
      *(_QWORD *)(i + 48) = *(_QWORD *)(v29 + 48);
      v31 = *(_QWORD *)(v29 + 56);
      *(_QWORD *)(v29 + 48) = 0;
      *(_QWORD *)(v29 + 40) = 0;
      *(_QWORD *)(v29 + 32) = 0;
      *(_QWORD *)(i + 56) = v31;
      *(_QWORD *)(i + 64) = *(_QWORD *)(v29 + 64);
      *(_QWORD *)(i + 72) = *(_QWORD *)(v29 + 72);
      *(_QWORD *)(v29 + 72) = 0;
      *(_QWORD *)(v29 + 56) = 0;
    }
    else
    {
      v33 = *(_QWORD *)(v29 + 56);
      if ( v33 )
        j_j___libc_free_0(v33);
    }
    v32 = *(_QWORD *)(v29 + 32);
    if ( v32 )
      j_j___libc_free_0(v32);
    v29 += 80LL;
  }
  v34 = i + 80;
  if ( a2 != v7 )
  {
    v35 = a2;
    v36 = i + 80;
    do
    {
      v37 = *v35;
      v35 += 10;
      v36 += 80;
      *(_QWORD *)(v36 - 80) = v37;
      *(_QWORD *)(v36 - 72) = *(v35 - 9);
      *(_BYTE *)(v36 - 64) = *((_BYTE *)v35 - 64);
      *(_BYTE *)(v36 - 63) = *((_BYTE *)v35 - 63);
      *(_QWORD *)(v36 - 56) = *(v35 - 7);
      *(_QWORD *)(v36 - 48) = *(v35 - 6);
      *(_QWORD *)(v36 - 40) = *(v35 - 5);
      *(_QWORD *)(v36 - 32) = *(v35 - 4);
      *(_QWORD *)(v36 - 24) = *(v35 - 3);
      *(_QWORD *)(v36 - 16) = *(v35 - 2);
      *(_QWORD *)(v36 - 8) = *(v35 - 1);
    }
    while ( v35 != v7 );
    v34 += 16
         * (5 * ((0xCCCCCCCCCCCCCCDLL * ((unsigned __int64)((char *)v35 - (char *)a2 - 80) >> 4)) & 0xFFFFFFFFFFFFFFFLL)
          + 5);
  }
  v38 = v48;
  if ( v48 )
  {
    v49 = v34;
    j_j___libc_free_0(v38);
    v34 = v49;
  }
  a1[1] = v34;
  *a1 = v50;
  a1[2] = v50 + 80 * v47;
  return 80 * v47;
}
