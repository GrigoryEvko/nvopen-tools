// Function: sub_3258570
// Address: 0x3258570
//
void __fastcall sub_3258570(__int64 a1, _BYTE *a2, _QWORD *a3, _BYTE *a4)
{
  signed __int64 v4; // r9
  unsigned __int64 v5; // r14
  _BYTE *v8; // r8
  signed __int64 v9; // r15
  _BYTE *v10; // r15
  char *v11; // rax
  __int64 v12; // r9
  __int64 v13; // rax
  _BYTE *v14; // r11
  unsigned __int64 v15; // rax
  bool v16; // cf
  unsigned __int64 v17; // r14
  unsigned __int64 v18; // r14
  char *v19; // r15
  size_t v20; // rcx
  char *v21; // rcx
  char *v22; // rax
  __int64 v23; // r13
  char *v24; // rax
  char *v25; // r13
  __int64 v26; // rcx
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // rcx
  char *v30; // rdi
  __int64 v31; // rax
  unsigned __int64 v32; // r14
  __int64 v33; // rax
  _BYTE *v34; // [rsp-60h] [rbp-60h]
  _BYTE *v35; // [rsp-58h] [rbp-58h]
  signed __int64 v36; // [rsp-50h] [rbp-50h]
  _BYTE *v37; // [rsp-50h] [rbp-50h]
  size_t v38; // [rsp-48h] [rbp-48h]
  _BYTE *v39; // [rsp-48h] [rbp-48h]
  __int64 v40; // [rsp-40h] [rbp-40h]
  _BYTE *v41; // [rsp-40h] [rbp-40h]
  signed __int64 v42; // [rsp-40h] [rbp-40h]
  signed __int64 v43; // [rsp-40h] [rbp-40h]

  if ( a3 == (_QWORD *)a4 )
    return;
  v4 = a4 - (_BYTE *)a3;
  v5 = (a4 - (_BYTE *)a3) >> 3;
  v8 = *(_BYTE **)(a1 + 8);
  if ( *(_QWORD *)(a1 + 16) - (_QWORD)v8 >= (unsigned __int64)(a4 - (_BYTE *)a3) )
  {
    v9 = v8 - a2;
    if ( v4 >= (unsigned __int64)(v8 - a2) )
    {
      v26 = a4 - ((char *)a3 + v9);
      if ( v26 <= 0 )
      {
        v28 = *(_QWORD *)(a1 + 8);
      }
      else
      {
        v27 = 0;
        do
        {
          *(_QWORD *)&v8[8 * v27] = *(_QWORD *)((char *)&a3[v27] + v9);
          ++v27;
        }
        while ( (v26 >> 3) - v27 > 0 );
        v28 = *(_QWORD *)(a1 + 8);
      }
      v29 = v9 >> 3;
      v30 = (char *)(v28 + 8 * (v5 - (v9 >> 3)));
      *(_QWORD *)(a1 + 8) = v30;
      if ( a2 != v8 )
      {
        memmove(v30, a2, v9);
        v30 = *(char **)(a1 + 8);
        v29 = v9 >> 3;
      }
      v31 = 0;
      *(_QWORD *)(a1 + 8) = &v30[v9];
      if ( v9 > 0 )
      {
        do
        {
          *(_QWORD *)&a2[8 * v31] = a3[v31];
          ++v31;
        }
        while ( v29 - v31 > 0 );
      }
    }
    else
    {
      v40 = a4 - (_BYTE *)a3;
      v10 = &v8[-v4];
      v11 = (char *)memmove(*(void **)(a1 + 8), &v8[-v4], a4 - (_BYTE *)a3);
      v12 = v40;
      *(_QWORD *)(a1 + 8) += v40;
      if ( a2 != v10 )
      {
        memmove(&v11[-(v10 - a2)], a2, v10 - a2);
        v12 = v40;
      }
      if ( v12 > 0 )
      {
        v13 = 0;
        do
        {
          *(_QWORD *)&a2[8 * v13] = a3[v13];
          ++v13;
        }
        while ( (__int64)(v5 - v13) > 0 );
      }
    }
    return;
  }
  v14 = *(_BYTE **)a1;
  v15 = (__int64)&v8[-*(_QWORD *)a1] >> 3;
  if ( v5 > 0xFFFFFFFFFFFFFFFLL - v15 )
    sub_4262D8((__int64)"vector::_M_range_insert");
  if ( v5 < v15 )
    v5 = (__int64)&v8[-*(_QWORD *)a1] >> 3;
  v16 = __CFADD__(v15, v5);
  v17 = v15 + v5;
  if ( v16 )
  {
    v32 = 0x7FFFFFFFFFFFFFF8LL;
    goto LABEL_38;
  }
  if ( v17 )
  {
    if ( v17 > 0xFFFFFFFFFFFFFFFLL )
      v17 = 0xFFFFFFFFFFFFFFFLL;
    v32 = 8 * v17;
LABEL_38:
    v43 = a4 - (_BYTE *)a3;
    v33 = sub_22077B0(v32);
    v14 = *(_BYTE **)a1;
    v8 = *(_BYTE **)(a1 + 8);
    v4 = v43;
    v19 = (char *)v33;
    v18 = v33 + v32;
    v20 = (size_t)&a2[-*(_QWORD *)a1];
    if ( a2 == *(_BYTE **)a1 )
      goto LABEL_17;
    goto LABEL_16;
  }
  v18 = 0;
  v19 = 0;
  v20 = a2 - v14;
  if ( a2 != v14 )
  {
LABEL_16:
    v34 = v8;
    v36 = v4;
    v38 = v20;
    v41 = v14;
    memmove(v19, v14, v20);
    v8 = v34;
    v4 = v36;
    v20 = v38;
    v14 = v41;
  }
LABEL_17:
  v21 = &v19[v20];
  if ( v4 > 0 )
  {
    v35 = v14;
    v37 = v8;
    v42 = v4;
    v22 = (char *)memcpy(v21, a3, v4);
    v14 = v35;
    v8 = v37;
    v21 = &v22[v42];
  }
  v23 = v8 - a2;
  if ( a2 != v8 )
  {
    v39 = v14;
    v24 = (char *)memcpy(v21, a2, v8 - a2);
    v14 = v39;
    v21 = v24;
  }
  v25 = &v21[v23];
  if ( v14 )
    j_j___libc_free_0((unsigned __int64)v14);
  *(_QWORD *)a1 = v19;
  *(_QWORD *)(a1 + 8) = v25;
  *(_QWORD *)(a1 + 16) = v18;
}
