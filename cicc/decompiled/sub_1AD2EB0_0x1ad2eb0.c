// Function: sub_1AD2EB0
// Address: 0x1ad2eb0
//
void __fastcall sub_1AD2EB0(__int64 a1, char *a2, char *a3, char *a4)
{
  char *v4; // r8
  __int64 v7; // rcx
  char *v8; // r12
  char *v9; // rbx
  unsigned __int64 v10; // rsi
  __int64 v11; // r10
  char *v12; // r9
  unsigned __int64 v13; // r14
  signed __int64 v14; // r10
  unsigned __int64 v15; // rdi
  size_t v16; // rdx
  char *v17; // rax
  char *v18; // r13
  char *v19; // rax
  __int64 v20; // rax
  char *v21; // r14
  unsigned __int64 v22; // rax
  bool v23; // cf
  unsigned __int64 v24; // rsi
  __int64 v25; // r12
  char *v26; // r11
  size_t v27; // rdx
  char *v28; // rax
  char *v29; // rsi
  char *v30; // rax
  char *v31; // rdx
  unsigned __int64 v32; // r13
  signed __int64 v33; // rbx
  __int64 v34; // r13
  __int64 v35; // rcx
  void *v36; // rax
  __int64 v37; // rbx
  __int64 v38; // r14
  char *v39; // rax
  char *v40; // rdx
  char *v41; // rax
  char *v42; // rdi
  unsigned __int64 v43; // rax
  __int64 v44; // rdx
  __int64 v45; // r12
  __int64 v46; // rax
  char *v47; // [rsp-58h] [rbp-58h]
  char *v48; // [rsp-50h] [rbp-50h]
  char *v49; // [rsp-50h] [rbp-50h]
  __int64 v50; // [rsp-48h] [rbp-48h]
  __int64 v51; // [rsp-48h] [rbp-48h]
  char *v52; // [rsp-48h] [rbp-48h]
  __int64 v53; // [rsp-40h] [rbp-40h]
  __int64 v54; // [rsp-40h] [rbp-40h]
  size_t v55; // [rsp-40h] [rbp-40h]
  __int64 v56; // [rsp-40h] [rbp-40h]
  char *v57; // [rsp-40h] [rbp-40h]
  signed __int64 v58; // [rsp-40h] [rbp-40h]
  char *v59; // [rsp-40h] [rbp-40h]

  if ( a3 == a4 )
    return;
  v4 = a2;
  v7 = a4 - a3;
  v8 = a2;
  v9 = a3;
  v10 = 0xAAAAAAAAAAAAAAABLL * (v7 >> 3);
  v11 = *(_QWORD *)(a1 + 16);
  v12 = *(char **)(a1 + 8);
  v13 = v10;
  if ( v10 <= (v11 - (__int64)v12) >> 3 )
  {
    v14 = v12 - v4;
    v15 = (v12 - v4) >> 3;
    if ( v10 >= v15 )
    {
      v38 = 24 * v15;
      v39 = &a3[24 * v15];
      if ( a4 == v39 )
      {
        v41 = v12;
      }
      else
      {
        v40 = v12;
        do
        {
          if ( v40 )
            *(_QWORD *)v40 = *(_QWORD *)v39;
          v39 += 24;
          v40 += 8;
        }
        while ( a4 != v39 );
        v41 = *(char **)(a1 + 8);
      }
      v42 = &v41[8 * (v10 - v15)];
      *(_QWORD *)(a1 + 8) = v42;
      if ( v4 != v12 )
      {
        v58 = v12 - v4;
        memmove(v42, v4, v12 - v4);
        v42 = *(char **)(a1 + 8);
        v14 = v58;
      }
      *(_QWORD *)(a1 + 8) = &v42[v14];
      v43 = 0xAAAAAAAAAAAAAAABLL * (v38 >> 3);
      if ( v38 > 0 )
      {
        do
        {
          v44 = *(_QWORD *)v9;
          v8 += 8;
          v9 += 24;
          *((_QWORD *)v8 - 1) = v44;
          --v43;
        }
        while ( v43 );
      }
    }
    else
    {
      v16 = 0x5555555555555558LL * (v7 >> 3);
      v17 = v12;
      v18 = &v12[-v16];
      if ( v12 != &v12[-v16] )
      {
        v48 = v4;
        v50 = v7;
        v53 = 0x5555555555555558LL * (v7 >> 3);
        v19 = (char *)memmove(v12, &v12[-v53], v16);
        v4 = v48;
        v7 = v50;
        v12 = v19;
        v16 = v53;
        v17 = *(char **)(a1 + 8);
      }
      *(_QWORD *)(a1 + 8) = &v17[v16];
      if ( v4 != v18 )
      {
        v54 = v7;
        memmove(&v12[-(v18 - v4)], v4, v18 - v4);
        v7 = v54;
      }
      if ( v7 > 0 )
      {
        do
        {
          v20 = *(_QWORD *)v9;
          v8 += 8;
          v9 += 24;
          *((_QWORD *)v8 - 1) = v20;
          --v13;
        }
        while ( v13 );
      }
    }
    return;
  }
  v21 = *(char **)a1;
  v22 = (__int64)&v12[-*(_QWORD *)a1] >> 3;
  if ( v10 > 0xFFFFFFFFFFFFFFFLL - v22 )
    sub_4262D8((__int64)"vector::_M_range_insert");
  if ( v10 < v22 )
    v10 = (__int64)&v12[-*(_QWORD *)a1] >> 3;
  v23 = __CFADD__(v22, v10);
  v24 = v22 + v10;
  if ( v23 )
  {
    v45 = 0x7FFFFFFFFFFFFFF8LL;
  }
  else
  {
    if ( !v24 )
    {
      v25 = 0;
      v26 = 0;
      v27 = v4 - v21;
      if ( v4 == v21 )
        goto LABEL_18;
      goto LABEL_17;
    }
    if ( v24 > 0xFFFFFFFFFFFFFFFLL )
      v24 = 0xFFFFFFFFFFFFFFFLL;
    v45 = 8 * v24;
  }
  v59 = v4;
  v46 = sub_22077B0(v45);
  v4 = v59;
  v21 = *(char **)a1;
  v12 = *(char **)(a1 + 8);
  v11 = *(_QWORD *)(a1 + 16);
  v26 = (char *)v46;
  v25 = v46 + v45;
  v27 = (size_t)&v59[-*(_QWORD *)a1];
  if ( v59 != *(char **)a1 )
  {
LABEL_17:
    v47 = v4;
    v49 = v12;
    v51 = v11;
    v55 = v27;
    v28 = (char *)memmove(v26, v21, v27);
    v4 = v47;
    v12 = v49;
    v11 = v51;
    v27 = v55;
    v26 = v28;
  }
LABEL_18:
  v29 = &v26[v27];
  v30 = v9;
  v31 = &v26[v27];
  do
  {
    if ( v31 )
      *(_QWORD *)v31 = *(_QWORD *)v30;
    v30 += 24;
    v31 += 8;
  }
  while ( a4 != v30 );
  v32 = a4 - 24 - v9;
  v33 = v12 - v4;
  v34 = 0xAAAAAAAAAAAAAABLL * (v32 >> 3);
  v35 = (__int64)&v29[8 * v34 + 8];
  if ( v4 != v12 )
  {
    v52 = v26;
    v56 = v11;
    v36 = memcpy(&v29[8 * v34 + 8], v4, v12 - v4);
    v26 = v52;
    v11 = v56;
    v35 = (__int64)v36;
  }
  v37 = v35 + v33;
  if ( v21 )
  {
    v57 = v26;
    j_j___libc_free_0(v21, v11 - (_QWORD)v21);
    v26 = v57;
  }
  *(_QWORD *)a1 = v26;
  *(_QWORD *)(a1 + 8) = v37;
  *(_QWORD *)(a1 + 16) = v25;
}
