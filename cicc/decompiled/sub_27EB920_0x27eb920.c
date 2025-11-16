// Function: sub_27EB920
// Address: 0x27eb920
//
void __fastcall sub_27EB920(__int64 a1, char *a2, char *a3, char *a4)
{
  __int64 v4; // r10
  unsigned __int64 v9; // rcx
  unsigned __int64 v10; // rbx
  char *v11; // r8
  size_t v12; // r9
  unsigned __int64 v13; // rdi
  size_t v14; // rdx
  char *v15; // rax
  char *v16; // r12
  char *v17; // rax
  __int64 v18; // rax
  char *v19; // r10
  unsigned __int64 v20; // rax
  bool v21; // cf
  unsigned __int64 v22; // rcx
  unsigned __int64 v23; // rbx
  char *v24; // r11
  size_t v25; // rdx
  char *v26; // rax
  char *v27; // rsi
  char *v28; // rax
  char *v29; // rdx
  char *v30; // rcx
  signed __int64 v31; // r12
  char *v32; // rax
  char *v33; // r12
  __int64 v34; // rbx
  char *v35; // rax
  char *v36; // rdx
  char *v37; // rax
  char *v38; // rdi
  __int64 v39; // rdx
  __int64 v40; // rax
  unsigned __int64 v41; // rbx
  __int64 v42; // rax
  char *v43; // [rsp-58h] [rbp-58h]
  char *v44; // [rsp-50h] [rbp-50h]
  __int64 v45; // [rsp-48h] [rbp-48h]
  size_t v46; // [rsp-48h] [rbp-48h]
  char *v47; // [rsp-48h] [rbp-48h]
  __int64 v48; // [rsp-40h] [rbp-40h]
  __int64 v49; // [rsp-40h] [rbp-40h]
  char *v50; // [rsp-40h] [rbp-40h]
  char *v51; // [rsp-40h] [rbp-40h]
  size_t v52; // [rsp-40h] [rbp-40h]

  if ( a3 == a4 )
    return;
  v4 = a4 - a3;
  v9 = (a4 - a3) >> 5;
  v10 = v9;
  v11 = *(char **)(a1 + 8);
  if ( v9 <= (__int64)(*(_QWORD *)(a1 + 16) - (_QWORD)v11) >> 3 )
  {
    v12 = v11 - a2;
    v13 = (v11 - a2) >> 3;
    if ( v9 >= v13 )
    {
      v34 = 32 * v13;
      v35 = &a3[32 * v13];
      if ( a4 == v35 )
      {
        v37 = v11;
      }
      else
      {
        v36 = v11;
        do
        {
          if ( v36 )
            *(_QWORD *)v36 = *(_QWORD *)v35;
          v35 += 32;
          v36 += 8;
        }
        while ( a4 != v35 );
        v37 = *(char **)(a1 + 8);
      }
      v38 = &v37[8 * (v9 - v13)];
      *(_QWORD *)(a1 + 8) = v38;
      if ( a2 != v11 )
      {
        v52 = v11 - a2;
        memmove(v38, a2, v12);
        v38 = *(char **)(a1 + 8);
        v12 = v52;
      }
      *(_QWORD *)(a1 + 8) = &v38[v12];
      v39 = v34 >> 5;
      if ( v34 > 0 )
      {
        v40 = 0;
        do
        {
          *(_QWORD *)&a2[v40] = *(_QWORD *)&a3[4 * v40];
          v40 += 8;
          --v39;
        }
        while ( v39 );
      }
    }
    else
    {
      v14 = 8 * v9;
      v15 = v11;
      v16 = &v11[-8 * v9];
      if ( v11 != v16 )
      {
        v45 = v4;
        v48 = 8 * v9;
        v17 = (char *)memmove(v11, &v11[-8 * v9], v14);
        v4 = v45;
        v14 = v48;
        v11 = v17;
        v15 = *(char **)(a1 + 8);
      }
      *(_QWORD *)(a1 + 8) = &v15[v14];
      if ( a2 != v16 )
      {
        v49 = v4;
        memmove(&v11[-(v16 - a2)], a2, v16 - a2);
        v4 = v49;
      }
      if ( v4 > 0 )
      {
        v18 = 0;
        do
        {
          *(_QWORD *)&a2[v18] = *(_QWORD *)&a3[4 * v18];
          v18 += 8;
          --v10;
        }
        while ( v10 );
      }
    }
    return;
  }
  v19 = *(char **)a1;
  v20 = (__int64)&v11[-*(_QWORD *)a1] >> 3;
  if ( v9 > 0xFFFFFFFFFFFFFFFLL - v20 )
    sub_4262D8((__int64)"vector::_M_range_insert");
  if ( v9 < v20 )
    v9 = (__int64)&v11[-*(_QWORD *)a1] >> 3;
  v21 = __CFADD__(v20, v9);
  v22 = v20 + v9;
  if ( v21 )
  {
    v41 = 0x7FFFFFFFFFFFFFF8LL;
  }
  else
  {
    if ( !v22 )
    {
      v23 = 0;
      v24 = 0;
      v25 = a2 - v19;
      if ( a2 == v19 )
        goto LABEL_19;
      goto LABEL_18;
    }
    if ( v22 > 0xFFFFFFFFFFFFFFFLL )
      v22 = 0xFFFFFFFFFFFFFFFLL;
    v41 = 8 * v22;
  }
  v42 = sub_22077B0(v41);
  v19 = *(char **)a1;
  v11 = *(char **)(a1 + 8);
  v24 = (char *)v42;
  v23 = v42 + v41;
  v25 = (size_t)&a2[-*(_QWORD *)a1];
  if ( a2 != *(char **)a1 )
  {
LABEL_18:
    v43 = v11;
    v46 = v25;
    v50 = v19;
    v26 = (char *)memmove(v24, v19, v25);
    v11 = v43;
    v25 = v46;
    v19 = v50;
    v24 = v26;
  }
LABEL_19:
  v27 = &v24[v25];
  v28 = a3;
  v29 = &v24[v25];
  do
  {
    if ( v29 )
      *(_QWORD *)v29 = *(_QWORD *)v28;
    v28 += 32;
    v29 += 8;
  }
  while ( a4 != v28 );
  v30 = &v27[8 * ((unsigned __int64)(a4 - 32 - a3) >> 5) + 8];
  v31 = v11 - a2;
  if ( a2 != v11 )
  {
    v44 = v19;
    v47 = v24;
    v32 = (char *)memcpy(v30, a2, v11 - a2);
    v19 = v44;
    v24 = v47;
    v30 = v32;
  }
  v33 = &v30[v31];
  if ( v19 )
  {
    v51 = v24;
    j_j___libc_free_0((unsigned __int64)v19);
    v24 = v51;
  }
  *(_QWORD *)a1 = v24;
  *(_QWORD *)(a1 + 8) = v33;
  *(_QWORD *)(a1 + 16) = v23;
}
