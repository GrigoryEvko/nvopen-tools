// Function: sub_1CFD050
// Address: 0x1cfd050
//
void __fastcall sub_1CFD050(__int64 a1, char *a2, unsigned __int64 a3, _QWORD *a4)
{
  __int64 v6; // rcx
  char *v7; // r14
  char *v10; // rbx
  __int64 v11; // rcx
  unsigned __int64 v12; // rax
  size_t v13; // r12
  __int64 v14; // rax
  char *v15; // rsi
  char *v16; // r12
  char *v17; // rdi
  unsigned __int64 v18; // r12
  char *v19; // rax
  char *v20; // r8
  unsigned __int64 v21; // r14
  unsigned __int64 v22; // rax
  bool v23; // cf
  unsigned __int64 v24; // r14
  signed __int64 v25; // rbx
  size_t v26; // r10
  char *v27; // r9
  char *v28; // rax
  __int64 v29; // r12
  __int64 v30; // rdi
  char *v31; // r12
  char *v32; // rax
  size_t v33; // rax
  size_t v34; // rbx
  char *v35; // r12
  __int64 v36; // r14
  __int64 v37; // rax
  char *v38; // [rsp+8h] [rbp-48h]
  __int64 v39; // [rsp+10h] [rbp-40h]
  __int64 v40; // [rsp+10h] [rbp-40h]
  char *v41; // [rsp+10h] [rbp-40h]
  __int64 v42; // [rsp+18h] [rbp-38h]
  __int64 v43; // [rsp+18h] [rbp-38h]
  char *v44; // [rsp+18h] [rbp-38h]
  __int64 v45; // [rsp+18h] [rbp-38h]
  char *v46; // [rsp+18h] [rbp-38h]
  _QWORD *v47; // [rsp+18h] [rbp-38h]

  if ( !a3 )
    return;
  v6 = *(_QWORD *)(a1 + 16);
  v7 = *(char **)(a1 + 8);
  v10 = a2;
  if ( (v6 - (__int64)v7) >> 3 < a3 )
  {
    v20 = *(char **)a1;
    v21 = (__int64)&v7[-*(_QWORD *)a1] >> 3;
    if ( a3 > 0xFFFFFFFFFFFFFFFLL - v21 )
      sub_4262D8((__int64)"vector::_M_fill_insert");
    v22 = v21;
    if ( a3 >= v21 )
      v22 = a3;
    v23 = __CFADD__(v22, v21);
    v24 = v22 + v21;
    v25 = a2 - v20;
    if ( v23 )
    {
      v36 = 0x7FFFFFFFFFFFFFF8LL;
    }
    else
    {
      if ( !v24 )
      {
        v26 = a2 - v20;
        v27 = 0;
LABEL_25:
        v28 = &v27[v25];
        v29 = 8 * a3;
        v30 = *a4;
        do
        {
          *(_QWORD *)v28 = v30;
          v28 += 8;
        }
        while ( &v27[v25 + v29] != v28 );
        v31 = &v27[v26 + v29];
        if ( a2 == v20 )
        {
          v34 = 0;
          v33 = *(_QWORD *)(a1 + 8) - (_QWORD)a2;
          if ( *(char **)(a1 + 8) == a2 )
          {
LABEL_30:
            v35 = &v31[v34];
            if ( !v20 )
            {
LABEL_31:
              *(_QWORD *)a1 = v27;
              *(_QWORD *)(a1 + 8) = v35;
              *(_QWORD *)(a1 + 16) = v24;
              return;
            }
LABEL_33:
            v46 = v27;
            j_j___libc_free_0(v20, v6 - (_QWORD)v20);
            v27 = v46;
            goto LABEL_31;
          }
        }
        else
        {
          v40 = v6;
          v44 = v20;
          v32 = (char *)memmove(v27, v20, v26);
          v20 = v44;
          v27 = v32;
          v6 = v40;
          v33 = *(_QWORD *)(a1 + 8) - (_QWORD)a2;
          if ( a2 == *(char **)(a1 + 8) )
          {
            v35 = &v31[v33];
            goto LABEL_33;
          }
        }
        v38 = v20;
        v41 = v27;
        v34 = v33;
        v45 = v6;
        memcpy(v31, a2, v33);
        v20 = v38;
        v27 = v41;
        v6 = v45;
        goto LABEL_30;
      }
      if ( v24 > 0xFFFFFFFFFFFFFFFLL )
        v24 = 0xFFFFFFFFFFFFFFFLL;
      v36 = 8 * v24;
    }
    v47 = a4;
    v37 = sub_22077B0(v36);
    v20 = *(char **)a1;
    v6 = *(_QWORD *)(a1 + 16);
    a4 = v47;
    v27 = (char *)v37;
    v24 = v37 + v36;
    v26 = (size_t)&a2[-*(_QWORD *)a1];
    goto LABEL_25;
  }
  v11 = *a4;
  v12 = (v7 - a2) >> 3;
  if ( a3 >= v12 )
  {
    v17 = *(char **)(a1 + 8);
    v18 = a3 - v12;
    if ( v18 )
    {
      v17 = &v7[8 * v18];
      if ( v7 != v17 )
      {
        v19 = v7;
        do
        {
          *(_QWORD *)v19 = v11;
          v19 += 8;
        }
        while ( v17 != v19 );
      }
    }
    v43 = v11;
    *(_QWORD *)(a1 + 8) = v17;
    if ( v7 == a2 )
    {
      *(_QWORD *)(a1 + 8) = &v17[v7 - a2];
    }
    else
    {
      memmove(v17, a2, v7 - a2);
      *(_QWORD *)(a1 + 8) += v7 - a2;
      do
      {
        *(_QWORD *)v10 = v43;
        v10 += 8;
      }
      while ( v7 != v10 );
    }
  }
  else
  {
    v13 = 8 * a3;
    v14 = *(_QWORD *)(a1 + 8);
    v15 = &v7[-v13];
    if ( v7 != &v7[-v13] )
    {
      v39 = *a4;
      memmove(*(void **)(a1 + 8), v15, v13);
      v14 = *(_QWORD *)(a1 + 8);
      v11 = v39;
      v15 = &v7[-v13];
    }
    *(_QWORD *)(a1 + 8) = v13 + v14;
    if ( a2 != v15 )
    {
      v42 = v11;
      memmove(&v7[-(v15 - a2)], a2, v15 - a2);
      v11 = v42;
    }
    v16 = &a2[v13];
    if ( a2 != v16 )
    {
      do
      {
        *(_QWORD *)v10 = v11;
        v10 += 8;
      }
      while ( v16 != v10 );
    }
  }
}
