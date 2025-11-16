// Function: sub_29E54B0
// Address: 0x29e54b0
//
void __fastcall sub_29E54B0(__int64 a1, char *a2, unsigned __int64 a3, __int64 *a4)
{
  __int64 *v5; // rdx
  char *v6; // r14
  char *v9; // rbx
  __int64 v10; // rcx
  unsigned __int64 v11; // rax
  size_t v12; // r12
  __int64 v13; // rax
  char *v14; // rsi
  char *v15; // r12
  char *v16; // rdi
  unsigned __int64 v17; // r12
  char *v18; // rax
  char *v19; // r8
  unsigned __int64 v20; // r14
  unsigned __int64 v21; // rax
  bool v22; // cf
  unsigned __int64 v23; // r14
  signed __int64 v24; // rbx
  size_t v25; // r10
  char *v26; // r9
  char *v27; // rax
  __int64 v28; // r12
  __int64 v29; // rdi
  char *v30; // r12
  char *v31; // rax
  size_t v32; // rax
  size_t v33; // rbx
  char *v34; // r12
  unsigned __int64 v35; // r14
  __int64 v36; // rax
  char *v37; // [rsp+8h] [rbp-48h]
  __int64 v38; // [rsp+10h] [rbp-40h]
  char *v39; // [rsp+10h] [rbp-40h]
  __int64 v40; // [rsp+18h] [rbp-38h]
  __int64 v41; // [rsp+18h] [rbp-38h]
  char *v42; // [rsp+18h] [rbp-38h]
  char *v43; // [rsp+18h] [rbp-38h]

  if ( !a3 )
    return;
  v5 = a4;
  v6 = *(char **)(a1 + 8);
  v9 = a2;
  if ( (__int64)(*(_QWORD *)(a1 + 16) - (_QWORD)v6) >> 3 < a3 )
  {
    v19 = *(char **)a1;
    v20 = (__int64)&v6[-*(_QWORD *)a1] >> 3;
    if ( a3 > 0xFFFFFFFFFFFFFFFLL - v20 )
      sub_4262D8((__int64)"vector::_M_fill_insert");
    v21 = v20;
    if ( a3 >= v20 )
      v21 = a3;
    v22 = __CFADD__(v21, v20);
    v23 = v21 + v20;
    v24 = a2 - v19;
    if ( v22 )
    {
      v35 = 0x7FFFFFFFFFFFFFF8LL;
    }
    else
    {
      if ( !v23 )
      {
        v25 = a2 - v19;
        v26 = 0;
LABEL_25:
        v27 = &v26[v24];
        v28 = 8 * a3;
        v29 = *v5;
        do
        {
          *(_QWORD *)v27 = v29;
          v27 += 8;
        }
        while ( &v26[v24 + v28] != v27 );
        v30 = &v26[v25 + v28];
        if ( a2 == v19 )
        {
          v33 = 0;
          v32 = *(_QWORD *)(a1 + 8) - (_QWORD)a2;
          if ( *(char **)(a1 + 8) == a2 )
          {
LABEL_30:
            v34 = &v30[v33];
            if ( !v19 )
            {
LABEL_31:
              *(_QWORD *)a1 = v26;
              *(_QWORD *)(a1 + 8) = v34;
              *(_QWORD *)(a1 + 16) = v23;
              return;
            }
LABEL_33:
            v43 = v26;
            j_j___libc_free_0((unsigned __int64)v19);
            v26 = v43;
            goto LABEL_31;
          }
        }
        else
        {
          v42 = v19;
          v31 = (char *)memmove(v26, v19, v25);
          v19 = v42;
          v26 = v31;
          v32 = *(_QWORD *)(a1 + 8) - (_QWORD)a2;
          if ( a2 == *(char **)(a1 + 8) )
          {
            v34 = &v30[v32];
            goto LABEL_33;
          }
        }
        v37 = v19;
        v39 = v26;
        v33 = v32;
        memcpy(v30, a2, v32);
        v19 = v37;
        v26 = v39;
        goto LABEL_30;
      }
      if ( v23 > 0xFFFFFFFFFFFFFFFLL )
        v23 = 0xFFFFFFFFFFFFFFFLL;
      v35 = 8 * v23;
    }
    v36 = sub_22077B0(v35);
    v19 = *(char **)a1;
    v5 = a4;
    v26 = (char *)v36;
    v23 = v36 + v35;
    v25 = (size_t)&a2[-*(_QWORD *)a1];
    goto LABEL_25;
  }
  v10 = *a4;
  v11 = (v6 - a2) >> 3;
  if ( a3 >= v11 )
  {
    v16 = *(char **)(a1 + 8);
    v17 = a3 - v11;
    if ( v17 )
    {
      v16 = &v6[8 * v17];
      if ( v6 != v16 )
      {
        v18 = v6;
        do
        {
          *(_QWORD *)v18 = v10;
          v18 += 8;
        }
        while ( v16 != v18 );
      }
    }
    v41 = v10;
    *(_QWORD *)(a1 + 8) = v16;
    if ( v6 == a2 )
    {
      *(_QWORD *)(a1 + 8) = &v16[v6 - a2];
    }
    else
    {
      memmove(v16, a2, v6 - a2);
      *(_QWORD *)(a1 + 8) += v6 - a2;
      do
      {
        *(_QWORD *)v9 = v41;
        v9 += 8;
      }
      while ( v6 != v9 );
    }
  }
  else
  {
    v12 = 8 * a3;
    v13 = *(_QWORD *)(a1 + 8);
    v14 = &v6[-v12];
    if ( v6 != &v6[-v12] )
    {
      v38 = *v5;
      memmove(*(void **)(a1 + 8), v14, v12);
      v13 = *(_QWORD *)(a1 + 8);
      v10 = v38;
      v14 = &v6[-v12];
    }
    *(_QWORD *)(a1 + 8) = v12 + v13;
    if ( a2 != v14 )
    {
      v40 = v10;
      memmove(&v6[-(v14 - a2)], a2, v14 - a2);
      v10 = v40;
    }
    v15 = &a2[v12];
    if ( a2 != v15 )
    {
      do
      {
        *(_QWORD *)v9 = v10;
        v9 += 8;
      }
      while ( v15 != v9 );
    }
  }
}
