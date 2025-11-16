// Function: sub_1038AF0
// Address: 0x1038af0
//
void __fastcall sub_1038AF0(__int64 a1, char *a2, char *a3, char *a4)
{
  __int64 v7; // rcx
  char *v9; // rdi
  size_t v10; // r8
  unsigned __int64 v12; // r14
  signed __int64 v13; // r15
  char *v14; // r14
  char *v15; // rax
  size_t v16; // r8
  size_t v17; // rdx
  _BYTE *v18; // r9
  unsigned __int64 v19; // rdi
  bool v20; // cf
  unsigned __int64 v21; // r14
  char *v22; // r14
  char *v23; // r15
  size_t v24; // r9
  __int64 v25; // rcx
  size_t v26; // rax
  size_t v27; // r13
  char *v28; // r15
  char *v29; // rsi
  char *v30; // rcx
  char *v31; // rax
  char *v32; // rax
  char *v33; // r14
  __int64 v34; // r15
  __int64 v35; // rax
  char *dest; // [rsp+8h] [rbp-58h]
  __int64 v37; // [rsp+10h] [rbp-50h]
  size_t v38; // [rsp+18h] [rbp-48h]
  size_t v39; // [rsp+18h] [rbp-48h]
  size_t v40; // [rsp+18h] [rbp-48h]
  _BYTE *v41; // [rsp+20h] [rbp-40h]
  __int64 v42; // [rsp+20h] [rbp-40h]
  __int64 v43; // [rsp+20h] [rbp-40h]
  size_t v44; // [rsp+20h] [rbp-40h]
  size_t n; // [rsp+28h] [rbp-38h]
  size_t na; // [rsp+28h] [rbp-38h]
  size_t nb; // [rsp+28h] [rbp-38h]

  if ( a3 == a4 )
    return;
  v7 = *(_QWORD *)(a1 + 16);
  v9 = *(char **)(a1 + 8);
  v10 = a4 - a3;
  v12 = (__int64)v10 >> 4;
  if ( v7 - (__int64)v9 < v10 )
  {
    v18 = *(_BYTE **)a1;
    v19 = (__int64)&v9[-*(_QWORD *)a1] >> 4;
    if ( v12 > 0x7FFFFFFFFFFFFFFLL - v19 )
      sub_4262D8((__int64)"vector::_M_range_insert");
    if ( v12 < v19 )
      v12 = v19;
    v20 = __CFADD__(v19, v12);
    v21 = v19 + v12;
    if ( v20 )
    {
      v34 = 0x7FFFFFFFFFFFFFF0LL;
    }
    else
    {
      if ( !v21 )
      {
        na = 0;
        v22 = 0;
        goto LABEL_14;
      }
      if ( v21 > 0x7FFFFFFFFFFFFFFLL )
        v21 = 0x7FFFFFFFFFFFFFFLL;
      v34 = 16 * v21;
    }
    v44 = v10;
    v35 = sub_22077B0(v34);
    v18 = *(_BYTE **)a1;
    v7 = *(_QWORD *)(a1 + 16);
    v22 = (char *)v35;
    v10 = v44;
    na = v34 + v35;
LABEL_14:
    v23 = &v22[a2 - v18 + v10];
    if ( a2 == v18 )
    {
      v29 = a3;
      v40 = (size_t)v18;
      v43 = v7;
      v27 = 0;
      memcpy(&v22[a2 - v18], v29, v10);
      v25 = v43;
      v24 = v40;
      v26 = *(_QWORD *)(a1 + 8) - (_QWORD)a2;
      if ( *(char **)(a1 + 8) == a2 )
      {
LABEL_17:
        v28 = &v23[v27];
        if ( !v24 )
        {
LABEL_18:
          *(_QWORD *)a1 = v22;
          *(_QWORD *)(a1 + 8) = v28;
          *(_QWORD *)(a1 + 16) = na;
          return;
        }
LABEL_29:
        j_j___libc_free_0(v24, v25 - v24);
        goto LABEL_18;
      }
    }
    else
    {
      v37 = v7;
      v41 = v18;
      dest = &v22[a2 - v18];
      v38 = v10;
      memmove(v22, v18, a2 - v18);
      memcpy(dest, a3, v38);
      v24 = (size_t)v41;
      v25 = v37;
      v26 = *(_QWORD *)(a1 + 8) - (_QWORD)a2;
      if ( a2 == *(char **)(a1 + 8) )
      {
        v28 = &v23[v26];
        goto LABEL_29;
      }
    }
    v39 = v24;
    v42 = v25;
    v27 = v26;
    memcpy(v23, a2, v26);
    v24 = v39;
    v25 = v42;
    goto LABEL_17;
  }
  v13 = v9 - a2;
  if ( v10 < v9 - a2 )
  {
    n = v10;
    v14 = &v9[-v10];
    v15 = (char *)memmove(v9, &v9[-v10], v10);
    v16 = n;
    *(_QWORD *)(a1 + 8) += n;
    if ( a2 != v14 )
    {
      memmove(&v15[-(v14 - a2)], a2, v14 - a2);
      v16 = n;
    }
    v17 = v16;
    goto LABEL_7;
  }
  v30 = &a3[v13];
  v31 = v9;
  if ( a4 != &a3[v13] )
  {
    v32 = (char *)memmove(v9, &a3[v13], a4 - v30);
    v30 = &a3[v13];
    v9 = v32;
    v31 = *(char **)(a1 + 8);
  }
  v33 = &v31[16 * (v12 - (v13 >> 4))];
  *(_QWORD *)(a1 + 8) = v33;
  if ( a2 != v9 )
  {
    nb = (size_t)v30;
    memmove(v33, a2, v13);
    v33 = *(char **)(a1 + 8);
    v30 = (char *)nb;
  }
  *(_QWORD *)(a1 + 8) = &v33[v13];
  if ( a3 != v30 )
  {
    v17 = v13;
LABEL_7:
    memmove(a2, a3, v17);
  }
}
