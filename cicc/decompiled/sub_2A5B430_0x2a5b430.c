// Function: sub_2A5B430
// Address: 0x2a5b430
//
void __fastcall sub_2A5B430(__int64 a1, char *a2, char *a3, char *a4)
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
  unsigned __int64 v24; // r9
  size_t v25; // rax
  size_t v26; // r13
  char *v27; // r15
  char *v28; // rsi
  char *v29; // rcx
  char *v30; // rax
  char *v31; // rax
  unsigned __int64 v32; // r14
  char *v33; // r8
  unsigned __int64 v34; // r15
  __int64 v35; // rax
  char *dest; // [rsp+8h] [rbp-58h]
  size_t v37; // [rsp+18h] [rbp-48h]
  size_t v38; // [rsp+18h] [rbp-48h]
  size_t v39; // [rsp+18h] [rbp-48h]
  _BYTE *v40; // [rsp+20h] [rbp-40h]
  size_t v41; // [rsp+20h] [rbp-40h]
  size_t n; // [rsp+28h] [rbp-38h]
  size_t na; // [rsp+28h] [rbp-38h]
  size_t nb; // [rsp+28h] [rbp-38h]

  if ( a3 == a4 )
    return;
  v7 = *(_QWORD *)(a1 + 16);
  v9 = *(char **)(a1 + 8);
  v10 = a4 - a3;
  v12 = (__int64)v10 >> 3;
  if ( v7 - (__int64)v9 < v10 )
  {
    v18 = *(_BYTE **)a1;
    v19 = (__int64)&v9[-*(_QWORD *)a1] >> 3;
    if ( v12 > 0xFFFFFFFFFFFFFFFLL - v19 )
      sub_4262D8((__int64)"vector::_M_range_insert");
    if ( v12 < v19 )
      v12 = v19;
    v20 = __CFADD__(v19, v12);
    v21 = v19 + v12;
    if ( v20 )
    {
      v34 = 0x7FFFFFFFFFFFFFF8LL;
    }
    else
    {
      if ( !v21 )
      {
        na = 0;
        v22 = 0;
        goto LABEL_14;
      }
      if ( v21 > 0xFFFFFFFFFFFFFFFLL )
        v21 = 0xFFFFFFFFFFFFFFFLL;
      v34 = 8 * v21;
    }
    v41 = v10;
    v35 = sub_22077B0(v34);
    v18 = *(_BYTE **)a1;
    v22 = (char *)v35;
    v10 = v41;
    na = v34 + v35;
LABEL_14:
    v23 = &v22[a2 - v18 + v10];
    if ( a2 == v18 )
    {
      v28 = a3;
      v39 = (size_t)v18;
      v26 = 0;
      memcpy(&v22[a2 - v18], v28, v10);
      v24 = v39;
      v25 = *(_QWORD *)(a1 + 8) - (_QWORD)a2;
      if ( *(char **)(a1 + 8) == a2 )
      {
LABEL_17:
        v27 = &v23[v26];
        if ( !v24 )
        {
LABEL_18:
          *(_QWORD *)a1 = v22;
          *(_QWORD *)(a1 + 8) = v27;
          *(_QWORD *)(a1 + 16) = na;
          return;
        }
LABEL_29:
        j_j___libc_free_0(v24);
        goto LABEL_18;
      }
    }
    else
    {
      v40 = v18;
      dest = &v22[a2 - v18];
      v37 = v10;
      memmove(v22, v18, a2 - v18);
      memcpy(dest, a3, v37);
      v24 = (unsigned __int64)v40;
      v25 = *(_QWORD *)(a1 + 8) - (_QWORD)a2;
      if ( a2 == *(char **)(a1 + 8) )
      {
        v27 = &v23[v25];
        goto LABEL_29;
      }
    }
    v38 = v24;
    v26 = v25;
    memcpy(v23, a2, v25);
    v24 = v38;
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
  v29 = &a3[v13];
  v30 = v9;
  if ( a4 != &a3[v13] )
  {
    v31 = (char *)memmove(v9, &a3[v13], a4 - v29);
    v29 = &a3[v13];
    v9 = v31;
    v30 = *(char **)(a1 + 8);
  }
  v32 = v12 - (v13 >> 3);
  v33 = &v30[8 * v32];
  *(_QWORD *)(a1 + 8) = v33;
  if ( a2 != v9 )
  {
    nb = (size_t)v29;
    memmove(&v30[8 * v32], a2, v13);
    v33 = *(char **)(a1 + 8);
    v29 = (char *)nb;
  }
  *(_QWORD *)(a1 + 8) = &v33[v13];
  if ( a3 != v29 )
  {
    v17 = v13;
LABEL_7:
    memmove(a2, a3, v17);
  }
}
