// Function: sub_263FF00
// Address: 0x263ff00
//
void __fastcall sub_263FF00(__int64 a1, char *a2, char *a3, char *a4)
{
  __int64 v7; // rcx
  char *v9; // rdi
  size_t v10; // r8
  unsigned __int64 v12; // r12
  signed __int64 v13; // r15
  char *v14; // r12
  char *v15; // rax
  size_t v16; // r8
  size_t v17; // rdx
  __int64 v18; // rax
  _BYTE *v19; // r9
  unsigned __int64 v20; // rdi
  bool v21; // cf
  unsigned __int64 v22; // rdi
  char *v23; // r12
  char *v24; // r15
  unsigned __int64 v25; // r9
  size_t v26; // rax
  size_t v27; // r14
  char *v28; // r15
  char *v29; // rsi
  char *v30; // rcx
  char *v31; // rax
  char *v32; // rax
  char *v33; // r12
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
  v12 = (__int64)v10 >> 4;
  if ( v7 - (__int64)v9 < v10 )
  {
    v18 = 0x7FFFFFFFFFFFFFFLL;
    v19 = *(_BYTE **)a1;
    v20 = (__int64)&v9[-*(_QWORD *)a1] >> 4;
    if ( v12 > 0x7FFFFFFFFFFFFFFLL - v20 )
      sub_4262D8((__int64)"vector::_M_range_insert");
    if ( v12 < v20 )
      v12 = v20;
    v21 = __CFADD__(v12, v20);
    v22 = v12 + v20;
    if ( v21 )
    {
      v34 = 0x7FFFFFFFFFFFFFF0LL;
    }
    else
    {
      if ( !v22 )
      {
        na = 0;
        v23 = 0;
        goto LABEL_14;
      }
      if ( v22 <= 0x7FFFFFFFFFFFFFFLL )
        v18 = v22;
      v34 = 16 * v18;
    }
    v41 = v10;
    v35 = sub_22077B0(v34);
    v19 = *(_BYTE **)a1;
    v23 = (char *)v35;
    v10 = v41;
    na = v34 + v35;
LABEL_14:
    v24 = &v23[a2 - v19 + v10];
    if ( a2 == v19 )
    {
      v29 = a3;
      v39 = (size_t)v19;
      v27 = 0;
      memcpy(&v23[a2 - v19], v29, v10);
      v25 = v39;
      v26 = *(_QWORD *)(a1 + 8) - (_QWORD)a2;
      if ( *(char **)(a1 + 8) == a2 )
      {
LABEL_17:
        v28 = &v24[v27];
        if ( !v25 )
        {
LABEL_18:
          *(_QWORD *)a1 = v23;
          *(_QWORD *)(a1 + 8) = v28;
          *(_QWORD *)(a1 + 16) = na;
          return;
        }
LABEL_29:
        j_j___libc_free_0(v25);
        goto LABEL_18;
      }
    }
    else
    {
      v40 = v19;
      dest = &v23[a2 - v19];
      v37 = v10;
      memmove(v23, v19, a2 - v19);
      memcpy(dest, a3, v37);
      v25 = (unsigned __int64)v40;
      v26 = *(_QWORD *)(a1 + 8) - (_QWORD)a2;
      if ( a2 == *(char **)(a1 + 8) )
      {
        v28 = &v24[v26];
        goto LABEL_29;
      }
    }
    v38 = v25;
    v27 = v26;
    memcpy(v24, a2, v26);
    v25 = v38;
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
