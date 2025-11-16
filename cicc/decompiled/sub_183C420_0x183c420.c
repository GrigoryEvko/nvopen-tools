// Function: sub_183C420
// Address: 0x183c420
//
__int64 __fastcall sub_183C420(_QWORD *a1, __int64 a2, size_t a3)
{
  __int64 v3; // r13
  _QWORD *v5; // rbx
  _BYTE *v6; // rax
  _BYTE *v7; // rsi
  int v8; // r14d
  size_t v9; // r8
  signed __int64 v10; // r15
  __int64 v11; // rax
  void *v12; // r9
  void *v13; // rax
  _BYTE *v14; // rax
  int v15; // r14d
  size_t v16; // r8
  signed __int64 v17; // r15
  __int64 v18; // rax
  void *v19; // r9
  int v20; // r14d
  size_t v21; // r8
  size_t v22; // r15
  __int64 v23; // rax
  void *v24; // r9
  void *v25; // rdx
  int v27; // eax
  void *v28; // rdx
  int v29; // eax
  void *v30; // rdx
  char *v31; // rsi
  const void *v32; // rdi
  size_t v33; // rdx
  void *v34; // rdx
  int v35; // eax
  size_t n; // [rsp+8h] [rbp-38h]
  size_t na; // [rsp+8h] [rbp-38h]
  size_t nb; // [rsp+8h] [rbp-38h]
  size_t nc; // [rsp+8h] [rbp-38h]
  size_t nd; // [rsp+8h] [rbp-38h]
  size_t ne; // [rsp+8h] [rbp-38h]

  v3 = a3;
  v5 = a1;
  v6 = (_BYTE *)a1[3];
  v7 = (_BYTE *)a1[2];
  v8 = *((_DWORD *)a1 + 2);
  v9 = v6 - v7;
  v10 = v6 - v7;
  if ( v6 == v7 )
  {
    v12 = 0;
  }
  else
  {
    if ( v9 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_66;
    a1 = (_QWORD *)(a1[3] - (_QWORD)v7);
    v11 = sub_22077B0(v9);
    v7 = (_BYTE *)v5[2];
    v12 = (void *)v11;
    v6 = (_BYTE *)v5[3];
    v9 = v6 - v7;
  }
  if ( v7 != v6 )
  {
    n = v9;
    v13 = memmove(v12, v7, v9);
    v9 = n;
    v12 = v13;
    if ( *(_DWORD *)a2 == v8 )
    {
      a1 = *(_QWORD **)(a2 + 8);
      if ( *(_QWORD *)(a2 + 16) - (_QWORD)a1 == n )
      {
LABEL_29:
        if ( !v9 )
        {
          if ( !v12 )
            goto LABEL_32;
          goto LABEL_31;
        }
        nc = (size_t)v12;
        v27 = memcmp(a1, v12, v9);
        v12 = (void *)nc;
        if ( !v27 )
        {
LABEL_31:
          j_j___libc_free_0(v12, v10);
LABEL_32:
          v28 = *(void **)(v3 + 24);
          if ( *(_QWORD *)(v3 + 16) - (_QWORD)v28 > 0xAu )
          {
            qmemcpy(v28, "Undefined  ", 11);
            *(_QWORD *)(v3 + 24) += 11LL;
            return 0x656E696665646E55LL;
          }
          v31 = "Undefined  ";
          return sub_16E7EE0(v3, v31, 0xBu);
        }
        goto LABEL_6;
      }
    }
    goto LABEL_6;
  }
  if ( *(_DWORD *)a2 == v8 )
  {
    a1 = *(_QWORD **)(a2 + 8);
    if ( *(_QWORD *)(a2 + 16) - (_QWORD)a1 == v9 )
      goto LABEL_29;
  }
  if ( v12 )
  {
LABEL_6:
    a1 = v12;
    j_j___libc_free_0(v12, v10);
  }
  v14 = (_BYTE *)v5[7];
  v7 = (_BYTE *)v5[6];
  v15 = *((_DWORD *)v5 + 10);
  v16 = v14 - v7;
  v17 = v14 - v7;
  if ( v14 == v7 )
  {
    v19 = 0;
  }
  else
  {
    if ( v16 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_66;
    a1 = (_QWORD *)(v5[7] - (_QWORD)v7);
    v18 = sub_22077B0(v16);
    v7 = (_BYTE *)v5[6];
    v19 = (void *)v18;
    v14 = (_BYTE *)v5[7];
    v16 = v14 - v7;
  }
  if ( v7 != v14 )
  {
    na = v16;
    v19 = memmove(v19, v7, v16);
    if ( v15 == *(_DWORD *)a2 )
    {
      a1 = *(_QWORD **)(a2 + 8);
      a3 = *(_QWORD *)(a2 + 16) - (_QWORD)a1;
      if ( na == a3 )
      {
LABEL_39:
        if ( !a3 )
        {
          if ( !v19 )
            goto LABEL_42;
          goto LABEL_41;
        }
        nd = (size_t)v19;
        v29 = memcmp(a1, v19, a3);
        v19 = (void *)nd;
        if ( !v29 )
        {
LABEL_41:
          j_j___libc_free_0(v19, v17);
LABEL_42:
          v30 = *(void **)(v3 + 24);
          if ( *(_QWORD *)(v3 + 16) - (_QWORD)v30 > 0xAu )
          {
            qmemcpy(v30, "Overdefined", 11);
            *(_QWORD *)(v3 + 24) += 11LL;
            return 0x696665647265764FLL;
          }
          v31 = "Overdefined";
          return sub_16E7EE0(v3, v31, 0xBu);
        }
        goto LABEL_12;
      }
    }
    goto LABEL_12;
  }
  if ( *(_DWORD *)a2 == v15 )
  {
    a1 = *(_QWORD **)(a2 + 8);
    a3 = *(_QWORD *)(a2 + 16) - (_QWORD)a1;
    if ( v16 == a3 )
      goto LABEL_39;
  }
  if ( v19 )
  {
LABEL_12:
    a1 = v19;
    j_j___libc_free_0(v19, v17);
  }
  v7 = (_BYTE *)v5[10];
  v20 = *((_DWORD *)v5 + 18);
  v21 = v5[11] - (_QWORD)v7;
  v22 = v21;
  if ( !v21 )
  {
    v24 = 0;
    if ( v7 != (_BYTE *)v5[11] )
      goto LABEL_16;
LABEL_35:
    if ( v20 != *(_DWORD *)a2 || (v32 = *(const void **)(a2 + 8), v33 = *(_QWORD *)(a2 + 16) - (_QWORD)v32, v21 != v33) )
    {
      if ( !v24 )
        goto LABEL_18;
      goto LABEL_17;
    }
    goto LABEL_56;
  }
  if ( v21 > 0x7FFFFFFFFFFFFFF8LL )
LABEL_66:
    sub_4261EA(a1, v7, a3);
  v23 = sub_22077B0(v21);
  v7 = (_BYTE *)v5[10];
  v24 = (void *)v23;
  v21 = v5[11] - (_QWORD)v7;
  if ( v7 == (_BYTE *)v5[11] )
    goto LABEL_35;
LABEL_16:
  nb = v21;
  v24 = memmove(v24, v7, v21);
  if ( v20 != *(_DWORD *)a2 || (v32 = *(const void **)(a2 + 8), v33 = *(_QWORD *)(a2 + 16) - (_QWORD)v32, nb != v33) )
  {
LABEL_17:
    j_j___libc_free_0(v24, v22);
LABEL_18:
    v25 = *(void **)(v3 + 24);
    if ( *(_QWORD *)(v3 + 16) - (_QWORD)v25 > 0xAu )
    {
      qmemcpy(v25, "FunctionSet", 11);
      *(_QWORD *)(v3 + 24) += 11LL;
      return 25939;
    }
    v31 = "FunctionSet";
    return sub_16E7EE0(v3, v31, 0xBu);
  }
LABEL_56:
  if ( !v33 )
  {
    if ( !v24 )
      goto LABEL_58;
    goto LABEL_65;
  }
  ne = (size_t)v24;
  v35 = memcmp(v32, v24, v33);
  v24 = (void *)ne;
  if ( v35 )
    goto LABEL_17;
LABEL_65:
  j_j___libc_free_0(v24, v22);
LABEL_58:
  v34 = *(void **)(v3 + 24);
  if ( *(_QWORD *)(v3 + 16) - (_QWORD)v34 <= 0xAu )
  {
    v31 = "Untracked  ";
    return sub_16E7EE0(v3, v31, 0xBu);
  }
  qmemcpy(v34, "Untracked  ", 11);
  *(_QWORD *)(v3 + 24) += 11LL;
  return 0x656B636172746E55LL;
}
