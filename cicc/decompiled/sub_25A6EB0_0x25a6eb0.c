// Function: sub_25A6EB0
// Address: 0x25a6eb0
//
__int64 __fastcall sub_25A6EB0(_QWORD *a1, __int64 a2, size_t a3)
{
  __int64 v3; // r13
  _QWORD *v5; // rbx
  _BYTE *v6; // rax
  _BYTE *v7; // rsi
  int v8; // r14d
  unsigned __int64 v9; // r8
  __int64 v10; // rax
  void *v11; // r9
  void *v12; // rax
  _BYTE *v13; // rax
  int v14; // r14d
  unsigned __int64 v15; // r8
  __int64 v16; // rax
  void *v17; // r9
  int v18; // r14d
  unsigned __int64 v19; // r8
  __int64 v20; // rax
  void *v21; // r9
  void *v22; // rdx
  int v24; // eax
  void *v25; // rdx
  int v26; // eax
  void *v27; // rdx
  char *v28; // rsi
  const void *v29; // rdi
  size_t v30; // rdx
  void *v31; // rdx
  int v32; // eax
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
  if ( v6 == v7 )
  {
    v11 = 0;
  }
  else
  {
    if ( v9 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_66;
    a1 = (_QWORD *)(a1[3] - (_QWORD)v7);
    v10 = sub_22077B0(v9);
    v7 = (_BYTE *)v5[2];
    v11 = (void *)v10;
    v6 = (_BYTE *)v5[3];
    v9 = v6 - v7;
  }
  if ( v7 != v6 )
  {
    n = v9;
    v12 = memmove(v11, v7, v9);
    v9 = n;
    v11 = v12;
    if ( *(_DWORD *)a2 == v8 )
    {
      a1 = *(_QWORD **)(a2 + 8);
      if ( *(_QWORD *)(a2 + 16) - (_QWORD)a1 == n )
      {
LABEL_29:
        if ( !v9 )
        {
          if ( !v11 )
            goto LABEL_32;
          goto LABEL_31;
        }
        nc = (size_t)v11;
        v24 = memcmp(a1, v11, v9);
        v11 = (void *)nc;
        if ( !v24 )
        {
LABEL_31:
          j_j___libc_free_0((unsigned __int64)v11);
LABEL_32:
          v25 = *(void **)(v3 + 32);
          if ( *(_QWORD *)(v3 + 24) - (_QWORD)v25 > 0xAu )
          {
            qmemcpy(v25, "Undefined  ", 11);
            *(_QWORD *)(v3 + 32) += 11LL;
            return 0x656E696665646E55LL;
          }
          v28 = "Undefined  ";
          return sub_CB6200(v3, (unsigned __int8 *)v28, 0xBu);
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
  if ( v11 )
  {
LABEL_6:
    a1 = v11;
    j_j___libc_free_0((unsigned __int64)v11);
  }
  v13 = (_BYTE *)v5[7];
  v7 = (_BYTE *)v5[6];
  v14 = *((_DWORD *)v5 + 10);
  v15 = v13 - v7;
  if ( v13 == v7 )
  {
    v17 = 0;
  }
  else
  {
    if ( v15 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_66;
    a1 = (_QWORD *)(v5[7] - (_QWORD)v7);
    v16 = sub_22077B0(v15);
    v7 = (_BYTE *)v5[6];
    v17 = (void *)v16;
    v13 = (_BYTE *)v5[7];
    v15 = v13 - v7;
  }
  if ( v7 != v13 )
  {
    na = v15;
    v17 = memmove(v17, v7, v15);
    if ( v14 == *(_DWORD *)a2 )
    {
      a1 = *(_QWORD **)(a2 + 8);
      a3 = *(_QWORD *)(a2 + 16) - (_QWORD)a1;
      if ( na == a3 )
      {
LABEL_39:
        if ( !a3 )
        {
          if ( !v17 )
            goto LABEL_42;
          goto LABEL_41;
        }
        nd = (size_t)v17;
        v26 = memcmp(a1, v17, a3);
        v17 = (void *)nd;
        if ( !v26 )
        {
LABEL_41:
          j_j___libc_free_0((unsigned __int64)v17);
LABEL_42:
          v27 = *(void **)(v3 + 32);
          if ( *(_QWORD *)(v3 + 24) - (_QWORD)v27 > 0xAu )
          {
            qmemcpy(v27, "Overdefined", 11);
            *(_QWORD *)(v3 + 32) += 11LL;
            return 0x696665647265764FLL;
          }
          v28 = "Overdefined";
          return sub_CB6200(v3, (unsigned __int8 *)v28, 0xBu);
        }
        goto LABEL_12;
      }
    }
    goto LABEL_12;
  }
  if ( *(_DWORD *)a2 == v14 )
  {
    a1 = *(_QWORD **)(a2 + 8);
    a3 = *(_QWORD *)(a2 + 16) - (_QWORD)a1;
    if ( v15 == a3 )
      goto LABEL_39;
  }
  if ( v17 )
  {
LABEL_12:
    a1 = v17;
    j_j___libc_free_0((unsigned __int64)v17);
  }
  v7 = (_BYTE *)v5[10];
  v18 = *((_DWORD *)v5 + 18);
  v19 = v5[11] - (_QWORD)v7;
  if ( !v19 )
  {
    v21 = 0;
    if ( v7 != (_BYTE *)v5[11] )
      goto LABEL_16;
LABEL_35:
    if ( v18 != *(_DWORD *)a2 || (v29 = *(const void **)(a2 + 8), v30 = *(_QWORD *)(a2 + 16) - (_QWORD)v29, v19 != v30) )
    {
      if ( !v21 )
        goto LABEL_18;
      goto LABEL_17;
    }
    goto LABEL_56;
  }
  if ( v19 > 0x7FFFFFFFFFFFFFF8LL )
LABEL_66:
    sub_4261EA(a1, v7, a3);
  v20 = sub_22077B0(v19);
  v7 = (_BYTE *)v5[10];
  v21 = (void *)v20;
  v19 = v5[11] - (_QWORD)v7;
  if ( v7 == (_BYTE *)v5[11] )
    goto LABEL_35;
LABEL_16:
  nb = v19;
  v21 = memmove(v21, v7, v19);
  if ( v18 != *(_DWORD *)a2 || (v29 = *(const void **)(a2 + 8), v30 = *(_QWORD *)(a2 + 16) - (_QWORD)v29, nb != v30) )
  {
LABEL_17:
    j_j___libc_free_0((unsigned __int64)v21);
LABEL_18:
    v22 = *(void **)(v3 + 32);
    if ( *(_QWORD *)(v3 + 24) - (_QWORD)v22 > 0xAu )
    {
      qmemcpy(v22, "FunctionSet", 11);
      *(_QWORD *)(v3 + 32) += 11LL;
      return 25939;
    }
    v28 = "FunctionSet";
    return sub_CB6200(v3, (unsigned __int8 *)v28, 0xBu);
  }
LABEL_56:
  if ( !v30 )
  {
    if ( !v21 )
      goto LABEL_58;
    goto LABEL_65;
  }
  ne = (size_t)v21;
  v32 = memcmp(v29, v21, v30);
  v21 = (void *)ne;
  if ( v32 )
    goto LABEL_17;
LABEL_65:
  j_j___libc_free_0((unsigned __int64)v21);
LABEL_58:
  v31 = *(void **)(v3 + 32);
  if ( *(_QWORD *)(v3 + 24) - (_QWORD)v31 <= 0xAu )
  {
    v28 = "Untracked  ";
    return sub_CB6200(v3, (unsigned __int8 *)v28, 0xBu);
  }
  qmemcpy(v31, "Untracked  ", 11);
  *(_QWORD *)(v3 + 32) += 11LL;
  return 0x656B636172746E55LL;
}
