// Function: sub_E55880
// Address: 0xe55880
//
int __fastcall sub_E55880(__int64 *a1, const char **a2)
{
  _BYTE *v3; // rax
  __int64 v4; // r14
  const char *v5; // rbx
  size_t v6; // r12
  const char *v7; // r13
  _QWORD *v8; // r12
  unsigned __int64 v9; // rdx
  char *v10; // rsi
  _BYTE *v11; // rsi
  _BYTE *v12; // rdx
  size_t v13; // rdx
  unsigned __int64 v14; // r12
  unsigned __int64 v15; // r14
  unsigned __int64 i; // rax
  __int64 v17; // r8
  __int64 v18; // r9
  unsigned __int64 v19; // r9
  size_t v20; // r14
  _QWORD *v21; // r10
  unsigned __int64 v22; // rdx
  _BYTE *v23; // rdi
  _QWORD *v24; // rdi
  unsigned __int64 v25; // rbx
  __int64 v26; // rax
  __int64 v27; // rdi
  __int64 v28; // rax
  size_t v29; // r10
  const void *v30; // r11
  unsigned __int64 v31; // rcx
  unsigned __int64 v32; // rdi
  __int64 v33; // rax
  size_t v34; // rdx
  unsigned __int64 v35; // rax
  char *v36; // rsi
  const void *v38; // [rsp+8h] [rbp-88h]
  size_t v39; // [rsp+10h] [rbp-80h]
  size_t v40; // [rsp+10h] [rbp-80h]
  _QWORD *v41; // [rsp+10h] [rbp-80h]
  const void *v42; // [rsp+20h] [rbp-70h]
  __int64 v43; // [rsp+28h] [rbp-68h]
  char *v44; // [rsp+30h] [rbp-60h] BYREF
  unsigned __int64 v45; // [rsp+38h] [rbp-58h]
  void *src; // [rsp+40h] [rbp-50h] BYREF
  size_t n; // [rsp+48h] [rbp-48h]
  _QWORD v48[8]; // [rsp+50h] [rbp-40h] BYREF

  LODWORD(v3) = *((unsigned __int8 *)a2 + 32);
  if ( (_BYTE)v3 == 4 )
  {
    v5 = *(const char **)*a2;
    v6 = *((_QWORD *)*a2 + 1);
    goto LABEL_19;
  }
  if ( (unsigned __int8)v3 > 4u )
  {
    if ( (unsigned __int8)((_BYTE)v3 - 5) <= 1u )
    {
      v6 = (size_t)a2[1];
      v5 = *a2;
      goto LABEL_19;
    }
LABEL_78:
    BUG();
  }
  if ( (_BYTE)v3 == 1 )
  {
    v4 = a1[39];
    v44 = 0;
    v5 = 0;
    v6 = 0;
    v45 = 0;
    v7 = *(const char **)(v4 + 40);
    if ( !v7 )
      return (int)v3;
LABEL_5:
    v3 = (_BYTE *)strlen(v7);
    if ( v3 != (_BYTE *)v6 )
      goto LABEL_6;
    goto LABEL_21;
  }
  if ( (_BYTE)v3 != 3 )
    goto LABEL_78;
  v5 = *a2;
  v6 = 0;
  if ( *a2 )
    v6 = strlen(*a2);
LABEL_19:
  v4 = a1[39];
  v44 = (char *)v5;
  LODWORD(v3) = 0;
  v45 = v6;
  v7 = *(const char **)(v4 + 40);
  if ( v7 )
    goto LABEL_5;
  if ( v6 )
  {
LABEL_6:
    if ( v6 <= 1 )
      goto LABEL_28;
    goto LABEL_7;
  }
LABEL_21:
  if ( !v6 )
    return (int)v3;
  LODWORD(v3) = memcmp(v5, v7, v6);
  if ( !(_DWORD)v3 )
    return (int)v3;
  if ( v6 <= 1 )
    goto LABEL_28;
LABEL_7:
  if ( *(_WORD *)v5 == 12079 )
  {
    v8 = a1 + 42;
    sub_C58CA0(a1 + 42, "\t", "");
    sub_C58CA0(a1 + 42, *(_BYTE **)(a1[39] + 48), (_BYTE *)(*(_QWORD *)(a1[39] + 48) + *(_QWORD *)(a1[39] + 56)));
    if ( v45 > 1 )
    {
      v9 = v45 - 2;
      v10 = v44 + 2;
LABEL_10:
      src = v48;
      sub_E4CC80((__int64 *)&src, v10, (__int64)&v10[v9]);
      v11 = src;
      v12 = (char *)src + n;
LABEL_11:
      LODWORD(v3) = (unsigned int)sub_C58CA0(v8, v11, v12);
      if ( src != v48 )
        LODWORD(v3) = j_j___libc_free_0(src, v48[0] + 1LL);
LABEL_13:
      v5 = v44;
      v6 = v45;
      goto LABEL_14;
    }
    v10 = &v44[v45];
    if ( &v44[v45] )
    {
      v9 = 0;
      goto LABEL_10;
    }
LABEL_74:
    v12 = v48;
    n = 0;
    src = v48;
    v11 = v48;
    LOBYTE(v48[0]) = 0;
    goto LABEL_11;
  }
  if ( *(_WORD *)v5 == 10799 )
  {
    v14 = v6 - 2;
    v15 = 2;
    v43 = (__int64)(a1 + 42);
    v42 = a1 + 45;
    for ( i = sub_C934D0(&v44, (unsigned __int8 *)"\r\n", 2, 2u); ; i = sub_C934D0(
                                                                          &v44,
                                                                          (unsigned __int8 *)"\r\n",
                                                                          2,
                                                                          v15) )
    {
      if ( v14 <= i )
        i = v14;
      v25 = i;
      v26 = a1[43];
      if ( v26 + 1 > (unsigned __int64)a1[44] )
      {
        sub_C8D290(v43, v42, v26 + 1, 1u, v17, v18);
        v26 = a1[43];
      }
      *(_BYTE *)(a1[42] + v26) = 9;
      v27 = a1[43] + 1;
      v28 = a1[39];
      a1[43] = v27;
      v29 = *(_QWORD *)(v28 + 56);
      v30 = *(const void **)(v28 + 48);
      if ( v29 + v27 > a1[44] )
      {
        v38 = *(const void **)(v28 + 48);
        v40 = *(_QWORD *)(v28 + 56);
        sub_C8D290(v43, v42, v29 + v27, 1u, v17, v18);
        v27 = a1[43];
        v30 = v38;
        v29 = v40;
      }
      if ( v29 )
      {
        v39 = v29;
        memcpy((void *)(a1[42] + v27), v30, v29);
        v27 = a1[43];
        v29 = v39;
      }
      v31 = v45;
      v19 = v15;
      v22 = v29 + v27;
      a1[43] = v29 + v27;
      if ( v15 > v31 )
        v19 = v31;
      v32 = v19;
      if ( v25 >= v19 )
      {
        if ( v31 > v25 )
          v31 = v25;
        v32 = v31;
      }
      LODWORD(v3) = (_DWORD)v44;
      if ( &v44[v19] )
      {
        src = v48;
        sub_E4CC80((__int64 *)&src, &v44[v19], (__int64)&v44[v32]);
        v20 = n;
        v3 = (_BYTE *)a1[43];
        v21 = src;
        v22 = (unsigned __int64)&v3[n];
        if ( (unsigned __int64)&v3[n] <= a1[44] )
        {
          v23 = &v3[a1[42]];
          if ( n )
            goto LABEL_36;
          goto LABEL_64;
        }
      }
      else
      {
        src = v48;
        n = 0;
        LOBYTE(v48[0]) = 0;
        if ( v22 <= a1[44] )
        {
          a1[43] = v22;
          if ( v14 <= v25 )
            goto LABEL_40;
LABEL_59:
          v33 = a1[43];
          if ( v33 + 1 > (unsigned __int64)a1[44] )
          {
            sub_C8D290(v43, v42, v33 + 1, 1u, v17, v19);
            v33 = a1[43];
          }
          v3 = (_BYTE *)(a1[42] + v33);
          *v3 = 10;
          ++a1[43];
          goto LABEL_40;
        }
        v20 = 0;
        v21 = v48;
      }
      v41 = v21;
      sub_C8D290(v43, v42, v22, 1u, v17, v19);
      v3 = (_BYTE *)a1[43];
      v21 = v41;
      v23 = &v3[a1[42]];
      if ( v20 )
      {
LABEL_36:
        memcpy(v23, v21, v20);
        v24 = src;
        v3 = (_BYTE *)(v20 + a1[43]);
        goto LABEL_37;
      }
LABEL_64:
      v24 = src;
LABEL_37:
      a1[43] = (__int64)v3;
      if ( v24 != v48 )
        LODWORD(v3) = j_j___libc_free_0(v24, v48[0] + 1LL);
      if ( v14 > v25 )
        goto LABEL_59;
LABEL_40:
      v15 = v25 + 1;
      if ( v14 <= v25 + 1 )
        goto LABEL_13;
    }
  }
LABEL_28:
  v13 = *(_QWORD *)(v4 + 56);
  if ( v13 <= v6 )
  {
    if ( !v13 || (LODWORD(v3) = memcmp(v5, *(const void **)(v4 + 48), v13), !(_DWORD)v3) )
    {
      v8 = a1 + 42;
      sub_C58CA0(a1 + 42, "\t", "");
      if ( v44 )
      {
        src = v48;
        sub_E4CC80((__int64 *)&src, v44, (__int64)&v44[v45]);
        v11 = src;
        v12 = (char *)src + n;
        goto LABEL_11;
      }
      goto LABEL_74;
    }
  }
  if ( *v5 == 35 )
  {
    v8 = a1 + 42;
    sub_C58CA0(a1 + 42, "\t", "");
    sub_C58CA0(a1 + 42, *(_BYTE **)(a1[39] + 48), (_BYTE *)(*(_QWORD *)(a1[39] + 48) + *(_QWORD *)(a1[39] + 56)));
    v35 = v45;
    if ( v45 )
    {
      v35 = v45 - 1;
      v36 = v44 + 1;
    }
    else
    {
      v36 = v44;
      if ( !v44 )
        goto LABEL_74;
    }
    src = v48;
    sub_E4CC80((__int64 *)&src, v36, (__int64)&v36[v35]);
    v11 = src;
    v12 = (char *)src + n;
    goto LABEL_11;
  }
LABEL_14:
  if ( v5[v6 - 1] == 10 )
  {
    v34 = a1[43];
    if ( v34 )
      LODWORD(v3) = sub_A51340(a1[38], (const void *)a1[42], v34);
    a1[43] = 0;
  }
  return (int)v3;
}
