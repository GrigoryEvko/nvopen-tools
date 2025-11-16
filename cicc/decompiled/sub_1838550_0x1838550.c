// Function: sub_1838550
// Address: 0x1838550
//
_QWORD *__fastcall sub_1838550(_QWORD *a1, _QWORD *a2, char **a3)
{
  char *v5; // r15
  char *v6; // rdi
  char *v8; // rcx
  char *v9; // rbx
  char *v10; // r8
  char *v11; // rdx
  char *v12; // rsi
  char *v13; // rax
  _QWORD *result; // rax
  __int64 v15; // rax
  char *v16; // rcx
  __int64 v17; // rdx
  char *v18; // rax
  char *v19; // rax
  __int64 v20; // rax
  char *v21; // rsi
  __int64 v22; // rdx
  char *v23; // rax
  __int64 v24; // r12
  char *v25; // rdi
  char *v26; // rdx
  char *v27; // rcx
  char *v28; // rax
  signed __int64 v29; // [rsp+8h] [rbp-38h]
  signed __int64 v30; // [rsp+8h] [rbp-38h]

  if ( a2 == a1 + 1 )
  {
    if ( !a1[5] )
      return sub_1834350((__int64)a1, (__int64)a3);
    v24 = a1[4];
    v25 = a3[1];
    v26 = *a3;
    v27 = *(char **)(v24 + 40);
    v28 = *(char **)(v24 + 32);
    if ( v27 - v28 > v25 - v26 )
      v27 = &v28[v25 - v26];
    if ( v28 == v27 )
    {
LABEL_53:
      if ( v25 == v26 )
        return sub_1834350((__int64)a1, (__int64)a3);
    }
    else
    {
      while ( *(_QWORD *)v28 >= *(_QWORD *)v26 )
      {
        if ( *(_QWORD *)v28 > *(_QWORD *)v26 )
          return sub_1834350((__int64)a1, (__int64)a3);
        v28 += 8;
        v26 += 8;
        if ( v27 == v28 )
          goto LABEL_53;
      }
    }
    return 0;
  }
  v5 = a3[1];
  v6 = (char *)a2[5];
  v8 = (char *)a2[4];
  v9 = *a3;
  v10 = (char *)(v5 - *a3);
  v11 = v8;
  v12 = &v9[v6 - v8];
  if ( (__int64)v10 <= v6 - v8 )
    v12 = v5;
  if ( v9 != v12 )
  {
    v13 = v9;
    while ( *(_QWORD *)v13 >= *(_QWORD *)v11 )
    {
      if ( *(_QWORD *)v13 > *(_QWORD *)v11 )
        goto LABEL_23;
      v13 += 8;
      v11 += 8;
      if ( v12 == v13 )
        goto LABEL_22;
    }
    goto LABEL_9;
  }
LABEL_22:
  if ( v6 != v11 )
  {
LABEL_9:
    result = a2;
    if ( (_QWORD *)a1[3] == a2 )
      return result;
    v29 = (signed __int64)v10;
    v15 = sub_220EF80(a2);
    v16 = *(char **)(v15 + 40);
    v17 = v15;
    v18 = *(char **)(v15 + 32);
    if ( v29 < v16 - v18 )
      v16 = &v18[v29];
    if ( v18 != v16 )
    {
      while ( *(_QWORD *)v18 >= *(_QWORD *)v9 )
      {
        if ( *(_QWORD *)v18 > *(_QWORD *)v9 )
          return sub_1834350((__int64)a1, (__int64)a3);
        v18 += 8;
        v9 += 8;
        if ( v16 == v18 )
          goto LABEL_51;
      }
      goto LABEL_17;
    }
LABEL_51:
    if ( v5 != v9 )
    {
LABEL_17:
      result = 0;
      if ( *(_QWORD *)(v17 + 24) )
        return a2;
      return result;
    }
    return sub_1834350((__int64)a1, (__int64)a3);
  }
LABEL_23:
  if ( (__int64)v10 < v6 - v8 )
    v6 = &v10[(_QWORD)v8];
  v19 = v9;
  if ( v8 == v6 )
  {
LABEL_49:
    if ( v5 == v19 )
      return a2;
  }
  else
  {
    while ( *(_QWORD *)v8 >= *(_QWORD *)v19 )
    {
      if ( *(_QWORD *)v8 > *(_QWORD *)v19 )
        return a2;
      v8 += 8;
      v19 += 8;
      if ( v6 == v8 )
        goto LABEL_49;
    }
  }
  if ( (_QWORD *)a1[4] == a2 )
    return 0;
  v30 = (signed __int64)v10;
  v20 = sub_220EEE0(a2);
  v21 = *(char **)(v20 + 40);
  v22 = v20;
  v23 = *(char **)(v20 + 32);
  if ( v30 > v21 - v23 )
    v5 = &v9[v21 - v23];
  if ( v9 == v5 )
  {
LABEL_55:
    if ( v23 == v21 )
      return sub_1834350((__int64)a1, (__int64)a3);
  }
  else
  {
    while ( *(_QWORD *)v9 >= *(_QWORD *)v23 )
    {
      if ( *(_QWORD *)v9 > *(_QWORD *)v23 )
        return sub_1834350((__int64)a1, (__int64)a3);
      v9 += 8;
      v23 += 8;
      if ( v5 == v9 )
        goto LABEL_55;
    }
  }
  result = 0;
  if ( a2[3] )
    return (_QWORD *)v22;
  return result;
}
