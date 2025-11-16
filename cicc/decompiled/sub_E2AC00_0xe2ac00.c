// Function: sub_E2AC00
// Address: 0xe2ac00
//
char *__fastcall sub_E2AC00(__int64 a1, __int64 *a2)
{
  int v3; // edx
  char *v5; // r8
  __int64 v6; // rdi
  unsigned __int64 v7; // rax
  char *v8; // rcx
  unsigned __int64 v9; // rax
  __int64 v10; // rax
  unsigned __int64 v11; // rax
  __int64 v12; // rax
  size_t v13; // r13
  _BYTE *v14; // rdx
  unsigned __int64 v15; // rax
  __int64 v16; // rax
  char *v17; // rsi
  char *result; // rax
  unsigned __int64 v19; // rdx
  __int64 v20; // rdi
  unsigned __int64 v21; // rsi
  unsigned __int64 v22; // rdx
  __int64 v23; // rax
  size_t v24; // rdx
  const void *v25; // r14
  char *v26; // rsi
  unsigned __int64 v27; // rax
  __int64 v28; // rax
  unsigned __int64 v29; // rax
  __int64 v30; // rax
  unsigned __int64 v31; // rax
  __int64 v32; // rax

  v3 = *(_DWORD *)(a1 + 44);
  v5 = (char *)*a2;
  v6 = a2[1];
  v7 = a2[2];
  v8 = (char *)*a2;
  if ( v3 == 2 )
  {
    if ( v6 + 2 > v7 )
    {
      v31 = 2 * v7;
      if ( v6 + 994 > v31 )
        a2[2] = v6 + 994;
      else
        a2[2] = v31;
      v32 = realloc(v5);
      *a2 = v32;
      v8 = (char *)v32;
      if ( !v32 )
        goto LABEL_57;
      v6 = a2[1];
    }
    *(_WORD *)&v8[v6] = 8789;
LABEL_18:
    v5 = (char *)*a2;
    v6 = a2[1] + 2;
    v7 = a2[2];
    a2[1] = v6;
    goto LABEL_19;
  }
  if ( v3 > 2 )
  {
    if ( v3 != 3 )
      goto LABEL_19;
    if ( v6 + 2 > v7 )
    {
      v11 = 2 * v7;
      if ( v6 + 994 > v11 )
        a2[2] = v6 + 994;
      else
        a2[2] = v11;
      v12 = realloc(v5);
      *a2 = v12;
      v8 = (char *)v12;
      if ( !v12 )
        goto LABEL_57;
      v6 = a2[1];
    }
    *(_WORD *)&v8[v6] = 8780;
    goto LABEL_18;
  }
  if ( v3 )
  {
    if ( v3 != 1 )
      goto LABEL_19;
    if ( v6 + 2 > v7 )
    {
      v9 = 2 * v7;
      if ( v6 + 994 > v9 )
        a2[2] = v6 + 994;
      else
        a2[2] = v9;
      v10 = realloc(v5);
      *a2 = v10;
      v8 = (char *)v10;
      if ( !v10 )
        goto LABEL_57;
      v6 = a2[1];
    }
    *(_WORD *)&v8[v6] = 8821;
    goto LABEL_18;
  }
  if ( v6 + 1 > v7 )
  {
    v29 = 2 * v7;
    if ( v6 + 993 > v29 )
      a2[2] = v6 + 993;
    else
      a2[2] = v29;
    v30 = realloc(v5);
    *a2 = v30;
    v8 = (char *)v30;
    if ( !v30 )
      goto LABEL_57;
    v6 = a2[1];
  }
  v8[v6] = 34;
  v5 = (char *)*a2;
  v6 = a2[1] + 1;
  v7 = a2[2];
  a2[1] = v6;
LABEL_19:
  v13 = *(_QWORD *)(a1 + 24);
  if ( v13 )
  {
    v24 = v13 + v6;
    v25 = *(const void **)(a1 + 32);
    v26 = v5;
    if ( v7 < v13 + v6 )
    {
      v27 = 2 * v7;
      if ( v24 + 992 > v27 )
        a2[2] = v24 + 992;
      else
        a2[2] = v27;
      v28 = realloc(v5);
      *a2 = v28;
      v26 = (char *)v28;
      if ( !v28 )
        goto LABEL_57;
      v6 = a2[1];
    }
    memcpy(&v26[v6], v25, v13);
    v7 = a2[2];
    v5 = (char *)*a2;
    v6 = v13 + a2[1];
    a2[1] = v6;
  }
  v14 = v5;
  if ( v6 + 1 > v7 )
  {
    v15 = 2 * v7;
    if ( v6 + 993 <= v15 )
      a2[2] = v15;
    else
      a2[2] = v6 + 993;
    v16 = realloc(v5);
    *a2 = v16;
    v14 = (_BYTE *)v16;
    if ( !v16 )
      goto LABEL_57;
    v6 = a2[1];
  }
  v14[v6] = 34;
  v17 = (char *)a2[1];
  result = v17 + 1;
  a2[1] = (__int64)(v17 + 1);
  if ( !*(_BYTE *)(a1 + 40) )
    return result;
  v19 = a2[2];
  v20 = *a2;
  if ( (unsigned __int64)(v17 + 4) > v19 )
  {
    v21 = (unsigned __int64)(v17 + 996);
    v22 = 2 * v19;
    if ( v21 > v22 )
      a2[2] = v21;
    else
      a2[2] = v22;
    v23 = realloc((void *)v20);
    *a2 = v23;
    v20 = v23;
    if ( v23 )
    {
      result = (char *)a2[1];
      goto LABEL_31;
    }
LABEL_57:
    abort();
  }
LABEL_31:
  result += v20;
  *(_WORD *)result = 11822;
  result[2] = 46;
  a2[1] += 3;
  return result;
}
