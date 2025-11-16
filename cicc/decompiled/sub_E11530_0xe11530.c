// Function: sub_E11530
// Address: 0xe11530
//
char *__fastcall sub_E11530(__int64 a1, __int64 *a2)
{
  char *v4; // rsi
  unsigned __int64 v5; // rax
  __int64 v6; // rdi
  unsigned __int64 v7; // rsi
  unsigned __int64 v8; // rax
  __int64 v9; // rax
  char *result; // rax
  char *v11; // rdi
  size_t v12; // r13
  unsigned __int64 v13; // rax
  const void *v14; // r12
  __int64 v15; // r8
  unsigned __int64 v16; // rsi
  unsigned __int64 v17; // rax
  __int64 v18; // rax

  v4 = (char *)a2[1];
  v5 = a2[2];
  v6 = *a2;
  if ( (unsigned __int64)(v4 + 2) > v5 )
  {
    v7 = (unsigned __int64)(v4 + 994);
    v8 = 2 * v5;
    if ( v7 <= v8 )
      a2[2] = v8;
    else
      a2[2] = v7;
    v9 = realloc((void *)v6);
    *a2 = v9;
    v6 = v9;
    if ( !v9 )
      goto LABEL_15;
    v4 = (char *)a2[1];
  }
  *(_WORD *)&v4[v6] = 28774;
  result = (char *)a2[1];
  v11 = result + 2;
  a2[1] = (__int64)(result + 2);
  v12 = *(_QWORD *)(a1 + 16);
  if ( !v12 )
    return result;
  v13 = a2[2];
  v14 = *(const void **)(a1 + 24);
  v15 = *a2;
  if ( (unsigned __int64)&v11[v12] > v13 )
  {
    v16 = (unsigned __int64)&v11[v12 + 992];
    v17 = 2 * v13;
    if ( v16 > v17 )
      a2[2] = v16;
    else
      a2[2] = v17;
    v18 = realloc((void *)v15);
    *a2 = v18;
    v15 = v18;
    if ( v18 )
    {
      v11 = (char *)a2[1];
      goto LABEL_12;
    }
LABEL_15:
    abort();
  }
LABEL_12:
  result = (char *)memcpy(&v11[v15], v14, v12);
  a2[1] += v12;
  return result;
}
