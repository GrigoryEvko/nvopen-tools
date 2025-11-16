// Function: sub_E11D10
// Address: 0xe11d10
//
unsigned __int64 __fastcall sub_E11D10(__int64 a1, char **a2)
{
  char *v4; // rsi
  unsigned __int64 v5; // rax
  __int64 v6; // rdi
  unsigned __int64 v7; // rsi
  unsigned __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rdi
  size_t v11; // r13
  unsigned __int64 result; // rax
  char *v13; // r8
  unsigned __int64 v14; // rax
  size_t v15; // rax
  const void *v16; // r12
  char *v17; // r8
  unsigned __int64 v18; // rsi
  unsigned __int64 v19; // rax
  __int64 v20; // rax

  v4 = a2[1];
  v5 = (unsigned __int64)a2[2];
  v6 = (__int64)*a2;
  if ( (unsigned __int64)(v4 + 8) > v5 )
  {
    v7 = (unsigned __int64)(v4 + 1000);
    v8 = 2 * v5;
    if ( v7 <= v8 )
      a2[2] = (char *)v8;
    else
      a2[2] = (char *)v7;
    v9 = realloc((void *)v6);
    *a2 = (char *)v9;
    v6 = v9;
    if ( !v9 )
      goto LABEL_22;
    v4 = a2[1];
  }
  *(_QWORD *)&v4[v6] = 0x64656D616E6E7527LL;
  v10 = (__int64)(a2[1] + 8);
  a2[1] = (char *)v10;
  v11 = *(_QWORD *)(a1 + 16);
  if ( v11 )
  {
    v15 = (size_t)a2[2];
    v16 = *(const void **)(a1 + 24);
    v17 = *a2;
    if ( v11 + v10 > v15 )
    {
      v18 = v11 + v10 + 992;
      v19 = 2 * v15;
      if ( v18 > v19 )
        a2[2] = (char *)v18;
      else
        a2[2] = (char *)v19;
      v20 = realloc(v17);
      *a2 = (char *)v20;
      v17 = (char *)v20;
      if ( !v20 )
        goto LABEL_22;
      v10 = (__int64)a2[1];
    }
    memcpy(&v17[v10], v16, v11);
    v10 = (__int64)&a2[1][v11];
    a2[1] = (char *)v10;
  }
  result = (unsigned __int64)a2[2];
  v13 = *a2;
  if ( v10 + 1 > result )
  {
    v14 = 2 * result;
    if ( v10 + 993 <= v14 )
      a2[2] = (char *)v14;
    else
      a2[2] = (char *)(v10 + 993);
    result = realloc(v13);
    *a2 = (char *)result;
    v13 = (char *)result;
    if ( result )
    {
      v10 = (__int64)a2[1];
      goto LABEL_12;
    }
LABEL_22:
    abort();
  }
LABEL_12:
  v13[v10] = 39;
  ++a2[1];
  return result;
}
