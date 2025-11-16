// Function: sub_E17200
// Address: 0xe17200
//
__int64 __fastcall sub_E17200(__int64 a1, __int64 *a2)
{
  char *v4; // rsi
  unsigned __int64 v5; // rax
  char *v6; // rdi
  unsigned __int64 v7; // rsi
  unsigned __int64 v8; // rax
  __int64 v9; // rax
  char *v10; // rdi
  char *v11; // rsi
  unsigned __int64 v12; // rdx
  char *v13; // rax
  char *v14; // rcx
  unsigned __int64 v15; // rsi
  unsigned __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rdx
  char *v19; // rsi
  unsigned __int64 v20; // rax
  unsigned __int64 v21; // rsi
  unsigned __int64 v22; // rax
  __int64 result; // rax
  char *v24; // rsi

  v4 = (char *)a2[1];
  v5 = a2[2];
  v6 = (char *)*a2;
  if ( (unsigned __int64)(v4 + 5) > v5 )
  {
    v7 = (unsigned __int64)(v4 + 997);
    v8 = 2 * v5;
    if ( v7 > v8 )
      a2[2] = v7;
    else
      a2[2] = v8;
    v9 = realloc(v6);
    *a2 = v9;
    v6 = (char *)v9;
    if ( !v9 )
      goto LABEL_21;
    v4 = (char *)a2[1];
  }
  v10 = &v6[(_QWORD)v4];
  *(_DWORD *)v10 = 1869768820;
  v10[4] = 119;
  v11 = (char *)a2[1];
  v12 = a2[2];
  ++*((_DWORD *)a2 + 8);
  v13 = v11 + 5;
  v14 = v11 + 6;
  a2[1] = (__int64)(v11 + 5);
  if ( (unsigned __int64)(v11 + 6) <= v12 )
  {
    v18 = *a2;
  }
  else
  {
    v15 = (unsigned __int64)(v11 + 998);
    v16 = 2 * v12;
    if ( v15 > v16 )
      a2[2] = v15;
    else
      a2[2] = v16;
    v17 = realloc((void *)*a2);
    *a2 = v17;
    v18 = v17;
    if ( !v17 )
      goto LABEL_21;
    v13 = (char *)a2[1];
    v14 = v13 + 1;
  }
  a2[1] = (__int64)v14;
  v13[v18] = 40;
  sub_E161C0((_QWORD *)(a1 + 16), (char **)a2);
  v19 = (char *)a2[1];
  v20 = a2[2];
  --*((_DWORD *)a2 + 8);
  if ( (unsigned __int64)(v19 + 1) > v20 )
  {
    v21 = (unsigned __int64)(v19 + 993);
    v22 = 2 * v20;
    if ( v21 > v22 )
      a2[2] = v21;
    else
      a2[2] = v22;
    result = realloc((void *)*a2);
    *a2 = result;
    if ( result )
    {
      v24 = (char *)a2[1];
      a2[1] = (__int64)(v24 + 1);
      v24[result] = 41;
      return result;
    }
LABEL_21:
    abort();
  }
  result = *a2;
  a2[1] = (__int64)(v19 + 1);
  v19[result] = 41;
  return result;
}
