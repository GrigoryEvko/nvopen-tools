// Function: sub_E12FC0
// Address: 0xe12fc0
//
unsigned __int64 __fastcall sub_E12FC0(__int64 a1, char **a2)
{
  char *v4; // rsi
  unsigned __int64 v5; // rax
  char *v6; // rdx
  unsigned __int64 v7; // rsi
  unsigned __int64 v8; // rax
  __int64 v9; // rax
  _BYTE *v10; // r13
  char *v11; // rsi
  unsigned __int64 result; // rax
  unsigned __int64 v13; // rdi
  char *v14; // rdx
  unsigned __int64 v15; // rsi
  unsigned __int64 v16; // rax
  size_t v17; // r13
  char *v18; // rax
  size_t v19; // rdx
  const void *v20; // r12
  char *v21; // rdi
  unsigned __int64 v22; // rsi
  unsigned __int64 v23; // rdx
  __int64 v24; // rax

  sub_E12F20((__int64 *)a2, *(_QWORD *)(a1 + 16), *(const void **)(a1 + 24));
  v4 = a2[1];
  v5 = (unsigned __int64)a2[2];
  ++*((_DWORD *)a2 + 8);
  v6 = v4 + 1;
  if ( (unsigned __int64)(v4 + 1) <= v5 )
  {
    v9 = (__int64)*a2;
  }
  else
  {
    v7 = (unsigned __int64)(v4 + 993);
    v8 = 2 * v5;
    if ( v7 > v8 )
      a2[2] = (char *)v7;
    else
      a2[2] = (char *)v8;
    v9 = realloc(*a2);
    *a2 = (char *)v9;
    if ( !v9 )
      goto LABEL_25;
    v4 = a2[1];
    v6 = v4 + 1;
  }
  a2[1] = v6;
  v4[v9] = 40;
  v10 = *(_BYTE **)(a1 + 32);
  (*(void (__fastcall **)(_BYTE *, char **))(*(_QWORD *)v10 + 32LL))(v10, a2);
  if ( (v10[9] & 0xC0) != 0x40 )
    (*(void (__fastcall **)(_BYTE *, char **))(*(_QWORD *)v10 + 40LL))(v10, a2);
  v11 = a2[1];
  result = (unsigned __int64)a2[2];
  --*((_DWORD *)a2 + 8);
  v13 = (unsigned __int64)*a2;
  v14 = v11 + 1;
  if ( (unsigned __int64)(v11 + 1) > result )
  {
    v15 = (unsigned __int64)(v11 + 993);
    v16 = 2 * result;
    if ( v15 > v16 )
      a2[2] = (char *)v15;
    else
      a2[2] = (char *)v16;
    result = realloc((void *)v13);
    *a2 = (char *)result;
    v13 = result;
    if ( !result )
      goto LABEL_25;
    v11 = a2[1];
    v14 = v11 + 1;
  }
  a2[1] = v14;
  v11[v13] = 41;
  v17 = *(_QWORD *)(a1 + 40);
  if ( v17 )
  {
    v18 = a2[1];
    v19 = (size_t)a2[2];
    v20 = *(const void **)(a1 + 48);
    v21 = *a2;
    if ( (unsigned __int64)&v18[v17] <= v19 )
    {
LABEL_23:
      result = (unsigned __int64)memcpy(&v21[(_QWORD)v18], v20, v17);
      a2[1] += v17;
      return result;
    }
    v22 = (unsigned __int64)&v18[v17 + 992];
    v23 = 2 * v19;
    if ( v22 > v23 )
      a2[2] = (char *)v22;
    else
      a2[2] = (char *)v23;
    v24 = realloc(v21);
    *a2 = (char *)v24;
    v21 = (char *)v24;
    if ( v24 )
    {
      v18 = a2[1];
      goto LABEL_23;
    }
LABEL_25:
    abort();
  }
  return result;
}
