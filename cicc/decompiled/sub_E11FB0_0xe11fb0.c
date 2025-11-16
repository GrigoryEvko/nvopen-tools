// Function: sub_E11FB0
// Address: 0xe11fb0
//
unsigned __int64 __fastcall sub_E11FB0(_QWORD *a1, char **a2)
{
  size_t v3; // r13
  size_t v5; // rsi
  unsigned __int64 v6; // rax
  size_t v7; // rdx
  char *v8; // rdi
  unsigned __int64 v9; // rsi
  unsigned __int64 v10; // rax
  __int64 v11; // rax
  _BYTE *v12; // r12
  char *v13; // rsi
  unsigned __int64 result; // rax
  unsigned __int64 v15; // rdi
  char *v16; // rdx
  unsigned __int64 v17; // rsi
  unsigned __int64 v18; // rax
  size_t v19; // rdx
  size_t v20; // rax
  const void *v21; // r14
  char *v22; // rdi
  unsigned __int64 v23; // rdx
  __int64 v24; // rax

  v3 = a1[2];
  v5 = (size_t)a2[1];
  if ( v3 )
  {
    v19 = (size_t)a2[2];
    v20 = v3 + v5;
    v21 = (const void *)a1[3];
    v22 = *a2;
    if ( v3 + v5 > v19 )
    {
      v23 = 2 * v19;
      if ( v20 + 992 > v23 )
        a2[2] = (char *)(v20 + 992);
      else
        a2[2] = (char *)v23;
      v24 = realloc(v22);
      *a2 = (char *)v24;
      v22 = (char *)v24;
      if ( !v24 )
        goto LABEL_24;
      v5 = (size_t)a2[1];
    }
    memcpy(&v22[v5], v21, v3);
    v5 = (size_t)&a2[1][v3];
    a2[1] = (char *)v5;
  }
  v6 = (unsigned __int64)a2[2];
  v7 = v5 + 1;
  v8 = *a2;
  if ( v5 + 1 > v6 )
  {
    v9 = v5 + 993;
    v10 = 2 * v6;
    if ( v9 > v10 )
      a2[2] = (char *)v9;
    else
      a2[2] = (char *)v10;
    v11 = realloc(v8);
    *a2 = (char *)v11;
    v8 = (char *)v11;
    if ( !v11 )
      goto LABEL_24;
    v5 = (size_t)a2[1];
    v7 = v5 + 1;
  }
  a2[1] = (char *)v7;
  v8[v5] = 40;
  v12 = (_BYTE *)a1[4];
  (*(void (__fastcall **)(_BYTE *, char **))(*(_QWORD *)v12 + 32LL))(v12, a2);
  if ( (v12[9] & 0xC0) != 0x40 )
    (*(void (__fastcall **)(_BYTE *, char **))(*(_QWORD *)v12 + 40LL))(v12, a2);
  v13 = a2[1];
  result = (unsigned __int64)a2[2];
  v15 = (unsigned __int64)*a2;
  v16 = v13 + 1;
  if ( (unsigned __int64)(v13 + 1) > result )
  {
    v17 = (unsigned __int64)(v13 + 993);
    v18 = 2 * result;
    if ( v17 > v18 )
      a2[2] = (char *)v17;
    else
      a2[2] = (char *)v18;
    result = realloc((void *)v15);
    *a2 = (char *)result;
    v15 = result;
    if ( result )
    {
      v13 = a2[1];
      v16 = v13 + 1;
      goto LABEL_14;
    }
LABEL_24:
    abort();
  }
LABEL_14:
  a2[1] = v16;
  v13[v15] = 41;
  return result;
}
