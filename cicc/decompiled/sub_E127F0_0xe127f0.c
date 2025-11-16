// Function: sub_E127F0
// Address: 0xe127f0
//
unsigned __int64 __fastcall sub_E127F0(_QWORD *a1, char **a2)
{
  _BYTE *v4; // r13
  char *v5; // rsi
  unsigned __int64 v6; // rax
  __int64 v7; // rdi
  unsigned __int64 v8; // rsi
  unsigned __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rdi
  size_t v12; // r13
  unsigned __int64 result; // rax
  char *v14; // r8
  unsigned __int64 v15; // rax
  size_t v16; // rax
  const void *v17; // r12
  char *v18; // r8
  unsigned __int64 v19; // rsi
  unsigned __int64 v20; // rax
  __int64 v21; // rax

  v4 = (_BYTE *)a1[2];
  (*(void (__fastcall **)(_BYTE *))(*(_QWORD *)v4 + 32LL))(v4);
  if ( (v4[9] & 0xC0) != 0x40 )
    (*(void (__fastcall **)(_BYTE *, char **))(*(_QWORD *)v4 + 40LL))(v4, a2);
  v5 = a2[1];
  v6 = (unsigned __int64)a2[2];
  v7 = (__int64)*a2;
  if ( (unsigned __int64)(v5 + 1) > v6 )
  {
    v8 = (unsigned __int64)(v5 + 993);
    v9 = 2 * v6;
    if ( v8 <= v9 )
      a2[2] = (char *)v9;
    else
      a2[2] = (char *)v8;
    v10 = realloc((void *)v7);
    *a2 = (char *)v10;
    v7 = v10;
    if ( !v10 )
      goto LABEL_24;
    v5 = a2[1];
  }
  v5[v7] = 60;
  v11 = (__int64)(a2[1] + 1);
  a2[1] = (char *)v11;
  v12 = a1[3];
  if ( v12 )
  {
    v16 = (size_t)a2[2];
    v17 = (const void *)a1[4];
    v18 = *a2;
    if ( v12 + v11 > v16 )
    {
      v19 = v12 + v11 + 992;
      v20 = 2 * v16;
      if ( v19 > v20 )
        a2[2] = (char *)v19;
      else
        a2[2] = (char *)v20;
      v21 = realloc(v18);
      *a2 = (char *)v21;
      v18 = (char *)v21;
      if ( !v21 )
        goto LABEL_24;
      v11 = (__int64)a2[1];
    }
    memcpy(&v18[v11], v17, v12);
    v11 = (__int64)&a2[1][v12];
    a2[1] = (char *)v11;
  }
  result = (unsigned __int64)a2[2];
  v14 = *a2;
  if ( v11 + 1 > result )
  {
    v15 = 2 * result;
    if ( v11 + 993 <= v15 )
      a2[2] = (char *)v15;
    else
      a2[2] = (char *)(v11 + 993);
    result = realloc(v14);
    *a2 = (char *)result;
    v14 = (char *)result;
    if ( result )
    {
      v11 = (__int64)a2[1];
      goto LABEL_14;
    }
LABEL_24:
    abort();
  }
LABEL_14:
  v14[v11] = 62;
  ++a2[1];
  return result;
}
