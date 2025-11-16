// Function: sub_E11840
// Address: 0xe11840
//
__int64 __fastcall sub_E11840(_QWORD *a1, char **a2)
{
  size_t v3; // r13
  size_t v5; // rsi
  unsigned __int64 v6; // rax
  size_t v7; // rdx
  char *v8; // rdi
  unsigned __int64 v9; // rsi
  unsigned __int64 v10; // rax
  __int64 v11; // rax
  _BYTE *v12; // r13
  __int64 result; // rax
  size_t v14; // rdx
  size_t v15; // rax
  const void *v16; // r14
  char *v17; // rdi
  unsigned __int64 v18; // rdx
  __int64 v19; // rax

  v3 = a1[2];
  v5 = (size_t)a2[1];
  if ( v3 )
  {
    v14 = (size_t)a2[2];
    v15 = v3 + v5;
    v16 = (const void *)a1[3];
    v17 = *a2;
    if ( v3 + v5 > v14 )
    {
      v18 = 2 * v14;
      if ( v15 + 992 > v18 )
        a2[2] = (char *)(v15 + 992);
      else
        a2[2] = (char *)v18;
      v19 = realloc(v17);
      *a2 = (char *)v19;
      v17 = (char *)v19;
      if ( !v19 )
        goto LABEL_18;
      v5 = (size_t)a2[1];
    }
    memcpy(&v17[v5], v16, v3);
    v5 = (size_t)&a2[1][v3];
    a2[1] = (char *)v5;
  }
  v6 = (unsigned __int64)a2[2];
  v7 = v5 + 1;
  v8 = *a2;
  if ( v5 + 1 <= v6 )
    goto LABEL_7;
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
LABEL_18:
    abort();
  v5 = (size_t)a2[1];
  v7 = v5 + 1;
LABEL_7:
  a2[1] = (char *)v7;
  v8[v5] = 32;
  v12 = (_BYTE *)a1[4];
  (*(void (__fastcall **)(_BYTE *, char **))(*(_QWORD *)v12 + 32LL))(v12, a2);
  result = v12[9] & 0xC0;
  if ( (v12[9] & 0xC0) != 0x40 )
    return (*(__int64 (__fastcall **)(_BYTE *, char **))(*(_QWORD *)v12 + 40LL))(v12, a2);
  return result;
}
