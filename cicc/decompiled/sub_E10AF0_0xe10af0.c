// Function: sub_E10AF0
// Address: 0xe10af0
//
__int64 __fastcall sub_E10AF0(_QWORD *a1, char **a2)
{
  size_t v3; // r13
  _BYTE *v5; // r13
  __int64 result; // rax
  char *v7; // rax
  size_t v8; // rdx
  const void *v9; // r14
  char *v10; // rdi
  unsigned __int64 v11; // rsi
  unsigned __int64 v12; // rdx
  __int64 v13; // rax

  v3 = a1[2];
  if ( v3 )
  {
    v7 = a2[1];
    v8 = (size_t)a2[2];
    v9 = (const void *)a1[3];
    v10 = *a2;
    if ( (unsigned __int64)&v7[v3] > v8 )
    {
      v11 = (unsigned __int64)&v7[v3 + 992];
      v12 = 2 * v8;
      if ( v11 > v12 )
        a2[2] = (char *)v11;
      else
        a2[2] = (char *)v12;
      v13 = realloc(v10);
      *a2 = (char *)v13;
      v10 = (char *)v13;
      if ( !v13 )
        abort();
      v7 = a2[1];
    }
    memcpy(&v10[(_QWORD)v7], v9, v3);
    a2[1] += v3;
  }
  v5 = (_BYTE *)a1[4];
  (*(void (__fastcall **)(_BYTE *, char **))(*(_QWORD *)v5 + 32LL))(v5, a2);
  result = v5[9] & 0xC0;
  if ( (v5[9] & 0xC0) != 0x40 )
    return (*(__int64 (__fastcall **)(_BYTE *, char **))(*(_QWORD *)v5 + 40LL))(v5, a2);
  return result;
}
