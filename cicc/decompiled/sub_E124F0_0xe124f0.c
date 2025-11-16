// Function: sub_E124F0
// Address: 0xe124f0
//
__int64 __fastcall sub_E124F0(_QWORD *a1, char **a2)
{
  _BYTE *v3; // r13
  char *v5; // rsi
  unsigned __int64 v6; // rax
  __int64 v7; // rdi
  unsigned __int64 v8; // rsi
  unsigned __int64 v9; // rax
  __int64 v10; // rax
  __int64 result; // rax
  __int64 v12; // rdi
  size_t v13; // r13
  _BYTE *v14; // r13
  size_t v15; // rax
  const void *v16; // r14
  char *v17; // r8
  unsigned __int64 v18; // rsi
  unsigned __int64 v19; // rax
  __int64 v20; // rax

  v3 = (_BYTE *)a1[2];
  (*(void (__fastcall **)(_BYTE *))(*(_QWORD *)v3 + 32LL))(v3);
  if ( (v3[9] & 0xC0) != 0x40 )
    (*(void (__fastcall **)(_BYTE *, char **))(*(_QWORD *)v3 + 40LL))(v3, a2);
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
      goto LABEL_21;
    v5 = a2[1];
  }
  v5[v7] = 32;
  result = (__int64)a2[1];
  v12 = result + 1;
  a2[1] = (char *)(result + 1);
  v13 = a1[3];
  if ( !v13 )
    goto LABEL_9;
  v15 = (size_t)a2[2];
  v16 = (const void *)a1[4];
  v17 = *a2;
  if ( v13 + v12 > v15 )
  {
    v18 = v13 + v12 + 992;
    v19 = 2 * v15;
    if ( v18 > v19 )
      a2[2] = (char *)v18;
    else
      a2[2] = (char *)v19;
    v20 = realloc(v17);
    *a2 = (char *)v20;
    v17 = (char *)v20;
    if ( v20 )
    {
      v12 = (__int64)a2[1];
      goto LABEL_18;
    }
LABEL_21:
    abort();
  }
LABEL_18:
  result = (__int64)memcpy(&v17[v12], v16, v13);
  a2[1] += v13;
LABEL_9:
  v14 = (_BYTE *)a1[5];
  if ( v14 )
  {
    (*(void (__fastcall **)(_QWORD, char **))(*(_QWORD *)v14 + 32LL))(a1[5], a2);
    result = v14[9] & 0xC0;
    if ( (v14[9] & 0xC0) != 0x40 )
      return (*(__int64 (__fastcall **)(_BYTE *, char **))(*(_QWORD *)v14 + 40LL))(v14, a2);
  }
  return result;
}
