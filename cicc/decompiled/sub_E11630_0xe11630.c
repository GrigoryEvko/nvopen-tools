// Function: sub_E11630
// Address: 0xe11630
//
unsigned __int64 __fastcall sub_E11630(__int64 a1, char **a2)
{
  char *v4; // rsi
  unsigned __int64 v5; // rax
  char *v6; // rdi
  unsigned __int64 v7; // rsi
  unsigned __int64 v8; // rax
  __int64 v9; // rax
  _BYTE *v10; // r12
  char *v11; // rsi
  unsigned __int64 result; // rax
  char *v13; // rdi
  unsigned __int64 v14; // rsi
  unsigned __int64 v15; // rax

  v4 = a2[1];
  v5 = (unsigned __int64)a2[2];
  v6 = *a2;
  if ( (unsigned __int64)(v4 + 13) > v5 )
  {
    v7 = (unsigned __int64)(v4 + 1005);
    v8 = 2 * v5;
    if ( v7 <= v8 )
      a2[2] = (char *)v8;
    else
      a2[2] = (char *)v7;
    v9 = realloc(v6);
    *a2 = (char *)v9;
    v6 = (char *)v9;
    if ( !v9 )
      goto LABEL_16;
    v4 = a2[1];
  }
  qmemcpy(&v6[(_QWORD)v4], "pixel vector[", 13);
  a2[1] += 13;
  v10 = *(_BYTE **)(a1 + 16);
  (*(void (__fastcall **)(_BYTE *, char **))(*(_QWORD *)v10 + 32LL))(v10, a2);
  if ( (v10[9] & 0xC0) != 0x40 )
    (*(void (__fastcall **)(_BYTE *, char **))(*(_QWORD *)v10 + 40LL))(v10, a2);
  v11 = a2[1];
  result = (unsigned __int64)a2[2];
  v13 = *a2;
  if ( (unsigned __int64)(v11 + 1) > result )
  {
    v14 = (unsigned __int64)(v11 + 993);
    v15 = 2 * result;
    if ( v14 > v15 )
      a2[2] = (char *)v14;
    else
      a2[2] = (char *)v15;
    result = realloc(v13);
    *a2 = (char *)result;
    v13 = (char *)result;
    if ( result )
    {
      v11 = a2[1];
      goto LABEL_13;
    }
LABEL_16:
    abort();
  }
LABEL_13:
  v11[(_QWORD)v13] = 93;
  ++a2[1];
  return result;
}
