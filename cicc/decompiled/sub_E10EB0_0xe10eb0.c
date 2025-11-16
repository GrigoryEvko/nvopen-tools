// Function: sub_E10EB0
// Address: 0xe10eb0
//
__int64 __fastcall sub_E10EB0(__int64 a1, char **a2)
{
  _BYTE *v4; // r13
  char *v5; // rsi
  unsigned __int64 v6; // rax
  __int64 v7; // rdi
  unsigned __int64 v8; // rsi
  unsigned __int64 v9; // rax
  __int64 v10; // rax
  _BYTE *v11; // r13
  __int64 result; // rax

  v4 = *(_BYTE **)(a1 + 16);
  (*(void (__fastcall **)(_BYTE *))(*(_QWORD *)v4 + 32LL))(v4);
  if ( (v4[9] & 0xC0) != 0x40 )
    (*(void (__fastcall **)(_BYTE *, char **))(*(_QWORD *)v4 + 40LL))(v4, a2);
  v5 = a2[1];
  v6 = (unsigned __int64)a2[2];
  v7 = (__int64)*a2;
  if ( (unsigned __int64)(v5 + 2) > v6 )
  {
    v8 = (unsigned __int64)(v5 + 994);
    v9 = 2 * v6;
    if ( v8 > v9 )
      a2[2] = (char *)v8;
    else
      a2[2] = (char *)v9;
    v10 = realloc((void *)v7);
    *a2 = (char *)v10;
    v7 = v10;
    if ( !v10 )
      abort();
    v5 = a2[1];
  }
  *(_WORD *)&v5[v7] = 14906;
  a2[1] += 2;
  v11 = *(_BYTE **)(a1 + 24);
  (*(void (__fastcall **)(_BYTE *, char **))(*(_QWORD *)v11 + 32LL))(v11, a2);
  result = v11[9] & 0xC0;
  if ( (v11[9] & 0xC0) != 0x40 )
    return (*(__int64 (__fastcall **)(_BYTE *, char **))(*(_QWORD *)v11 + 40LL))(v11, a2);
  return result;
}
