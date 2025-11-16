// Function: sub_E108A0
// Address: 0xe108a0
//
__int64 __fastcall sub_E108A0(__int64 a1, char **a2)
{
  char *v4; // rsi
  unsigned __int64 v5; // rax
  __int64 v6; // rdi
  unsigned __int64 v7; // rsi
  unsigned __int64 v8; // rax
  __int64 v9; // rax
  _BYTE *v10; // r13
  __int64 result; // rax

  v4 = a2[1];
  v5 = (unsigned __int64)a2[2];
  v6 = (__int64)*a2;
  if ( (unsigned __int64)(v4 + 2) > v5 )
  {
    v7 = (unsigned __int64)(v4 + 994);
    v8 = 2 * v5;
    if ( v7 > v8 )
      a2[2] = (char *)v7;
    else
      a2[2] = (char *)v8;
    v9 = realloc((void *)v6);
    *a2 = (char *)v9;
    v6 = v9;
    if ( !v9 )
      abort();
    v4 = a2[1];
  }
  *(_WORD *)&v4[v6] = 14906;
  a2[1] += 2;
  v10 = *(_BYTE **)(a1 + 16);
  (*(void (__fastcall **)(_BYTE *, char **))(*(_QWORD *)v10 + 32LL))(v10, a2);
  result = v10[9] & 0xC0;
  if ( (v10[9] & 0xC0) != 0x40 )
    return (*(__int64 (__fastcall **)(_BYTE *, char **))(*(_QWORD *)v10 + 40LL))(v10, a2);
  return result;
}
