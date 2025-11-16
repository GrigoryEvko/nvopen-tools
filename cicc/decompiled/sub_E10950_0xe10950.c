// Function: sub_E10950
// Address: 0xe10950
//
__int64 __fastcall sub_E10950(__int64 a1, char **a2)
{
  char *v4; // rsi
  unsigned __int64 v5; // rax
  char *v6; // rdi
  unsigned __int64 v7; // rsi
  unsigned __int64 v8; // rax
  __int64 v9; // rax
  char *v10; // rdi
  _BYTE *v11; // r13
  __int64 result; // rax

  v4 = a2[1];
  v5 = (unsigned __int64)a2[2];
  v6 = *a2;
  if ( (unsigned __int64)(v4 + 5) > v5 )
  {
    v7 = (unsigned __int64)(v4 + 997);
    v8 = 2 * v5;
    if ( v7 > v8 )
      a2[2] = (char *)v7;
    else
      a2[2] = (char *)v8;
    v9 = realloc(v6);
    *a2 = (char *)v9;
    v6 = (char *)v9;
    if ( !v9 )
      abort();
    v4 = a2[1];
  }
  v10 = &v6[(_QWORD)v4];
  *(_DWORD *)v10 = 1936287860;
  v10[4] = 32;
  a2[1] += 5;
  v11 = *(_BYTE **)(a1 + 16);
  (*(void (__fastcall **)(_BYTE *, char **))(*(_QWORD *)v11 + 32LL))(v11, a2);
  result = v11[9] & 0xC0;
  if ( (v11[9] & 0xC0) != 0x40 )
    return (*(__int64 (__fastcall **)(_BYTE *, char **))(*(_QWORD *)v11 + 40LL))(v11, a2);
  return result;
}
