// Function: sub_E12980
// Address: 0xe12980
//
__int64 __fastcall sub_E12980(__int64 a1, char **a2)
{
  _BYTE *v4; // r13
  __int64 result; // rax
  char *v6; // rsi
  unsigned __int64 v7; // rax
  char *v8; // rdi
  unsigned __int64 v9; // rsi
  unsigned __int64 v10; // rax
  __int64 v11; // rax
  _BYTE *v12; // r13

  v4 = *(_BYTE **)(a1 + 16);
  (*(void (__fastcall **)(_BYTE *))(*(_QWORD *)v4 + 32LL))(v4);
  result = v4[9] & 0xC0;
  if ( (v4[9] & 0xC0) != 0x40 )
    result = (*(__int64 (__fastcall **)(_BYTE *, char **))(*(_QWORD *)v4 + 40LL))(v4, a2);
  if ( *(_QWORD *)(a1 + 40) )
  {
    v6 = a2[1];
    v7 = (unsigned __int64)a2[2];
    v8 = *a2;
    if ( (unsigned __int64)(v6 + 10) > v7 )
    {
      v9 = (unsigned __int64)(v6 + 1002);
      v10 = 2 * v7;
      if ( v9 > v10 )
        a2[2] = (char *)v9;
      else
        a2[2] = (char *)v10;
      v11 = realloc(v8);
      *a2 = (char *)v11;
      v8 = (char *)v11;
      if ( !v11 )
        abort();
      v6 = a2[1];
    }
    qmemcpy(&v8[(_QWORD)v6], " requires ", 10);
    a2[1] += 10;
    v12 = *(_BYTE **)(a1 + 40);
    (*(void (__fastcall **)(_BYTE *, char **))(*(_QWORD *)v12 + 32LL))(v12, a2);
    result = v12[9] & 0xC0;
    if ( (v12[9] & 0xC0) != 0x40 )
      return (*(__int64 (__fastcall **)(_BYTE *, char **))(*(_QWORD *)v12 + 40LL))(v12, a2);
  }
  return result;
}
