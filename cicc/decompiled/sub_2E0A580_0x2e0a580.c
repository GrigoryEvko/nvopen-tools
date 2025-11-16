// Function: sub_2E0A580
// Address: 0x2e0a580
//
char *__fastcall sub_2E0A580(__int64 a1, char *a2, char a3)
{
  _QWORD *v4; // rsi
  __int64 v6; // r14
  unsigned int v7; // eax
  __int64 v8; // rdx

  v4 = a2 + 24;
  v6 = *(v4 - 1);
  v7 = *(_DWORD *)(a1 + 8);
  v8 = *(_QWORD *)a1 + 24LL * v7;
  if ( (_QWORD *)v8 != v4 )
  {
    memmove(a2, v4, v8 - (_QWORD)v4);
    v7 = *(_DWORD *)(a1 + 8);
  }
  *(_DWORD *)(a1 + 8) = v7 - 1;
  if ( a3 )
    sub_2E0A490(a1, v6);
  return a2;
}
