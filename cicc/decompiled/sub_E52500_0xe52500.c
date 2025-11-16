// Function: sub_E52500
// Address: 0xe52500
//
_BYTE *__fastcall sub_E52500(__int64 a1, char a2, __int64 a3, unsigned int a4)
{
  unsigned int v4; // edx
  unsigned __int64 v5; // rsi
  __int64 v7; // [rsp+0h] [rbp-10h]

  v4 = *(_DWORD *)(*(_QWORD *)(a1 + 312) + 264LL);
  v5 = 1LL << a2;
  if ( v4 )
    return sub_E521E0(a1, v5, v4, 1, 1, a4);
  else
    return sub_E521E0(a1, v5, v7, 0, 1, a4);
}
