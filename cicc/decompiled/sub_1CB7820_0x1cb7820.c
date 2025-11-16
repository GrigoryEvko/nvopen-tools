// Function: sub_1CB7820
// Address: 0x1cb7820
//
__int64 __fastcall sub_1CB7820(unsigned int *a1, unsigned __int64 a2)
{
  unsigned __int64 v3; // rsi
  int v4; // eax
  int v5; // edx
  int v7; // r14d

  v3 = *(_QWORD *)(a2 - 24);
  if ( *(_BYTE *)(*(_QWORD *)v3 + 8LL) == 15 )
  {
    v7 = sub_1CB76C0(a1, v3);
    if ( v7 == (unsigned int)sub_1CB76C0(a1, a2) )
      return 0;
    sub_1CB7560(a1, a2, v7);
    return 1;
  }
  else
  {
    v4 = sub_1CB76C0(a1, a2);
    v5 = a1[1];
    if ( v5 == v4 )
      return 0;
    sub_1CB7560(a1, a2, v5);
    return 1;
  }
}
