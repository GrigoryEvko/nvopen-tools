// Function: sub_1B47330
// Address: 0x1b47330
//
__int64 __fastcall sub_1B47330(__int64 a1, __int64 a2, __int64 a3, unsigned int *a4, __int64 **a5, int a6)
{
  unsigned __int8 v6; // al
  unsigned int v8; // r8d

  if ( a6 == dword_4FB7060 )
  {
    return 0;
  }
  else
  {
    v6 = *(_BYTE *)(a1 + 16);
    if ( v6 > 0x17u )
      return sub_1B47110(a1, a2, a3, a4, a5, a6);
    v8 = 1;
    if ( v6 == 5 )
      return (unsigned int)sub_1593DF0(a1, a2, a3, a4) ^ 1;
  }
  return v8;
}
