// Function: sub_1D1FBF0
// Address: 0x1d1fbf0
//
char __fastcall sub_1D1FBF0(__int64 a1, __int64 a2)
{
  int v2; // edx
  __int64 *v3; // rsi
  __int64 v4; // r8
  char result; // al

  v2 = *(unsigned __int16 *)(a2 + 24);
  if ( v2 != 52 && v2 != 119 )
    return 0;
  v3 = *(__int64 **)(a2 + 32);
  v4 = v3[5];
  result = *(_WORD *)(v4 + 24) == 32 || *(_WORD *)(v4 + 24) == 10;
  if ( result )
  {
    if ( v2 == 119 )
      return sub_1D1F940(a1, *v3, v3[1], *(_QWORD *)(v4 + 88) + 24LL, 0);
  }
  return result;
}
