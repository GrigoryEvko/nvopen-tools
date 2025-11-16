// Function: sub_AC3670
// Address: 0xac3670
//
char __fastcall sub_AC3670(__int64 a1, __int64 a2)
{
  unsigned int v2; // ebx
  char result; // al
  __int64 v4; // rdx

  v2 = *(_DWORD *)(a1 + 8) >> 8;
  result = sub_BCAC40(a1, 1);
  if ( result )
    return (unsigned __int64)(a2 + 1) <= 2;
  if ( v2 > 0x3F )
    return 1;
  if ( !v2 )
  {
    v4 = 0;
    if ( a2 < 0 )
      return result;
    return a2 <= v4;
  }
  if ( a2 >= -(1LL << ((unsigned __int8)v2 - 1)) )
  {
    v4 = (1LL << ((unsigned __int8)v2 - 1)) - 1;
    return a2 <= v4;
  }
  return result;
}
