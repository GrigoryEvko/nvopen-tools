// Function: sub_396B790
// Address: 0x396b790
//
__int64 __fastcall sub_396B790(__int64 a1, __int64 a2, unsigned int a3)
{
  unsigned int v4; // eax
  unsigned int v6; // ebx
  unsigned int v7; // eax
  __int64 v8; // rdx

  if ( *(_BYTE *)(a1 + 16) == 3 )
  {
    v4 = sub_15AB000(a2, a1);
    if ( v4 >= a3 )
      a3 = v4;
  }
  if ( !(unsigned int)sub_15E4C60(a1) )
    return a3;
  v6 = -1;
  v7 = sub_15E4C60(a1);
  if ( v7 )
  {
    _BitScanReverse(&v7, v7);
    v6 = 31 - (v7 ^ 0x1F);
  }
  if ( a3 < v6 )
    return v6;
  sub_15E64D0(a1);
  if ( v8 )
    return v6;
  return a3;
}
