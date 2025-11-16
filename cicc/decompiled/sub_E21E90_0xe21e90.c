// Function: sub_E21E90
// Address: 0xe21e90
//
__int64 __fastcall sub_E21E90(__int64 a1, __int64 *a2)
{
  unsigned __int8 v2; // al
  unsigned __int8 v4; // r12
  int v5; // edx

  v2 = sub_E21D30(a1, a2);
  if ( !*(_BYTE *)(a1 + 8) )
  {
    if ( *a2 )
    {
      v4 = v2;
      v5 = (unsigned __int8)sub_E21D30(a1, a2);
      if ( !*(_BYTE *)(a1 + 8) )
        return v5 | (v4 << 8);
    }
  }
  *(_BYTE *)(a1 + 8) = 1;
  return 0;
}
