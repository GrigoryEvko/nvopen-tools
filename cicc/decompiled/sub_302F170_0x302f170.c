// Function: sub_302F170
// Address: 0x302f170
//
__int64 __fastcall sub_302F170(__int64 a1)
{
  unsigned int v2; // eax
  unsigned __int8 v3; // dl

  if ( (unsigned __int8)(*(_BYTE *)(a1 + 8) - 15) <= 3u )
    return 1;
  if ( sub_BCAC40(a1, 128) )
    return 1;
  v3 = *(_BYTE *)(a1 + 8);
  LOBYTE(v2) = v3 == 5;
  return (v3 <= 1u) | v2;
}
