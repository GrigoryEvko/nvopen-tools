// Function: sub_8D1A80
// Address: 0x8d1a80
//
__int64 __fastcall sub_8D1A80(__int64 a1, _DWORD *a2)
{
  if ( *(_BYTE *)(a1 + 140) != 8 || (*(_BYTE *)(a1 + 169) & 2) == 0 )
    return 0;
  if ( sub_72D8B0(a1) )
    return 0;
  *a2 = 1;
  return 1;
}
