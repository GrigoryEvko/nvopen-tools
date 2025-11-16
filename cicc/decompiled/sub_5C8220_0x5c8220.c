// Function: sub_5C8220
// Address: 0x5c8220
//
__int64 __fastcall sub_5C8220(__int64 a1, __int64 a2)
{
  if ( *(char *)(a2 + 192) < 0 || unk_4F077A8 <= 0x9EFBu )
  {
    sub_736C90(a2, 1);
    *(_BYTE *)(a2 + 202) |= 1u;
    return a2;
  }
  else
  {
    if ( (*(_BYTE *)(a1 + 11) & 1) != 0 )
      sub_684B30(2473, a1 + 56);
    *(_BYTE *)(a1 + 8) = 0;
    return a2;
  }
}
