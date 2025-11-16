// Function: sub_B4DE00
// Address: 0xb4de00
//
__int64 __fastcall sub_B4DE00(__int64 a1, char a2)
{
  int v3; // esi

  v3 = (*(_BYTE *)(a1 + 1) >> 1) & 0x7E;
  if ( a2 )
    v3 = (*(_BYTE *)(a1 + 1) >> 1) | 3;
  return sub_B4DDE0(a1, v3);
}
