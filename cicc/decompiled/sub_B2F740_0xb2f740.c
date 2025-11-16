// Function: sub_B2F740
// Address: 0xb2f740
//
__int64 __fastcall sub_B2F740(__int64 a1, unsigned __int16 a2)
{
  unsigned __int16 v2; // dx
  int v3; // ecx

  v2 = *(_WORD *)(a1 + 34);
  v3 = 0;
  if ( HIBYTE(a2) )
    v3 = (unsigned __int8)a2 + 1;
  *(_WORD *)(a1 + 34) = *(_WORD *)(a1 + 34) & 1 | (2 * (v3 | (*(_WORD *)(a1 + 34) >> 1) & 0x7FC0));
  return v2 & 1 | (2 * (v3 | (v2 >> 1) & 0x7FC0u));
}
