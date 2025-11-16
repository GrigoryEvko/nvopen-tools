// Function: sub_310F810
// Address: 0x310f810
//
__int64 __fastcall sub_310F810(__int64 a1)
{
  if ( (*(_BYTE *)(a1 + 32) & 0xFu) - 7 > 1 )
    return 0;
  else
    return (unsigned int)sub_B2DDD0(a1, 0, 0, 1, 0, 0, 0) ^ 1;
}
