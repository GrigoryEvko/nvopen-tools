// Function: sub_2E84660
// Address: 0x2e84660
//
__int64 __fastcall sub_2E84660(__int64 *a1, __int64 a2)
{
  if ( (*(_BYTE *)(a2 + 32) & 0xF) == 1 )
    return 0;
  else
    return sub_2E830A0(a1, a2);
}
