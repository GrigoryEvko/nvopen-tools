// Function: sub_E5CB00
// Address: 0xe5cb00
//
__int64 __fastcall sub_E5CB00(__int64 *a1, __int64 a2)
{
  if ( (*(_BYTE *)(a2 + 48) & 0x20) != 0 )
    return 0;
  else
    return sub_E5CAC0(a1, a2);
}
