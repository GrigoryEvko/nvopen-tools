// Function: sub_72F900
// Address: 0x72f900
//
__int64 __fastcall sub_72F900(__int64 a1, _BYTE *a2)
{
  __int64 result; // rax

  *(_QWORD *)(a1 + 56) = a2;
  result = (a2[170] >> 6 << 7) | *(_BYTE *)(a1 + 50) & 0x7Fu;
  *(_BYTE *)(a1 + 50) = (a2[170] >> 6 << 7) | *(_BYTE *)(a1 + 50) & 0x7F;
  if ( (a2[168] & 0x20) != 0 )
    result = (unsigned int)result | 0x10;
  *(_BYTE *)(a1 + 50) = result;
  if ( (a2[169] & 0x20) != 0 )
  {
    result = (unsigned int)result | 0x40;
    *(_BYTE *)(a1 + 50) = result;
  }
  if ( (a2[171] & 1) != 0 )
    *(_BYTE *)(a1 + 50) |= 0x20u;
  return result;
}
