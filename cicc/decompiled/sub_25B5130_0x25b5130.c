// Function: sub_25B5130
// Address: 0x25b5130
//
__int64 __fastcall sub_25B5130(__int64 a1)
{
  __int64 result; // rax

  sub_B2CA40(a1, 0);
  result = *(unsigned __int8 *)(a1 + 32);
  *(_BYTE *)(a1 + 32) &= 0xF0u;
  if ( (result & 0x30) != 0 )
    *(_BYTE *)(a1 + 33) |= 0x40u;
  return result;
}
