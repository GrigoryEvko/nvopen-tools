// Function: sub_65BAF0
// Address: 0x65baf0
//
__int64 __fastcall sub_65BAF0(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  result = *(_QWORD *)(a1 + 8);
  if ( (result & 0x2000) != 0 )
  {
    *(_BYTE *)(a2 + 193) |= 0x80u;
    result = *(_QWORD *)(a1 + 8);
  }
  if ( (result & 0x59100E) == 0 )
  {
    if ( (*(_BYTE *)(a1 + 120) & 0x7F) == 0 )
      return result;
    return sub_6851C0(749, a1 + 72);
  }
  result = sub_6851C0(2882, a1 + 32);
  if ( (*(_BYTE *)(a1 + 120) & 0x7F) != 0 )
    return sub_6851C0(749, a1 + 72);
  return result;
}
