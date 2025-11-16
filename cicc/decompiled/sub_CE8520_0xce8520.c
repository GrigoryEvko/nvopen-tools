// Function: sub_CE8520
// Address: 0xce8520
//
bool __fastcall sub_CE8520(__int64 a1)
{
  bool result; // al

  if ( *(_QWORD *)(a1 + 48) )
    return sub_B91F50(a1, "nv.used_bytes_mask", 0x12u) != 0;
  result = 0;
  if ( (*(_BYTE *)(a1 + 7) & 0x20) != 0 )
    return sub_B91F50(a1, "nv.used_bytes_mask", 0x12u) != 0;
  return result;
}
