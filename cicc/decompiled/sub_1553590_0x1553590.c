// Function: sub_1553590
// Address: 0x1553590
//
bool __fastcall sub_1553590(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int8 v5; // cl
  bool result; // al

  if ( (*(_BYTE *)(a1 + 23) & 0x20) != 0 || (v5 = *(_BYTE *)(a1 + 16), v5 <= 3u) || (result = v5 > 0x10u && v5 != 19) )
  {
    sub_1550E20(a2, a1, 0, a3, a4);
    return 1;
  }
  return result;
}
