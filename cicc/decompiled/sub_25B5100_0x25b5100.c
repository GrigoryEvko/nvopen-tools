// Function: sub_25B5100
// Address: 0x25b5100
//
_BOOL8 __fastcall sub_25B5100(__int64 a1, __int64 a2)
{
  unsigned __int8 v2; // cl
  _BOOL8 result; // rax
  unsigned __int8 v4; // cl

  v2 = **(_BYTE **)(a2 + 24);
  result = 1;
  if ( v2 > 0x1Cu )
  {
    v4 = v2 - 34;
    if ( v4 <= 0x33u )
      return ((0x8000000000041uLL >> v4) & 1) == 0;
  }
  return result;
}
