// Function: sub_730740
// Address: 0x730740
//
_BOOL8 __fastcall sub_730740(__int64 a1)
{
  _BOOL8 result; // rax
  unsigned __int8 v2; // cl

  result = 0;
  if ( *(_BYTE *)(a1 + 24) == 1 )
  {
    v2 = *(_BYTE *)(a1 + 56);
    if ( v2 <= 0x14u )
      return ((1LL << v2) & 0x1FD5E0) != 0;
  }
  return result;
}
