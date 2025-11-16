// Function: sub_228AA50
// Address: 0x228aa50
//
_BOOL8 __fastcall sub_228AA50(__int64 a1)
{
  _BOOL8 result; // rax

  if ( *(_BYTE *)a1 == 61 )
  {
    result = !(*(_WORD *)(a1 + 2) & 1);
    if ( ((*(_WORD *)(a1 + 2) >> 7) & 6) != 0 )
      return 0;
  }
  else
  {
    result = 0;
    if ( *(_BYTE *)a1 == 62 && ((*(_WORD *)(a1 + 2) >> 7) & 6) == 0 )
      return !(*(_WORD *)(a1 + 2) & 1);
  }
  return result;
}
