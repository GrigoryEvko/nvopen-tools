// Function: sub_825770
// Address: 0x825770
//
_BOOL8 __fastcall sub_825770(__int64 a1)
{
  _BOOL8 result; // rax

  result = 0;
  if ( (*(_BYTE *)(a1 + 193) & 2) != 0 )
  {
    result = 1;
    if ( (*(_BYTE *)(a1 + 193) & 4) == 0 )
      return dword_4D04530 != 0;
  }
  return result;
}
