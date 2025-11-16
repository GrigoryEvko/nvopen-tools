// Function: sub_127B330
// Address: 0x127b330
//
_BOOL8 __fastcall sub_127B330(__int64 a1)
{
  _BOOL8 result; // rax

  result = 0;
  if ( !unk_4D04630 )
  {
    result = 1;
    if ( (*(_BYTE *)(a1 + 201) & 2) == 0 )
      return (*(_BYTE *)(a1 + 198) & 0x20) != 0;
  }
  return result;
}
