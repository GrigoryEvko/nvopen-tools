// Function: sub_C4F870
// Address: 0xc4f870
//
bool __fastcall sub_C4F870(__int64 a1)
{
  bool result; // al

  result = 1;
  if ( (*(_BYTE *)(a1 + 13) & 0x10) == 0 )
    return ((*(_WORD *)(a1 + 12) >> 7) & 3u) - 2 <= 1;
  return result;
}
