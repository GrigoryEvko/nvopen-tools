// Function: sub_BCBD30
// Address: 0xbcbd30
//
bool __fastcall sub_BCBD30(__int64 a1)
{
  bool result; // al

  result = sub_BCBD20(a1);
  if ( result )
    return *(_BYTE *)(a1 + 8) != 13;
  return result;
}
