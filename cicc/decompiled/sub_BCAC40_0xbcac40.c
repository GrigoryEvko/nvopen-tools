// Function: sub_BCAC40
// Address: 0xbcac40
//
bool __fastcall sub_BCAC40(__int64 a1, int a2)
{
  bool result; // al

  result = 0;
  if ( *(_BYTE *)(a1 + 8) == 12 )
    return *(_DWORD *)(a1 + 8) >> 8 == a2;
  return result;
}
