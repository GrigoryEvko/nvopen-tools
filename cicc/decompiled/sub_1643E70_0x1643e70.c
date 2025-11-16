// Function: sub_1643E70
// Address: 0x1643e70
//
bool __fastcall sub_1643E70(__int64 a1, unsigned int a2)
{
  bool result; // al

  result = 1;
  if ( *(_BYTE *)(a1 + 8) == 13 )
    return a2 < *(_DWORD *)(a1 + 12);
  return result;
}
