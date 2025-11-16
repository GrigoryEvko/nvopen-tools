// Function: sub_15E4FF0
// Address: 0x15e4ff0
//
bool __fastcall sub_15E4FF0(__int64 a1)
{
  bool result; // al

  result = *(_BYTE *)(a1 + 16) == 3 || *(_BYTE *)(a1 + 16) == 0;
  if ( result )
    return sub_1626AA0(a1, 21) != 0;
  return result;
}
