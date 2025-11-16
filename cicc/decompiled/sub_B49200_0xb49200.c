// Function: sub_B49200
// Address: 0xb49200
//
bool __fastcall sub_B49200(__int64 a1)
{
  bool result; // al

  result = 0;
  if ( *(_BYTE *)a1 == 85 )
    return (*(_WORD *)(a1 + 2) & 3) == 2;
  return result;
}
