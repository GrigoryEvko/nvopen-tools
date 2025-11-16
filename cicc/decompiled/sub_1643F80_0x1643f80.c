// Function: sub_1643F80
// Address: 0x1643f80
//
bool __fastcall sub_1643F80(__int64 a1)
{
  bool result; // al

  result = sub_1643F60(a1);
  if ( result )
    return *(_BYTE *)(a1 + 8) != 12;
  return result;
}
