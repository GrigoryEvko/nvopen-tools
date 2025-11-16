// Function: sub_164DB40
// Address: 0x164db40
//
bool __fastcall sub_164DB40(char a1)
{
  bool result; // al

  result = 1;
  if ( (unsigned __int8)(a1 - 48) > 9u )
    return (unsigned __int8)((a1 & 0xDF) - 65) <= 5u;
  return result;
}
