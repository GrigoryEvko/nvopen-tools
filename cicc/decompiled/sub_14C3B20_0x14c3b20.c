// Function: sub_14C3B20
// Address: 0x14c3b20
//
bool __fastcall sub_14C3B20(int a1, int a2)
{
  bool result; // al

  if ( a1 == 147 )
    return a2 == 1;
  result = 0;
  if ( ((a1 - 31) & 0xFFFFFFFD) == 0 )
    return a2 == 1;
  return result;
}
