// Function: sub_B52830
// Address: 0xb52830
//
bool __fastcall sub_B52830(unsigned int a1)
{
  bool result; // al

  if ( a1 - 32 <= 9 )
    return a1 - 32 <= 1;
  if ( a1 > 0xF )
    BUG();
  result = a1 == 6 || a1 == 1;
  if ( !result )
    return a1 == 14 || a1 == 9;
  return result;
}
