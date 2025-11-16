// Function: sub_CEA400
// Address: 0xcea400
//
bool __fastcall sub_CEA400(unsigned int a1)
{
  bool result; // al

  if ( a1 > 0x2655 )
    return a1 - 9816 <= 5;
  result = 1;
  if ( a1 <= 0x2649 )
    return a1 - 9474 <= 5;
  return result;
}
