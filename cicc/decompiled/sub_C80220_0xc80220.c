// Function: sub_C80220
// Address: 0xc80220
//
bool __fastcall sub_C80220(char a1, unsigned int a2)
{
  bool result; // al

  result = 1;
  if ( a1 != 47 )
  {
    result = 0;
    if ( a2 > 1 )
      return a1 == 92;
  }
  return result;
}
