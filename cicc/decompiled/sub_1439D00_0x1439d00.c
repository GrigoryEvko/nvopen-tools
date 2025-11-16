// Function: sub_1439D00
// Address: 0x1439d00
//
bool __fastcall sub_1439D00(int a1)
{
  bool result; // al

  result = 0;
  if ( a1 <= 8 )
  {
    result = a1 != 3;
    if ( a1 >= 4 )
      return 1;
  }
  return result;
}
