// Function: sub_1439CD0
// Address: 0x1439cd0
//
bool __fastcall sub_1439CD0(int a1)
{
  bool result; // al

  result = 1;
  if ( a1 != 6 )
  {
    result = 0;
    if ( a1 <= 6 )
      return a1 < 3;
  }
  return result;
}
