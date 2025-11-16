// Function: sub_16C36C0
// Address: 0x16c36c0
//
bool __fastcall sub_16C36C0(char a1, int a2)
{
  bool result; // al

  result = 1;
  if ( a1 != 47 )
  {
    result = 0;
    if ( !a2 )
      return a1 == 92;
  }
  return result;
}
