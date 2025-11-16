// Function: sub_16C2170
// Address: 0x16c2170
//
bool __fastcall sub_16C2170(int a1, __int64 a2, __int64 a3, __int64 a4, int a5)
{
  bool result; // al

  if ( a2 == -1 )
  {
    a2 = 0;
    if ( (unsigned int)sub_16C5920(a1) )
      return 0;
  }
  result = 0;
  if ( a4 + a3 == a2 )
    return (a2 & (a5 - 1)) != 0;
  return result;
}
