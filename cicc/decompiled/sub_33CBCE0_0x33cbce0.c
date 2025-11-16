// Function: sub_33CBCE0
// Address: 0x33cbce0
//
__int64 __fastcall sub_33CBCE0(char a1, int a2)
{
  __int64 result; // rax

  if ( a2 == 2 )
    return 213;
  result = 214;
  if ( a2 != 3 )
  {
    if ( a2 != 1 )
      BUG();
    return a1 == 0 ? 215 : 233;
  }
  return result;
}
