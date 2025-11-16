// Function: sub_7B3970
// Address: 0x7b3970
//
__int64 __fastcall sub_7B3970(char a1)
{
  __int64 result; // rax

  if ( a1 > 63 )
  {
    if ( a1 > 95 )
      return (unsigned __int8)(a1 - 97) > 0x1Du;
    else
      return a1 == 64;
  }
  else
  {
    result = 1;
    if ( a1 > 8 )
      return ((1LL << a1) & 0xFFFFFFEF00001E00LL) == 0;
  }
  return result;
}
