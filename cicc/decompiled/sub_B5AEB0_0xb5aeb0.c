// Function: sub_B5AEB0
// Address: 0xb5aeb0
//
__int64 __fastcall sub_B5AEB0(unsigned int a1)
{
  __int64 result; // rax
  __int64 v2; // rdx

  result = a1;
  if ( a1 > 0xA8 )
  {
    if ( a1 - 404 <= 0x58 )
      return result;
  }
  else if ( a1 > 0xA3 )
  {
    return result;
  }
  v2 = a1 - 1;
  result = 0;
  if ( (unsigned int)v2 <= 0x190 )
    return word_3F2C6A0[v2];
  return result;
}
