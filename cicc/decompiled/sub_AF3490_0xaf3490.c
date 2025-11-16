// Function: sub_AF3490
// Address: 0xaf3490
//
__int64 __fastcall sub_AF3490(char a1, char a2, char a3, char a4, char a5)
{
  __int64 result; // rax
  unsigned int v6; // edx

  result = a4 & 3;
  if ( a1 )
    result = a4 & 3 | 4u;
  if ( a2 )
    result = (unsigned int)result | 8;
  if ( a3 )
    result = (unsigned int)result | 0x10;
  v6 = result;
  if ( a5 )
  {
    BYTE1(v6) = BYTE1(result) | 1;
    return v6;
  }
  return result;
}
