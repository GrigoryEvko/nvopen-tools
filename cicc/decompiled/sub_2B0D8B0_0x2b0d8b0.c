// Function: sub_2B0D8B0
// Address: 0x2b0d8b0
//
__int64 __fastcall sub_2B0D8B0(unsigned __int8 *a1)
{
  int v1; // edx
  __int64 result; // rax

  v1 = *a1;
  result = 0;
  if ( (unsigned __int8)v1 <= 0x15u )
  {
    LOBYTE(result) = (_BYTE)v1 == 5;
    LOBYTE(v1) = (unsigned __int8)v1 <= 3u;
    return (v1 | (unsigned int)result) ^ 1;
  }
  return result;
}
