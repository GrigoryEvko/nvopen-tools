// Function: sub_1D16EF0
// Address: 0x1d16ef0
//
__int64 __fastcall sub_1D16EF0(int a1, char a2)
{
  unsigned int v3; // edi
  __int64 result; // rax

  v3 = a1 ^ 7;
  result = a1 ^ 0xFu;
  if ( a2 )
    result = v3;
  if ( (unsigned int)result > 0x17 )
    return (unsigned int)result & 0xFFFFFFF7;
  return result;
}
