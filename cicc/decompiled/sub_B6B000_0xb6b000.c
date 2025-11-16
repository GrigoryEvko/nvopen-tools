// Function: sub_B6B000
// Address: 0xb6b000
//
__int64 __fastcall sub_B6B000(int a1)
{
  __int64 v1; // rdi
  __int64 result; // rax

  v1 = (unsigned int)(a1 - 93);
  result = 0;
  if ( (unsigned int)v1 <= 0x30 )
    return byte_3F2D160[v1];
  return result;
}
