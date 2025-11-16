// Function: sub_1683C90
// Address: 0x1683c90
//
__int64 __fastcall sub_1683C90(unsigned int a1)
{
  __int64 result; // rax

  for ( result = (unsigned int)-((a1 & (a1 - 1)) == 0); a1; a1 >>= 1 )
    result = (unsigned int)(result + 1);
  return result;
}
