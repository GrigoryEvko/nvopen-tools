// Function: sub_F6F0E0
// Address: 0xf6f0e0
//
__int64 __fastcall sub_F6F0E0(int a1)
{
  __int64 v1; // rdi
  __int64 result; // rax

  v1 = (unsigned int)(a1 - 390);
  result = 0;
  if ( (unsigned int)v1 <= 0xA )
    return *(unsigned int *)&asc_3F8AD60[4 * v1];
  return result;
}
