// Function: sub_AF1560
// Address: 0xaf1560
//
__int64 __fastcall sub_AF1560(unsigned __int64 a1)
{
  unsigned __int64 v1; // rdi

  v1 = (((a1 >> 1) | a1 | (((a1 >> 1) | a1) >> 2)) >> 4)
     | (a1 >> 1)
     | a1
     | (((a1 >> 1) | a1) >> 2)
     | (((((a1 >> 1) | a1 | (((a1 >> 1) | a1) >> 2)) >> 4) | (a1 >> 1) | a1 | (((a1 >> 1) | a1) >> 2)) >> 8);
  return ((v1 >> 16) | v1 | (((v1 >> 16) | v1) >> 32)) + 1;
}
