// Function: sub_1454B60
// Address: 0x1454b60
//
__int64 __fastcall sub_1454B60(unsigned __int64 a1)
{
  unsigned __int64 v1; // rdi

  v1 = (((a1 >> 1) | a1 | (((a1 >> 1) | a1) >> 2)) >> 4)
     | (a1 >> 1)
     | a1
     | (((a1 >> 1) | a1) >> 2)
     | (((((a1 >> 1) | a1 | (((a1 >> 1) | a1) >> 2)) >> 4) | (a1 >> 1) | a1 | (((a1 >> 1) | a1) >> 2)) >> 8);
  return ((v1 >> 16) | v1 | (((v1 >> 16) | v1) >> 32)) + 1;
}
