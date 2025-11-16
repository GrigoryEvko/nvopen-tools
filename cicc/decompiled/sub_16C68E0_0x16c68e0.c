// Function: sub_16C68E0
// Address: 0x16c68e0
//
__int64 sub_16C68E0()
{
  struct mallinfo v1; // [rsp+0h] [rbp-30h] BYREF

  mallinfo(&v1);
  return v1.uordblks;
}
