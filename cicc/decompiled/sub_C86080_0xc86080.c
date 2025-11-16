// Function: sub_C86080
// Address: 0xc86080
//
__int64 sub_C86080()
{
  struct mallinfo v1; // [rsp+0h] [rbp-30h] BYREF

  mallinfo(&v1);
  return v1.uordblks;
}
