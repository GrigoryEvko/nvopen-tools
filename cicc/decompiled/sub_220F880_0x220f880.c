// Function: sub_220F880
// Address: 0x220f880
//
__int64 sub_220F880()
{
  _QWORD v1[3]; // [rsp+0h] [rbp-18h] BYREF

  syscall(228, 1, v1);
  return v1[1] + 1000000000LL * v1[0];
}
