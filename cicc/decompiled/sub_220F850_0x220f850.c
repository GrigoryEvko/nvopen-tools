// Function: sub_220F850
// Address: 0x220f850
//
__int64 sub_220F850()
{
  _QWORD v1[3]; // [rsp+0h] [rbp-18h] BYREF

  syscall(228, 0, v1);
  return v1[1] + 1000000000LL * v1[0];
}
