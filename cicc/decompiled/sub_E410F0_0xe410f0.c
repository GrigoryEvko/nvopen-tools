// Function: sub_E410F0
// Address: 0xe410f0
//
void *__fastcall sub_E410F0(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD v5[12]; // [rsp+0h] [rbp-60h] BYREF

  v5[5] = 0x100000000LL;
  v5[6] = a2;
  v5[1] = 2;
  v5[0] = &unk_49DD288;
  memset(&v5[2], 0, 24);
  sub_CB5980((__int64)v5, 0, 0, 0);
  sub_E409B0(a1, (__int64)v5, a3);
  v5[0] = &unk_49DD388;
  return sub_CB5840((__int64)v5);
}
