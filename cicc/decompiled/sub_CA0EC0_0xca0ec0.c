// Function: sub_CA0EC0
// Address: 0xca0ec0
//
__int64 __fastcall sub_CA0EC0(__int64 a1, __int64 a2)
{
  _QWORD v3[10]; // [rsp+0h] [rbp-50h] BYREF

  v3[5] = 0x100000000LL;
  v3[6] = a2;
  v3[1] = 2;
  v3[0] = &unk_49DD288;
  memset(&v3[2], 0, 24);
  sub_CB5980(v3, 0, 0, 0);
  sub_CA0E80(a1, (__int64)v3);
  v3[0] = &unk_49DD388;
  return sub_CB5840(v3);
}
