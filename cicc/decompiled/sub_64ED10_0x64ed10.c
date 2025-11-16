// Function: sub_64ED10
// Address: 0x64ed10
//
__int64 sub_64ED10()
{
  __int64 v0; // r13
  _QWORD v2[12]; // [rsp+0h] [rbp-250h] BYREF
  _QWORD v3[62]; // [rsp+60h] [rbp-1F0h] BYREF

  memset(v3, 0, 0x1D8u);
  v3[19] = v3;
  v3[3] = *(_QWORD *)&dword_4F063F8;
  if ( dword_4F077BC && qword_4F077A8 <= 0x9F5Fu )
    BYTE2(v3[22]) |= 1u;
  memset(v2, 0, 0x58u);
  sub_672A20(524290, v3, v2);
  v0 = v3[34];
  if ( (v3[15] & 0x2000000000LL) != 0 )
    sub_6451E0((__int64)v3);
  *(_QWORD *)dword_4F07508 = v3[3];
  unk_4F061D8 = v2[5];
  return v0;
}
