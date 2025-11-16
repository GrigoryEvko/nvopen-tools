// Function: sub_624060
// Address: 0x624060
//
__int64 __fastcall sub_624060(__int64 a1)
{
  _QWORD v2[12]; // [rsp+0h] [rbp-250h] BYREF
  _QWORD v3[61]; // [rsp+60h] [rbp-1F0h] BYREF

  memset(v3, 0, 0x1D8u);
  v3[19] = v3;
  v3[3] = *(_QWORD *)&dword_4F063F8;
  if ( dword_4F077BC && qword_4F077A8 <= 0x9F5Fu )
    BYTE2(v3[22]) |= 1u;
  memset(v2, 0, 0x58u);
  sub_672A20(1024, v3, v2);
  if ( a1 )
    *(_QWORD *)(a1 + 56) = v2[5];
  return v3[15] & 0x7F;
}
