// Function: sub_65CF50
// Address: 0x65cf50
//
__int64 __fastcall sub_65CF50(char a1)
{
  _QWORD v2[61]; // [rsp+0h] [rbp-1F0h] BYREF

  memset(v2, 0, 0x1D8u);
  v2[19] = v2;
  v2[3] = *(_QWORD *)&dword_4F063F8;
  if ( dword_4F077BC && qword_4F077A8 <= 0x9F5Fu )
    BYTE2(v2[22]) |= 1u;
  BYTE4(v2[15]) = BYTE4(v2[15]) & 0xFD | (2 * (a1 & 1));
  sub_65C7C0((__int64)v2);
  sub_64EC60((__int64)v2);
  return v2[36];
}
