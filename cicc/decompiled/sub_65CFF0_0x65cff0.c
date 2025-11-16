// Function: sub_65CFF0
// Address: 0x65cff0
//
__int64 __fastcall sub_65CFF0(_BOOL4 *a1, char a2)
{
  __int64 result; // rax
  _BOOL4 v3; // edx
  _QWORD v4[62]; // [rsp+0h] [rbp-1F0h] BYREF

  memset(v4, 0, 0x1D8u);
  v4[19] = v4;
  v4[3] = *(_QWORD *)&dword_4F063F8;
  if ( dword_4F077BC && qword_4F077A8 <= 0x9F5Fu )
    BYTE2(v4[22]) |= 1u;
  v4[15] |= 0x401000000uLL;
  BYTE4(v4[16]) = (a2 << 7) | BYTE4(v4[16]) & 0x7F;
  sub_65C7C0((__int64)v4);
  sub_64EC60((__int64)v4);
  result = v4[36];
  if ( a1 )
  {
    v3 = 0;
    if ( (v4[16] & 0x2000000000LL) != 0 )
      v3 = (unsigned __int8)(*(_BYTE *)(v4[36] + 140LL) - 9) <= 2u;
    *a1 = v3;
  }
  return result;
}
