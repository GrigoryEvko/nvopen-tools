// Function: sub_65CD60
// Address: 0x65cd60
//
__int64 __fastcall sub_65CD60(_QWORD *a1)
{
  __int64 result; // rax
  _QWORD v2[62]; // [rsp+0h] [rbp-1F0h] BYREF

  memset(v2, 0, 0x1D8u);
  v2[19] = v2;
  v2[3] = *(_QWORD *)&dword_4F063F8;
  if ( dword_4F077BC && qword_4F077A8 <= 0x9F5Fu )
    BYTE2(v2[22]) |= 1u;
  sub_65C7C0((__int64)v2);
  sub_64EC60((__int64)v2);
  result = v2[36];
  *a1 = v2[36];
  return result;
}
