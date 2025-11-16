// Function: sub_8800F0
// Address: 0x8800f0
//
__int64 __fastcall sub_8800F0(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  int v3; // [rsp+Ch] [rbp-284h] BYREF
  __int64 v4; // [rsp+10h] [rbp-280h] BYREF
  __int64 v5; // [rsp+18h] [rbp-278h] BYREF
  _BYTE v6[112]; // [rsp+20h] [rbp-270h] BYREF
  _QWORD v7[64]; // [rsp+90h] [rbp-200h] BYREF

  sub_87E3B0((__int64)v6);
  memset(v7, 0, 0x1D8u);
  v7[19] = v7;
  v7[3] = *(_QWORD *)&dword_4F063F8;
  if ( dword_4F077BC && qword_4F077A8 <= 0x9F5Fu )
    BYTE2(v7[22]) |= 1u;
  WORD2(v7[33]) = 257;
  v7[36] = a2;
  sub_6523A0(a1, (__int64)v7, (__int64)v6, 1, &v3, &v5, &v4, 0);
  result = v7[0];
  *(_BYTE *)(*(_QWORD *)(v7[0] + 88LL) + 193LL) |= 0x10u;
  return result;
}
