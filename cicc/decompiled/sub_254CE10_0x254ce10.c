// Function: sub_254CE10
// Address: 0x254ce10
//
__int64 __fastcall sub_254CE10(__int64 a1, __int64 a2)
{
  unsigned __int8 *v2; // rax
  unsigned __int8 v4; // [rsp+Ah] [rbp-36h] BYREF
  char v5; // [rsp+Bh] [rbp-35h] BYREF
  int v6; // [rsp+Ch] [rbp-34h] BYREF
  _QWORD v7[6]; // [rsp+10h] [rbp-30h] BYREF

  v4 = 0;
  v2 = sub_250CBE0((__int64 *)(a1 + 72), a2);
  v7[2] = sub_ACA8A0(**(__int64 ****)(*((_QWORD *)v2 + 3) + 16LL));
  v7[0] = &v4;
  v7[1] = a2;
  v5 = 0;
  v6 = 1;
  sub_2526370(
    a2,
    (__int64 (__fastcall *)(__int64, unsigned __int64, __int64))sub_256EE00,
    (__int64)v7,
    a1,
    &v6,
    1,
    &v5,
    0,
    0);
  return v4 ^ 1u;
}
