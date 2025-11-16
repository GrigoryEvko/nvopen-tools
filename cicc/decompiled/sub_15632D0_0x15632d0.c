// Function: sub_15632D0
// Address: 0x15632d0
//
__int64 __fastcall sub_15632D0(__int64 *a1, __int64 *a2, _QWORD *a3)
{
  __int64 v4; // r12
  __m128i v6; // [rsp+0h] [rbp-80h] BYREF
  _QWORD *v7; // [rsp+18h] [rbp-68h]

  sub_1563030(&v6, *a1);
  sub_1561FA0((__int64)&v6, a3);
  v4 = sub_1560BF0(a2, &v6);
  sub_155CC10(v7);
  return v4;
}
