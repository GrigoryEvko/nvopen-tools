// Function: sub_C8D230
// Address: 0xc8d230
//
void __fastcall __noreturn sub_C8D230(unsigned __int64 a1, __int64 a2, __int64 a3, unsigned __int64 a4)
{
  __m128i v4[2]; // [rsp+0h] [rbp-60h] BYREF
  _QWORD v5[4]; // [rsp+20h] [rbp-40h] BYREF
  __int16 v6; // [rsp+40h] [rbp-20h]

  sub_BAE5F0((__int64)v5, a1, a3, a4);
  sub_95D570(v4, "SmallVector capacity unable to grow. Already at maximum size ", (__int64)v5);
  sub_2240A30(v5);
  v5[0] = v4;
  v6 = 260;
  sub_C64D30((__int64)v5, 1u);
}
