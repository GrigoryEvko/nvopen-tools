// Function: sub_AA69B0
// Address: 0xaa69b0
//
__m128i *__fastcall sub_AA69B0(__m128i *a1, __int64 a2, char a3)
{
  __int64 v3; // r8
  _BYTE v5[16]; // [rsp+0h] [rbp-60h] BYREF
  __int64 (__fastcall *v6)(_QWORD *, _BYTE *, int); // [rsp+10h] [rbp-50h]
  bool (__fastcall *v7)(_BYTE *, __int64); // [rsp+18h] [rbp-48h]
  _QWORD v8[2]; // [rsp+20h] [rbp-40h] BYREF
  void (__fastcall *v9)(_QWORD *, _QWORD *, __int64); // [rsp+30h] [rbp-30h]
  bool (__fastcall *v10)(_BYTE *, __int64); // [rsp+38h] [rbp-28h]

  v5[0] = a3;
  v7 = sub_AA4410;
  v6 = sub_AA4320;
  v9 = 0;
  sub_AA4320(v8, v5, 2);
  v10 = v7;
  v9 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))v6;
  sub_AA64E0(a1, v3, (__int64)v8);
  if ( v9 )
    v9(v8, v8, 3);
  if ( v6 )
    v6(v5, v5, 3);
  return a1;
}
