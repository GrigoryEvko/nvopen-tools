// Function: sub_15803F0
// Address: 0x15803f0
//
__m128i *__fastcall sub_15803F0(__m128i *a1)
{
  _BYTE *v1; // r8
  _BYTE v3[16]; // [rsp+0h] [rbp-60h] BYREF
  __int64 (__fastcall *v4)(_QWORD *, __int64, int); // [rsp+10h] [rbp-50h]
  bool (__fastcall *v5)(__int64, __int64); // [rsp+18h] [rbp-48h]
  _QWORD v6[2]; // [rsp+20h] [rbp-40h] BYREF
  void (__fastcall *v7)(_QWORD *, _QWORD *, __int64); // [rsp+30h] [rbp-30h]
  bool (__fastcall *v8)(__int64, __int64); // [rsp+38h] [rbp-28h]

  v5 = sub_157E950;
  v4 = sub_157E930;
  v7 = 0;
  sub_157E930(v6, (__int64)v3, 2);
  v8 = v5;
  v7 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))v4;
  sub_157FF80(a1, v1, (__int64)v6);
  if ( v7 )
    v7(v6, v6, 3);
  if ( v4 )
    v4(v3, (__int64)v3, 3);
  return a1;
}
