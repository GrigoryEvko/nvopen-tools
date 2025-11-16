// Function: sub_1700880
// Address: 0x1700880
//
__m128i *__fastcall sub_1700880(__m128i *a1, __int64 a2)
{
  __m128i v3; // [rsp+0h] [rbp-30h] BYREF
  __int64 (__fastcall *v4)(__m128i *, __m128i *, int); // [rsp+10h] [rbp-20h]
  _QWORD *(__fastcall *v5)(_QWORD *, __int64, __int64); // [rsp+18h] [rbp-18h]

  v3.m128i_i64[0] = a2;
  v5 = sub_16FF990;
  v4 = (__int64 (__fastcall *)(__m128i *, __m128i *, int))sub_16FF800;
  sub_14A3D00(a1, &v3);
  if ( v4 )
    v4(&v3, &v3, 3);
  return a1;
}
