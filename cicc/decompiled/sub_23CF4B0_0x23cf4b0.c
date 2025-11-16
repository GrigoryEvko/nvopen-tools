// Function: sub_23CF4B0
// Address: 0x23cf4b0
//
__m128i *__fastcall sub_23CF4B0(__m128i *a1, __int64 a2)
{
  __m128i v3; // [rsp+0h] [rbp-30h] BYREF
  __int64 (__fastcall *v4)(__m128i *, __m128i *, int); // [rsp+10h] [rbp-20h]
  _QWORD *(__fastcall *v5)(_QWORD *, __int64 *, __int64); // [rsp+18h] [rbp-18h]

  v3.m128i_i64[0] = a2;
  v5 = sub_23CE590;
  v4 = (__int64 (__fastcall *)(__m128i *, __m128i *, int))sub_23CE440;
  sub_DFE9B0(a1, &v3);
  if ( v4 )
    v4(&v3, &v3, 3);
  return a1;
}
