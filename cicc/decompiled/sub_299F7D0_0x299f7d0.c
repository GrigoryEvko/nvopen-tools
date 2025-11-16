// Function: sub_299F7D0
// Address: 0x299f7d0
//
__int64 __fastcall sub_299F7D0(
        __m128i *src,
        __m128i *a2,
        __m128i *a3,
        __int64 a4,
        unsigned __int8 (__fastcall *a5)(__m128i *, __int8 *))
{
  __int64 v7; // rax
  __m128i *v8; // rbx
  unsigned __int8 (__fastcall *v9)(__int64, __int64); // r9
  __int64 v10; // r10
  __int64 v12; // [rsp-10h] [rbp-50h]
  __int64 v13; // [rsp+0h] [rbp-40h]

  v7 = (0x6DB6DB6DB6DB6DB7LL * (((char *)a2 - (char *)src) >> 3) + 1) / 2;
  v13 = 56 * v7;
  v8 = (__m128i *)((char *)src + 56 * v7);
  if ( v7 <= a4 )
  {
    sub_299F6C0(src, (__m128i *)((char *)src + 56 * v7), a3, a5);
    sub_299F6C0(v8, a2, a3, a5);
    v10 = v13;
    v9 = (unsigned __int8 (__fastcall *)(__int64, __int64))a5;
  }
  else
  {
    sub_299F7D0(src);
    sub_299F7D0(v8);
    v9 = (unsigned __int8 (__fastcall *)(__int64, __int64))a5;
    v10 = v13;
  }
  sub_299F060(
    src,
    v8,
    a2,
    0x6DB6DB6DB6DB6DB7LL * (v10 >> 3),
    0x6DB6DB6DB6DB6DB7LL * (((char *)a2 - (char *)v8) >> 3),
    a3,
    a4,
    v9);
  return v12;
}
