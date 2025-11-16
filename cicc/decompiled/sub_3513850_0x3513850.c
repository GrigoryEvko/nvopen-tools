// Function: sub_3513850
// Address: 0x3513850
//
__int64 __fastcall sub_3513850(__m128i *src, __m128i *a2, __m128i *a3, const __m128i *a4)
{
  __int64 v6; // rax
  unsigned __int64 v7; // r9
  __m128i *v8; // rbx
  __int64 v10; // [rsp-10h] [rbp-50h]
  __int64 v11; // [rsp+8h] [rbp-38h]

  v6 = (__int64)(0xAAAAAAAAAAAAAAABLL * (((char *)a2 - (char *)src) >> 3) + 1) / 2;
  v7 = 8
     * (v6
      + ((0xAAAAAAAAAAAAAAABLL * (((char *)a2 - (char *)src) >> 3)
        + 1
        + ((0xAAAAAAAAAAAAAAABLL * (((char *)a2 - (char *)src) >> 3) + 1) >> 63))
       & 0xFFFFFFFFFFFFFFFELL));
  v11 = v7;
  v8 = (__m128i *)((char *)src + v7);
  if ( v6 <= (__int64)a4 )
  {
    sub_3510D80(src, (__m128i *)((char *)src + v7), a3);
    sub_3510D80(v8, a2, a3);
  }
  else
  {
    sub_3513850(src);
    sub_3513850(v8);
  }
  sub_35133A0(
    src,
    v8,
    (__int64)a2,
    0xAAAAAAAAAAAAAAABLL * (v11 >> 3),
    0xAAAAAAAAAAAAAAABLL * (((char *)a2 - (char *)v8) >> 3),
    a3,
    a4);
  return v10;
}
