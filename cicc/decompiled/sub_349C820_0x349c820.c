// Function: sub_349C820
// Address: 0x349c820
//
__int64 __fastcall sub_349C820(const __m128i *a1, const __m128i *a2, const __m128i *a3, __int64 a4)
{
  __int64 v6; // rax
  unsigned __int64 v7; // r9
  const __m128i *v8; // rbx
  __int64 v10; // [rsp-10h] [rbp-50h]
  __int64 v11; // [rsp+8h] [rbp-38h]

  v6 = (__int64)(0xAAAAAAAAAAAAAAABLL * (((char *)a2 - (char *)a1) >> 3) + 1) / 2;
  v7 = 8
     * (v6
      + ((0xAAAAAAAAAAAAAAABLL * (((char *)a2 - (char *)a1) >> 3)
        + 1
        + ((0xAAAAAAAAAAAAAAABLL * (((char *)a2 - (char *)a1) >> 3) + 1) >> 63))
       & 0xFFFFFFFFFFFFFFFELL));
  v11 = v7;
  v8 = (const __m128i *)((char *)a1 + v7);
  if ( v6 <= a4 )
  {
    sub_3440540(a1, (const __m128i *)((char *)a1 + v7), a3);
    sub_3440540(v8, a2, a3);
  }
  else
  {
    sub_349C820(a1, &a1->m128i_i8[v7]);
    sub_349C820(v8, a2);
  }
  sub_349C380(
    (__int64)a1,
    v8,
    (__int64)a2,
    0xAAAAAAAAAAAAAAABLL * (v11 >> 3),
    0xAAAAAAAAAAAAAAABLL * (((char *)a2 - (char *)v8) >> 3),
    a3,
    a4);
  return v10;
}
