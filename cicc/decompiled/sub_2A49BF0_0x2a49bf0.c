// Function: sub_2A49BF0
// Address: 0x2a49bf0
//
__int64 __fastcall sub_2A49BF0(__m128i *src, const __m128i *a2, __m128i *a3, __int64 a4, __int64 a5)
{
  __int64 v7; // rdx
  signed __int64 v8; // rbx
  const __m128i *v9; // r10
  __int64 v10; // rax
  __int64 v12; // [rsp-10h] [rbp-50h]

  v7 = (__int64)(0xAAAAAAAAAAAAAAABLL * (a2 - src) + 1) / 2;
  v8 = v7
     + ((0xAAAAAAAAAAAAAAABLL * (a2 - src) + 1 + ((0xAAAAAAAAAAAAAAABLL * (a2 - src) + 1) >> 63)) & 0xFFFFFFFFFFFFFFFELL);
  if ( v7 <= a4 )
  {
    sub_2A49AE0(src, &src[v8], a3, a5);
    sub_2A49AE0(&src[v8], a2, a3, a5);
    v10 = a5;
    v9 = &src[v8];
  }
  else
  {
    sub_2A49BF0(src);
    sub_2A49BF0(&src[v8]);
    v9 = &src[v8];
    v10 = a5;
  }
  sub_2A47900(
    src,
    v9,
    (__int64)a2,
    0xAAAAAAAAAAAAAAABLL * ((v8 * 16) >> 4),
    0xAAAAAAAAAAAAAAABLL * (a2 - v9),
    a3,
    a4,
    v10);
  return v12;
}
