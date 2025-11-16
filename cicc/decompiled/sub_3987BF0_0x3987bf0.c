// Function: sub_3987BF0
// Address: 0x3987bf0
//
__int64 __fastcall sub_3987BF0(__m128i *src, __m128i *a2, __m128i *a3, __int64 a4, void *a5)
{
  __int64 v8; // r9
  __m128i *v9; // r10
  __int64 v10; // r9
  __int64 v12; // [rsp-10h] [rbp-50h]
  __m128i *srca; // [rsp+8h] [rbp-38h]

  v8 = (a2 - src + 1) / 2;
  srca = &src[v8];
  if ( (a2 - src + 1) / 2 <= a4 )
  {
    sub_3986F10(src, &src[v8], a3, (__int64)a5);
    sub_3986F10(srca, a2, a3, (__int64)a5);
    v10 = 16 * ((a2 - src + 1) / 2);
    v9 = srca;
  }
  else
  {
    sub_3987BF0(src);
    sub_3987BF0(srca);
    v9 = srca;
    v10 = 16 * ((a2 - src + 1) / 2);
  }
  sub_3987760(src, v9, (__int64)a2, v10 >> 4, a2 - v9, a3, a4, a5);
  return v12;
}
