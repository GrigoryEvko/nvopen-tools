// Function: sub_3986980
// Address: 0x3986980
//
void __fastcall sub_3986980(__int64 *src, __int64 *a2, __int64 a3)
{
  __int64 v4; // rcx
  __m128i *v5; // r15
  __int64 v6; // rbx

  if ( (char *)a2 - (char *)src <= 224 )
  {
    sub_3986390(src, a2, a3);
  }
  else
  {
    v4 = ((char *)a2 - (char *)src) >> 5;
    v5 = (__m128i *)&src[2 * v4];
    v6 = (16 * v4) >> 4;
    sub_3986980(src);
    sub_3986980(v5);
    sub_39867E0(src, v5, (__int64)a2, v6, ((char *)a2 - (char *)v5) >> 4, a3);
  }
}
