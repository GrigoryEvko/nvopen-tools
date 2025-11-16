// Function: sub_1A00210
// Address: 0x1a00210
//
void __fastcall sub_1A00210(__m128i *src, __m128i *a2)
{
  __int64 v2; // rcx
  __m128i *v3; // r14
  __int64 v4; // rbx

  if ( (char *)a2 - (char *)src <= 224 )
  {
    sub_19FEDA0(src, a2);
  }
  else
  {
    v2 = ((char *)a2 - (char *)src) >> 5;
    v3 = &src[v2];
    v4 = (16 * v2) >> 4;
    sub_1A00210(src);
    sub_1A00210(v3);
    sub_1A000C0(src, v3, (__int64)a2, v4, a2 - v3);
  }
}
