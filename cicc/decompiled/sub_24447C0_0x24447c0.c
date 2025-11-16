// Function: sub_24447C0
// Address: 0x24447c0
//
void __fastcall sub_24447C0(__int64 *src, __int64 *a2)
{
  __int64 v2; // rcx
  __m128i *v3; // r14
  __int64 v4; // rbx

  if ( (char *)a2 - (char *)src <= 224 )
  {
    sub_2443FE0(src, a2);
  }
  else
  {
    v2 = ((char *)a2 - (char *)src) >> 5;
    v3 = (__m128i *)&src[2 * v2];
    v4 = (16 * v2) >> 4;
    sub_24447C0(src);
    sub_24447C0(v3);
    sub_2444670(src, v3, (__int64)a2, v4, ((char *)a2 - (char *)v3) >> 4);
  }
}
