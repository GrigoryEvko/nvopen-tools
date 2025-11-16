// Function: sub_1A003F0
// Address: 0x1a003f0
//
void __fastcall sub_1A003F0(__m128i *src, const __m128i *a2)
{
  __int64 v2; // rcx
  __m128i *v3; // r14
  __int64 v4; // rbx

  if ( (char *)a2 - (char *)src <= 224 )
  {
    sub_19FE890(src, a2);
  }
  else
  {
    v2 = ((char *)a2 - (char *)src) >> 5;
    v3 = &src[v2];
    v4 = (16 * v2) >> 4;
    sub_1A003F0(src);
    sub_1A003F0(v3);
    sub_1A00290(src, v3, (__int64)a2, v4, a2 - v3);
  }
}
