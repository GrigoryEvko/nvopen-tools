// Function: sub_273AB80
// Address: 0x273ab80
//
void __fastcall sub_273AB80(__m128i *src, const __m128i *a2)
{
  __int64 v2; // rcx
  __m128i *v3; // r14
  __int64 v4; // rbx

  if ( (char *)a2 - (char *)src <= 896 )
  {
    sub_273A870(src, a2);
  }
  else
  {
    v2 = ((char *)a2 - (char *)src) >> 7;
    v3 = &src[4 * v2];
    v4 = v2 << 6 >> 6;
    sub_273AB80(src);
    sub_273AB80(v3);
    sub_273A0C0(src, (__int64)v3, (__int64)a2, v4, ((char *)a2 - (char *)v3) >> 6);
  }
}
