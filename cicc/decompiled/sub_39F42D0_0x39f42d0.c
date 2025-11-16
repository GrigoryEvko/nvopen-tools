// Function: sub_39F42D0
// Address: 0x39f42d0
//
__int64 *__fastcall sub_39F42D0(__int64 a1)
{
  __m128i *v1; // rax
  __int64 *v2; // r12

  v1 = (__m128i *)sub_22077B0(0x108u);
  v2 = (__int64 *)v1;
  if ( v1 )
  {
    sub_38DCAE0(v1, a1);
    *v2 = (__int64)off_4A413F0;
  }
  return v2;
}
