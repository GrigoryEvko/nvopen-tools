// Function: sub_273AC00
// Address: 0x273ac00
//
void __fastcall sub_273AC00(__m128i *src, const __m128i *a2, __m128i *a3)
{
  __int64 v5; // r14
  __m128i *v6; // r15
  __m128i *v7; // rdi
  __int64 v8; // r15
  __int64 v9; // rcx
  __int64 v10; // [rsp+0h] [rbp-40h]
  __int64 v11; // [rsp+8h] [rbp-38h]

  v5 = (__int64)a3->m128i_i64 + (char *)a2 - (char *)src;
  v10 = (char *)a2 - (char *)src;
  v11 = ((char *)a2 - (char *)src) >> 6;
  if ( (char *)a2 - (char *)src <= 384 )
  {
    sub_273A870(src, a2);
  }
  else
  {
    v6 = src;
    do
    {
      v7 = v6;
      v6 += 28;
      sub_273A870(v7, v6);
    }
    while ( (char *)a2 - (char *)v6 > 384 );
    sub_273A870(v6, a2);
    if ( v10 > 448 )
    {
      v8 = 7;
      do
      {
        sub_273A5D0(src, (__int64)a2, a3, v8);
        v9 = 2 * v8;
        v8 *= 4;
        sub_273A5D0(a3, v5, src, v9);
      }
      while ( v11 > v8 );
    }
  }
}
