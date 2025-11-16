// Function: sub_1070BF0
// Address: 0x1070bf0
//
void __fastcall sub_1070BF0(__m128i *a1, const __m128i *a2)
{
  const __m128i *v2; // rbx
  const __m128i *v3; // rdi

  if ( (char *)a2 - (char *)a1 <= 384 )
  {
    sub_1070B30(a1->m128i_i8, a2);
  }
  else
  {
    v2 = a1 + 24;
    sub_1070B30(a1->m128i_i8, a1 + 24);
    if ( a2 != &a1[24] )
    {
      do
      {
        v3 = v2;
        v2 = (const __m128i *)((char *)v2 + 24);
        sub_1070780(v3);
      }
      while ( a2 != v2 );
    }
  }
}
