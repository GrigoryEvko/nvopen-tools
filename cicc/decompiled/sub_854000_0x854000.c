// Function: sub_854000
// Address: 0x854000
//
_QWORD *__fastcall sub_854000(__m128i *a1)
{
  __m128i *v1; // rbx
  __m128i *v2; // rdi
  _QWORD *result; // rax

  if ( a1 )
  {
    v1 = a1;
    do
    {
      v2 = v1;
      v1 = (__m128i *)v1->m128i_i64[0];
      result = sub_853F90(v2);
    }
    while ( v1 );
  }
  return result;
}
