// Function: sub_149DCE0
// Address: 0x149dce0
//
void __fastcall sub_149DCE0(
        __m128i *a1,
        __m128i *a2,
        unsigned __int8 (__fastcall *a3)(__m128i *, __int8 *),
        __int64 a4,
        __int64 a5,
        __int64 a6)
{
  __m128i *v7; // rbx
  const __m128i *v8; // rdi

  if ( (char *)a2 - (char *)a1 <= 640 )
  {
    sub_149DC00(a1, a2, a3, a4, a5, a6);
  }
  else
  {
    v7 = a1 + 40;
    sub_149DC00(a1, a1 + 40, a3, a4, a5, a6);
    if ( a2 != &a1[40] )
    {
      do
      {
        v8 = v7;
        v7 = (__m128i *)((char *)v7 + 40);
        sub_149DB70(v8, a3);
      }
      while ( a2 != v7 );
    }
  }
}
