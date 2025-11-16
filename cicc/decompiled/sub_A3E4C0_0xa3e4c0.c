// Function: sub_A3E4C0
// Address: 0xa3e4c0
//
void __fastcall sub_A3E4C0(__m128i *a1, __m128i *a2, __int64 a3)
{
  __int64 v4; // rcx
  char *m128i_i8; // r15
  __int64 v6; // rbx

  if ( (char *)a2 - (char *)a1 <= 224 )
  {
    sub_A3D360(a1, a2, a3);
  }
  else
  {
    v4 = ((char *)a2 - (char *)a1) >> 5;
    m128i_i8 = a1[v4].m128i_i8;
    v6 = (16 * v4) >> 4;
    sub_A3E4C0(a1, m128i_i8);
    sub_A3E4C0(m128i_i8, a2);
    sub_A3E340(a1, m128i_i8, (__int64)a2, v6, ((char *)a2 - m128i_i8) >> 4, a3);
  }
}
