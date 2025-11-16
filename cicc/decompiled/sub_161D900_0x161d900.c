// Function: sub_161D900
// Address: 0x161d900
//
void __fastcall sub_161D900(__m128i *a1, __m128i *a2)
{
  __m128i *v2; // r8
  char *v3; // r9
  __m128i *v4; // r11
  unsigned __int64 v5; // r10
  unsigned __int64 v6; // rcx
  __m128i *v7; // rdi
  __int64 v8; // rsi
  __int64 v9; // rax
  __int64 v10; // rdx
  __m128i v11; // [rsp-28h] [rbp-28h]

  if ( a1 != a2 )
  {
    v2 = (__m128i *)((char *)a1 + 24);
    v3 = (char *)a1;
    v4 = a2;
    if ( a2 != (__m128i *)&a1[1].m128i_u64[1] )
    {
      v5 = 0xAAAAAAAAAAAAAAABLL;
      do
      {
        v6 = v2[1].m128i_u64[0];
        v7 = v2;
        v2 = (__m128i *)((char *)v2 + 24);
        if ( v6 >= *((_QWORD *)v3 + 2) )
        {
          sub_161CBB0(v7);
        }
        else
        {
          v8 = v2[-2].m128i_i64[1];
          v11 = _mm_loadu_si128((__m128i *)((char *)v2 - 24));
          v9 = v5 * (((char *)v7 - v3) >> 3);
          if ( (char *)v7 - v3 > 0 )
          {
            do
            {
              v10 = v7[-2].m128i_i64[1];
              v7 = (__m128i *)((char *)v7 - 24);
              v7[1].m128i_i64[1] = v10;
              v7[2].m128i_i64[0] = v7->m128i_i64[1];
              v7[2].m128i_i64[1] = v7[1].m128i_i64[0];
              --v9;
            }
            while ( v9 );
          }
          *(_QWORD *)v3 = v8;
          *((_QWORD *)v3 + 2) = v6;
          *((_QWORD *)v3 + 1) = v11.m128i_i64[1];
        }
      }
      while ( v4 != v2 );
    }
  }
}
