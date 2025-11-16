// Function: sub_32185D0
// Address: 0x32185d0
//
void __fastcall sub_32185D0(__m128i *a1, __m128i *a2)
{
  __int64 v2; // rbp
  __int64 *v3; // r8
  __m128i *v4; // r9
  __m128i *v5; // r11
  unsigned __int64 v6; // r10
  __int64 v7; // rcx
  __int64 *v8; // rdi
  __m128i v9; // xmm1
  __int64 v10; // rax
  __m128i v11; // xmm0
  __int64 v12; // rdx
  __m128i v13; // xmm2
  __m128i v14; // [rsp-28h] [rbp-28h] BYREF
  __int64 v15; // [rsp-18h] [rbp-18h]
  __int64 v16; // [rsp-8h] [rbp-8h]

  if ( a1 != a2 )
  {
    v3 = &a1[1].m128i_i64[1];
    v4 = a1;
    v5 = a2;
    if ( a2 != (__m128i *)&a1[1].m128i_u64[1] )
    {
      v6 = 0xAAAAAAAAAAAAAAABLL;
      v16 = v2;
      do
      {
        v7 = v3[2];
        v8 = v3;
        v3 += 3;
        if ( *(_DWORD *)(v7 + 16) >= *(_DWORD *)(v4[1].m128i_i64[0] + 16) )
        {
          sub_32184B0((__m128i *)v8);
        }
        else
        {
          v9 = _mm_loadu_si128((const __m128i *)(v3 - 3));
          v15 = *(v3 - 1);
          v14 = v9;
          v10 = v6 * (((char *)v8 - (char *)v4) >> 3);
          if ( (char *)v8 - (char *)v4 > 0 )
          {
            do
            {
              v11 = _mm_loadu_si128((const __m128i *)(v8 - 3));
              v12 = *(v8 - 1);
              v8 -= 3;
              *(__m128i *)(v8 + 3) = v11;
              v8[5] = v12;
              --v10;
            }
            while ( v10 );
          }
          v13 = _mm_loadu_si128(&v14);
          v4[1].m128i_i64[0] = v7;
          *v4 = v13;
        }
      }
      while ( v5 != (__m128i *)v3 );
    }
  }
}
