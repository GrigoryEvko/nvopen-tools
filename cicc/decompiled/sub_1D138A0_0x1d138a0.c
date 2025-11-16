// Function: sub_1D138A0
// Address: 0x1d138a0
//
void __fastcall sub_1D138A0(char *src, char *a2)
{
  char *v2; // rbx
  char *v3; // r13
  __int32 v4; // ecx
  __int64 v5; // r12
  __m128i *v6; // rdx
  __int64 v7; // r15
  const __m128i *v8; // rax
  __m128i v9; // xmm0
  int v10; // [rsp-3Ch] [rbp-3Ch]

  if ( src != a2 )
  {
    v2 = src + 24;
    if ( a2 != src + 24 )
    {
      v3 = &src[((a2 - 48 - src) & 0xFFFFFFFFFFFFFFF8LL) + 48];
      do
      {
        while ( 1 )
        {
          v5 = *(_QWORD *)v2;
          v4 = *((_DWORD *)v2 + 2);
          v6 = (__m128i *)v2;
          v7 = *((_QWORD *)v2 + 2);
          if ( *(_QWORD *)src > *(_QWORD *)v2 )
            break;
          v8 = (const __m128i *)(v2 - 24);
          if ( *((_QWORD *)v2 - 3) > v5 )
          {
            do
            {
              v9 = _mm_loadu_si128(v8);
              v8[2].m128i_i64[1] = v8[1].m128i_i64[0];
              v6 = (__m128i *)v8;
              v8 = (const __m128i *)((char *)v8 - 24);
              v8[3] = v9;
            }
            while ( v8->m128i_i64[0] > v5 );
          }
          v2 += 24;
          v6->m128i_i64[0] = v5;
          v6->m128i_i32[2] = v4;
          v6[1].m128i_i64[0] = v7;
          if ( v2 == v3 )
            return;
        }
        if ( src != v2 )
        {
          v10 = *((_DWORD *)v2 + 2);
          memmove(src + 24, src, v2 - src);
          v4 = v10;
        }
        v2 += 24;
        *(_QWORD *)src = v5;
        *((_DWORD *)src + 2) = v4;
        *((_QWORD *)src + 2) = v7;
      }
      while ( v2 != v3 );
    }
  }
}
