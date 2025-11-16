// Function: sub_2A69D20
// Address: 0x2a69d20
//
void __fastcall sub_2A69D20(__int64 a1, __m128i *a2)
{
  const __m128i *v2; // rax
  __int64 v3; // rcx
  __m128i v4; // xmm0
  __int8 v5; // dl
  __int32 v6; // edx
  const __m128i *v7; // r12
  __int64 v8; // rbx
  int v9; // eax
  unsigned __int64 v10; // rdi
  unsigned __int64 v11; // rdi

  v2 = *(const __m128i **)a1;
  v3 = *(_QWORD *)a1 + 56LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v3 )
  {
    do
    {
      if ( a2 )
      {
        v4 = _mm_loadu_si128(v2);
        a2[1].m128i_i8[1] = 0;
        *a2 = v4;
        v5 = v2[1].m128i_i8[0];
        a2[1].m128i_i8[0] = v5;
        if ( (unsigned __int8)v5 <= 3u )
        {
          if ( (unsigned __int8)v5 > 1u )
            a2[1].m128i_i64[1] = v2[1].m128i_i64[1];
        }
        else if ( (unsigned __int8)(v5 - 4) <= 1u )
        {
          a2[2].m128i_i32[0] = v2[2].m128i_i32[0];
          a2[1].m128i_i64[1] = v2[1].m128i_i64[1];
          v6 = v2[3].m128i_i32[0];
          v2[2].m128i_i32[0] = 0;
          a2[3].m128i_i32[0] = v6;
          a2[2].m128i_i64[1] = v2[2].m128i_i64[1];
          LOBYTE(v6) = v2[1].m128i_i8[1];
          v2[3].m128i_i32[0] = 0;
          a2[1].m128i_i8[1] = v6;
        }
        v2[1].m128i_i8[0] = 0;
      }
      v2 = (const __m128i *)((char *)v2 + 56);
      a2 = (__m128i *)((char *)a2 + 56);
    }
    while ( (const __m128i *)v3 != v2 );
    v7 = *(const __m128i **)a1;
    v8 = *(_QWORD *)a1 + 56LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v8 )
    {
      do
      {
        while ( 1 )
        {
          v9 = *(unsigned __int8 *)(v8 - 40);
          v8 -= 56;
          if ( (unsigned int)(v9 - 4) <= 1 )
          {
            if ( *(_DWORD *)(v8 + 48) > 0x40u )
            {
              v10 = *(_QWORD *)(v8 + 40);
              if ( v10 )
                j_j___libc_free_0_0(v10);
            }
            if ( *(_DWORD *)(v8 + 32) > 0x40u )
            {
              v11 = *(_QWORD *)(v8 + 24);
              if ( v11 )
                break;
            }
          }
          if ( (const __m128i *)v8 == v7 )
            return;
        }
        j_j___libc_free_0_0(v11);
      }
      while ( (const __m128i *)v8 != v7 );
    }
  }
}
