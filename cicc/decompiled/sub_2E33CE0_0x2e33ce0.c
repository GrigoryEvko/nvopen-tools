// Function: sub_2E33CE0
// Address: 0x2e33ce0
//
const __m128i *__fastcall sub_2E33CE0(__int64 a1, unsigned int a2, __int64 a3, __int64 a4)
{
  const __m128i *v4; // rax
  __int8 *v5; // rdx
  __int64 v6; // rcx
  unsigned __int64 v7; // rdx
  const __m128i *v8; // rax
  __int32 v9; // ecx
  __m128i v10; // xmm0
  const __m128i *v11; // r9
  const __m128i *result; // rax
  const __m128i *v14; // rsi
  signed __int64 v15; // rdx
  const __m128i *v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // rdi

  v11 = *(const __m128i **)(a1 + 192);
  result = (const __m128i *)a2;
  v14 = *(const __m128i **)(a1 + 184);
  v15 = 0xAAAAAAAAAAAAAAABLL * (((char *)v11 - (char *)v14) >> 3);
  if ( v15 >> 2 > 0 )
  {
    v16 = &v14[6 * (v15 >> 2)];
    while ( (_DWORD)result != v14->m128i_i32[0] )
    {
      if ( (_DWORD)result == v14[1].m128i_i32[2] )
      {
        v14 = (const __m128i *)((char *)v14 + 24);
        goto LABEL_8;
      }
      if ( (_DWORD)result == v14[3].m128i_i32[0] )
      {
        v14 += 3;
        goto LABEL_8;
      }
      if ( (_DWORD)result == v14[4].m128i_i32[2] )
      {
        v14 = (const __m128i *)((char *)v14 + 72);
        goto LABEL_8;
      }
      v14 += 6;
      if ( v14 == v16 )
      {
        v15 = 0xAAAAAAAAAAAAAAABLL * (((char *)v11 - (char *)v14) >> 3);
        goto LABEL_12;
      }
    }
    goto LABEL_8;
  }
LABEL_12:
  if ( v15 == 2 )
  {
LABEL_18:
    if ( (_DWORD)result != v14->m128i_i32[0] )
    {
      v14 = (const __m128i *)((char *)v14 + 24);
      goto LABEL_20;
    }
    goto LABEL_8;
  }
  if ( v15 != 3 )
  {
    if ( v15 != 1 )
      return result;
LABEL_20:
    if ( (_DWORD)result != v14->m128i_i32[0] )
      return result;
    goto LABEL_8;
  }
  if ( (_DWORD)result != v14->m128i_i32[0] )
  {
    v14 = (const __m128i *)((char *)v14 + 24);
    goto LABEL_18;
  }
LABEL_8:
  if ( v11 != v14 )
  {
    v17 = v14[1].m128i_i64[0] & ~a4;
    v18 = v14->m128i_i64[1] & ~a3;
    v14->m128i_i64[1] = v18;
    v14[1].m128i_i64[0] = v17;
    if ( !(v17 | v18) )
    {
      v19 = a1 + 184;
      v4 = *(const __m128i **)(v19 + 8);
      v5 = &v14[1].m128i_i8[8];
      if ( &v14[1].m128i_u64[1] != (unsigned __int64 *)v4 )
      {
        v6 = (char *)v4 - v5;
        v7 = 0xAAAAAAAAAAAAAAABLL * (((char *)v4 - v5) >> 3);
        if ( v6 > 0 )
        {
          v8 = v14;
          do
          {
            v9 = v8[1].m128i_i32[2];
            v10 = _mm_loadu_si128(v8 + 2);
            v8 = (const __m128i *)((char *)v8 + 24);
            v8[-2].m128i_i32[2] = v9;
            v8[-1] = v10;
            --v7;
          }
          while ( v7 );
          v4 = *(const __m128i **)(v19 + 8);
        }
      }
      *(_QWORD *)(v19 + 8) = (char *)v4 - 24;
      return v14;
    }
  }
  return result;
}
