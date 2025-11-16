// Function: sub_3988A40
// Address: 0x3988a40
//
void __fastcall sub_3988A40(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rax
  const __m128i *v8; // rbx
  const __m128i *v9; // r13
  __m128i *v10; // rax
  __int64 v11; // rdi
  __m128i *v12; // rsi
  __int64 v13; // rcx
  __int32 v14; // ecx
  __m128i *v15; // rdx
  __m128i v16; // xmm0
  __int32 v17; // edx
  _BYTE v18[64]; // [rsp+0h] [rbp-40h] BYREF

  if ( !*(_DWORD *)(a1 + 48)
    || (v7 = *(_QWORD *)(*(_QWORD *)(a1 + 40) + 16LL * *(unsigned int *)(a1 + 48) - 8)) != 0
    && (sub_15B1350((__int64)v18, *(unsigned __int64 **)(v7 + 24), *(unsigned __int64 **)(v7 + 32)), v18[16]) )
  {
    v8 = *(const __m128i **)(a2 + 40);
    v9 = &v8[*(unsigned int *)(a2 + 48)];
    if ( v9 != v8 )
    {
      while ( 1 )
      {
        while ( 1 )
        {
          v10 = *(__m128i **)(a1 + 40);
          v11 = *(unsigned int *)(a1 + 48);
          v12 = &v10[v11];
          v13 = (16 * v11) >> 4;
          if ( !((16 * v11) >> 6) )
            break;
          v14 = v8->m128i_i32[0];
          v15 = &v10[4 * ((16 * v11) >> 6)];
          while ( 1 )
          {
            if ( v14 == v10->m128i_i32[0] )
            {
              a5 = v10->m128i_i64[1];
              if ( v8->m128i_i64[1] == a5 )
                break;
            }
            if ( v14 == v10[1].m128i_i32[0] )
            {
              a6 = v10[1].m128i_i64[1];
              if ( v8->m128i_i64[1] == a6 )
              {
                ++v10;
                break;
              }
            }
            if ( v14 == v10[2].m128i_i32[0] && v8->m128i_i64[1] == v10[2].m128i_i64[1] )
            {
              v10 += 2;
              break;
            }
            if ( v14 == v10[3].m128i_i32[0] && v8->m128i_i64[1] == v10[3].m128i_i64[1] )
            {
              v10 += 3;
              break;
            }
            v10 += 4;
            if ( v15 == v10 )
            {
              v13 = v12 - v10;
              goto LABEL_13;
            }
          }
LABEL_21:
          if ( v12 == v10 )
            goto LABEL_16;
          if ( v9 == ++v8 )
            return;
        }
LABEL_13:
        if ( v13 == 2 )
          break;
        if ( v13 == 3 )
        {
          v17 = v8->m128i_i32[0];
          if ( v8->m128i_i32[0] == v10->m128i_i32[0] && v8->m128i_i64[1] == v10->m128i_i64[1] )
            goto LABEL_21;
          ++v10;
          goto LABEL_36;
        }
        if ( v13 != 1 )
          goto LABEL_16;
        v17 = v8->m128i_i32[0];
LABEL_31:
        if ( v10->m128i_i32[0] == v17 && v8->m128i_i64[1] == v10->m128i_i64[1] )
          goto LABEL_21;
LABEL_16:
        if ( (unsigned int)v11 >= *(_DWORD *)(a1 + 52) )
        {
          sub_16CD150(a1 + 40, (const void *)(a1 + 56), 0, 16, a5, a6);
          v12 = (__m128i *)(*(_QWORD *)(a1 + 40) + 16LL * *(unsigned int *)(a1 + 48));
        }
        v16 = _mm_loadu_si128(v8++);
        *v12 = v16;
        ++*(_DWORD *)(a1 + 48);
        if ( v9 == v8 )
          return;
      }
      v17 = v8->m128i_i32[0];
LABEL_36:
      if ( v17 == v10->m128i_i32[0] && v8->m128i_i64[1] == v10->m128i_i64[1] )
        goto LABEL_21;
      ++v10;
      goto LABEL_31;
    }
  }
}
