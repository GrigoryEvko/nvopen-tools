// Function: sub_2D3F240
// Address: 0x2d3f240
//
__int64 __fastcall sub_2D3F240(__int64 a1, __m128i *a2, __m128i *a3)
{
  __m128i *v4; // r13
  bool v6; // zf
  __int64 result; // rax
  __int64 v8; // rdx
  __int64 i; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  int v12; // esi
  __int64 v13; // rdi
  int v14; // r10d
  __m128i *v15; // r9
  unsigned int j; // eax
  __m128i *v17; // rdx
  __int64 v18; // r11
  int v19; // esi
  __int64 v20; // rax
  __m128i *v21; // rdi
  __m128i *v22; // rax
  __m128i *v23; // rdx
  __int64 v24; // rsi
  __int64 v25; // rdi
  unsigned int v26; // eax

  v4 = a2;
  v6 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  if ( v6 )
  {
    result = *(_QWORD *)(a1 + 16);
    v8 = 96LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    result = a1 + 16;
    v8 = 384;
  }
  for ( i = result + v8; i != result; result += 96 )
  {
    if ( result )
    {
      *(_QWORD *)result = -4096;
      *(_QWORD *)(result + 8) = -4096;
    }
  }
  if ( a2 != a3 )
  {
    do
    {
      while ( 1 )
      {
        v10 = v4->m128i_i64[0];
        if ( v4->m128i_i64[0] != -4096 )
          break;
        if ( v4->m128i_i64[1] == -4096 )
        {
LABEL_23:
          v4 += 6;
          if ( a3 == v4 )
            return result;
        }
        else
        {
LABEL_10:
          if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
          {
            v11 = a1 + 16;
            v12 = 3;
          }
          else
          {
            v19 = *(_DWORD *)(a1 + 24);
            v11 = *(_QWORD *)(a1 + 16);
            if ( !v19 )
            {
              MEMORY[0] = v4->m128i_i64[0];
              BUG();
            }
            v12 = v19 - 1;
          }
          v13 = v4->m128i_i64[1];
          v14 = 1;
          v15 = 0;
          for ( j = v12
                  & (((0xBF58476D1CE4E5B9LL
                     * (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4)
                      | ((unsigned __int64)(((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4)) << 32))) >> 31)
                   ^ (484763065 * (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4)))); ; j = v12 & v26 )
          {
            v17 = (__m128i *)(v11 + 96LL * j);
            v18 = v17->m128i_i64[0];
            if ( v10 == v17->m128i_i64[0] && v17->m128i_i64[1] == v13 )
              break;
            if ( v18 == -4096 )
            {
              if ( v17->m128i_i64[1] == -4096 )
              {
                if ( v15 )
                  v17 = v15;
                break;
              }
            }
            else if ( v18 == -8192 && v17->m128i_i64[1] == -8192 && !v15 )
            {
              v15 = (__m128i *)(v11 + 96LL * j);
            }
            v26 = v14 + j;
            ++v14;
          }
          v17->m128i_i64[0] = v10;
          v20 = v4->m128i_i64[1];
          v21 = v17 + 1;
          v17[1].m128i_i64[0] = 0;
          v17->m128i_i64[1] = v20;
          v22 = v17 + 2;
          v23 = v17 + 6;
          v23[-5].m128i_i64[1] = 1;
          do
          {
            if ( v22 )
            {
              v22->m128i_i64[0] = -1;
              v22->m128i_i64[1] = -1;
            }
            ++v22;
          }
          while ( v23 != v22 );
          sub_2D3F0A0(v21, v4 + 1);
          result = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1u;
          *(_DWORD *)(a1 + 8) = result;
          if ( (v4[1].m128i_i8[8] & 1) != 0 )
            goto LABEL_23;
          v24 = v4[2].m128i_u32[2];
          v25 = v4[2].m128i_i64[0];
          v4 += 6;
          result = sub_C7D6A0(v25, 16 * v24, 8);
          if ( a3 == v4 )
            return result;
        }
      }
      if ( v10 != -8192 || v4->m128i_i64[1] != -8192 )
        goto LABEL_10;
      v4 += 6;
    }
    while ( a3 != v4 );
  }
  return result;
}
