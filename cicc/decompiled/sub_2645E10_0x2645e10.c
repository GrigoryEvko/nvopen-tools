// Function: sub_2645E10
// Address: 0x2645e10
//
__int64 __fastcall sub_2645E10(__int64 a1, int a2)
{
  unsigned int v3; // r13d
  __int64 v4; // r12
  unsigned int v5; // eax
  __int64 result; // rax
  __int64 v7; // rdx
  __int64 v8; // r9
  __int64 i; // rdx
  __int64 v10; // rdx
  __int64 v11; // rsi
  int v12; // r10d
  int v13; // ecx
  int v14; // r10d
  __int64 v15; // rdi
  __m128i *v16; // r14
  int v17; // r11d
  unsigned int j; // eax
  __m128i *v19; // r8
  __int64 v20; // r15
  __int32 v21; // eax
  unsigned int v22; // eax
  __int64 v23; // rdx
  __int64 k; // rdx
  __int64 v25; // [rsp+8h] [rbp-38h]

  v3 = *(_DWORD *)(a1 + 24);
  v4 = *(_QWORD *)(a1 + 8);
  v5 = sub_AF1560((unsigned int)(a2 - 1));
  if ( v5 < 0x40 )
    v5 = 64;
  *(_DWORD *)(a1 + 24) = v5;
  result = sub_C7D670(32LL * v5, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    v7 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v25 = 32LL * v3;
    v8 = v4 + v25;
    for ( i = result + 32 * v7; i != result; result += 32 )
    {
      if ( result )
      {
        *(_QWORD *)result = -4096;
        *(_DWORD *)(result + 8) = -1;
      }
    }
    if ( v8 != v4 )
    {
      v10 = v4;
      while ( 1 )
      {
        v11 = *(_QWORD *)v10;
        if ( *(_QWORD *)v10 == -4096 )
        {
          if ( *(_DWORD *)(v10 + 8) != -1 )
            goto LABEL_12;
          v10 += 32;
          if ( v8 == v10 )
            return sub_C7D6A0(v4, v25, 8);
        }
        else if ( v11 == -8192 && *(_DWORD *)(v10 + 8) == -2 )
        {
          v10 += 32;
          if ( v8 == v10 )
            return sub_C7D6A0(v4, v25, 8);
        }
        else
        {
LABEL_12:
          v12 = *(_DWORD *)(a1 + 24);
          if ( !v12 )
          {
            MEMORY[0] = *(_QWORD *)v10;
            BUG();
          }
          v13 = *(_DWORD *)(v10 + 8);
          v14 = v12 - 1;
          v16 = 0;
          v17 = 1;
          for ( j = v14
                  & (((0xBF58476D1CE4E5B9LL
                     * ((unsigned int)(37 * v13)
                      | ((unsigned __int64)(((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4)) << 32))) >> 31)
                   ^ (756364221 * v13)); ; j = v14 & v22 )
          {
            v15 = *(_QWORD *)(a1 + 8);
            v19 = (__m128i *)(v15 + 32LL * j);
            v20 = v19->m128i_i64[0];
            if ( v11 == v19->m128i_i64[0] && v19->m128i_i32[2] == v13 )
              break;
            if ( v20 == -4096 )
            {
              if ( v19->m128i_i32[2] == -1 )
              {
                if ( v16 )
                  v19 = v16;
                break;
              }
            }
            else if ( v20 == -8192 && v19->m128i_i32[2] == -2 && !v16 )
            {
              v16 = (__m128i *)(v15 + 32LL * j);
            }
            v22 = v17 + j;
            ++v17;
          }
          v19->m128i_i64[0] = v11;
          v21 = *(_DWORD *)(v10 + 8);
          v10 += 32;
          v19->m128i_i32[2] = v21;
          v19[1] = _mm_loadu_si128((const __m128i *)(v10 - 16));
          ++*(_DWORD *)(a1 + 16);
          if ( v8 == v10 )
            return sub_C7D6A0(v4, v25, 8);
        }
      }
    }
    return sub_C7D6A0(v4, v25, 8);
  }
  else
  {
    v23 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = result + 32 * v23; k != result; result += 32 )
    {
      if ( result )
      {
        *(_QWORD *)result = -4096;
        *(_DWORD *)(result + 8) = -1;
      }
    }
  }
  return result;
}
