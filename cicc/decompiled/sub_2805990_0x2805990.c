// Function: sub_2805990
// Address: 0x2805990
//
_QWORD *__fastcall sub_2805990(__int64 a1, int a2)
{
  unsigned __int64 v2; // rax
  __int64 v4; // rbx
  __int64 v5; // r14
  unsigned int v6; // edi
  _QWORD *result; // rax
  __int64 v8; // rdx
  __int64 v9; // r15
  _QWORD *i; // rdx
  const __m128i *v11; // rbx
  const __m128i *v12; // r14
  __int64 v13; // r12
  int v14; // r15d
  int v15; // eax
  __m128i *v16; // rcx
  int v17; // r11d
  unsigned int j; // esi
  __m128i *v19; // rax
  __int64 v20; // rdi
  __m128i v21; // xmm1
  unsigned int v22; // esi
  __int64 v23; // rdx
  _QWORD *k; // rdx
  __int64 v25; // [rsp+0h] [rbp-70h]
  __int64 v26; // [rsp+8h] [rbp-68h]
  __int64 v27; // [rsp+20h] [rbp-50h]
  __int64 v28; // [rsp+28h] [rbp-48h]
  unsigned int v29; // [rsp+38h] [rbp-38h] BYREF
  unsigned int v30[13]; // [rsp+3Ch] [rbp-34h] BYREF

  v2 = (unsigned int)(a2 - 1);
  v4 = *(unsigned int *)(a1 + 24);
  v5 = *(_QWORD *)(a1 + 8);
  v6 = (((((((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4) | ((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 8)
        | ((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4)
        | ((v2 | (v2 >> 1)) >> 2)
        | v2
        | (v2 >> 1)) >> 16)
      | ((((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4) | ((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 8)
      | ((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4)
      | ((v2 | (v2 >> 1)) >> 2)
      | v2
      | (v2 >> 1))
     + 1;
  if ( v6 < 0x40 )
    v6 = 64;
  *(_DWORD *)(a1 + 24) = v6;
  result = (_QWORD *)sub_C7D670(16LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v26 = 16 * v4;
    v9 = v5 + 16 * v4;
    for ( i = &result[2 * v8]; i != result; result += 2 )
    {
      if ( result )
      {
        *result = -4096;
        result[1] = -4096;
      }
    }
    if ( v9 != v5 )
    {
      v25 = v5;
      v11 = (const __m128i *)v5;
      v12 = (const __m128i *)v9;
      while ( 1 )
      {
        v13 = v11->m128i_i64[0];
        if ( v11->m128i_i64[0] == -4096 )
        {
          if ( v11->m128i_i64[1] != -4096 )
            goto LABEL_12;
          if ( v12 == ++v11 )
            goto LABEL_23;
        }
        else if ( v13 == -8192 && v11->m128i_i64[1] == -8192 )
        {
          if ( v12 == ++v11 )
            goto LABEL_23;
        }
        else
        {
LABEL_12:
          v14 = *(_DWORD *)(a1 + 24);
          if ( !v14 )
          {
            MEMORY[0] = _mm_loadu_si128(v11);
            BUG();
          }
          v27 = *(_QWORD *)(a1 + 8);
          v28 = v11->m128i_i64[1];
          v29 = ((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4);
          v30[0] = ((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4);
          v15 = sub_28052C0(v30, &v29);
          v16 = 0;
          v17 = 1;
          for ( j = (v14 - 1) & v15; ; j = (v14 - 1) & v22 )
          {
            v19 = (__m128i *)(v27 + 16LL * j);
            v20 = v19->m128i_i64[0];
            if ( v13 == v19->m128i_i64[0] && v19->m128i_i64[1] == v28 )
              break;
            if ( v20 == -4096 )
            {
              if ( v19->m128i_i64[1] == -4096 )
              {
                if ( v16 )
                  v19 = v16;
                break;
              }
            }
            else if ( v20 == -8192 && v19->m128i_i64[1] == -8192 && !v16 )
            {
              v16 = (__m128i *)(v27 + 16LL * j);
            }
            v22 = v17 + j;
            ++v17;
          }
          v21 = _mm_loadu_si128(v11++);
          *v19 = v21;
          ++*(_DWORD *)(a1 + 16);
          if ( v12 == v11 )
          {
LABEL_23:
            v5 = v25;
            return (_QWORD *)sub_C7D6A0(v5, v26, 8);
          }
        }
      }
    }
    return (_QWORD *)sub_C7D6A0(v5, v26, 8);
  }
  else
  {
    v23 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = &result[2 * v23]; k != result; result += 2 )
    {
      if ( result )
      {
        *result = -4096;
        result[1] = -4096;
      }
    }
  }
  return result;
}
