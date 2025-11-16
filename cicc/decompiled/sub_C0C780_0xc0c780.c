// Function: sub_C0C780
// Address: 0xc0c780
//
__int64 __fastcall sub_C0C780(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v4; // rbx
  unsigned int v5; // eax
  __int64 result; // rax
  const __m128i *v7; // r11
  const __m128i *v8; // r10
  __int64 i; // rdx
  const __m128i *v10; // rbx
  __int32 v11; // r12d
  int v12; // r14d
  int v13; // r14d
  __int64 v14; // r9
  __m128i *v15; // r11
  unsigned int j; // r8d
  __m128i *v17; // rcx
  __int32 v18; // r15d
  __m128i *v19; // r15
  char *v20; // rdi
  const void *v21; // rsi
  bool v22; // al
  size_t v23; // rdx
  unsigned int v24; // r8d
  int v25; // eax
  __int64 v26; // rdx
  __int64 k; // rdx
  __m128i *v28; // [rsp+0h] [rbp-70h]
  const __m128i *v29; // [rsp+8h] [rbp-68h]
  __m128i *v30; // [rsp+10h] [rbp-60h]
  unsigned int v31; // [rsp+1Ch] [rbp-54h]
  __int64 v32; // [rsp+20h] [rbp-50h]
  const __m128i *v33; // [rsp+28h] [rbp-48h]
  __int64 v34; // [rsp+30h] [rbp-40h]
  const __m128i *v35; // [rsp+38h] [rbp-38h]
  int v36; // [rsp+38h] [rbp-38h]

  v2 = (unsigned int)(a2 - 1);
  v4 = *(unsigned int *)(a1 + 24);
  v35 = *(const __m128i **)(a1 + 8);
  v5 = (((((((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4) | ((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 8)
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
  if ( v5 < 0x40 )
    v5 = 64;
  *(_DWORD *)(a1 + 24) = v5;
  result = sub_C7D670(24LL * v5, 8);
  v7 = v35;
  *(_QWORD *)(a1 + 8) = result;
  if ( v35 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v34 = 24 * v4;
    v8 = (const __m128i *)((char *)v35 + 24 * v4);
    for ( i = result + 24LL * *(unsigned int *)(a1 + 24); i != result; result += 24 )
    {
      if ( result )
      {
        *(_QWORD *)result = -1;
        *(_DWORD *)(result + 8) = 0;
        *(_DWORD *)(result + 12) = 0;
      }
    }
    if ( v8 != v35 )
    {
      v33 = v35;
      v10 = v35;
      while ( 1 )
      {
        v11 = v10->m128i_i32[3];
        if ( v11 )
        {
          if ( v11 == 1 && v10->m128i_i64[0] == -2 )
            goto LABEL_21;
        }
        else if ( v10->m128i_i64[0] == -1 )
        {
          goto LABEL_21;
        }
        v12 = *(_DWORD *)(a1 + 24);
        if ( !v12 )
        {
          MEMORY[0] = _mm_loadu_si128(v10);
          BUG();
        }
        v13 = v12 - 1;
        v14 = *(_QWORD *)(a1 + 8);
        v15 = 0;
        v36 = 1;
        for ( j = v13 & v11; ; j = v13 & v24 )
        {
          v17 = (__m128i *)(v14 + 24LL * j);
          v18 = v17->m128i_i32[3];
          if ( v11 == v18 )
          {
            v20 = (char *)v10->m128i_i64[0];
            v21 = (const void *)v17->m128i_i64[0];
            v22 = v10->m128i_i64[0] == -1;
            if ( v17->m128i_i64[0] != -1 )
            {
              v22 = v20 + 2 == 0;
              if ( v21 != (const void *)-2LL )
              {
                v23 = v10->m128i_u32[2];
                if ( (_DWORD)v23 != v17->m128i_i32[2] )
                {
                  if ( !v11 )
                    goto LABEL_34;
                  goto LABEL_30;
                }
                v30 = v15;
                v31 = j;
                v32 = v14;
                if ( !v10->m128i_i32[2] )
                  goto LABEL_38;
                v28 = (__m128i *)(v14 + 24LL * j);
                v29 = v8;
                v25 = memcmp(v20, v21, v23);
                v15 = v30;
                j = v31;
                v14 = v32;
                v17 = v28;
                v8 = v29;
                v22 = v25 == 0;
              }
            }
            if ( v22 )
            {
LABEL_38:
              v19 = v17;
              goto LABEL_39;
            }
          }
          if ( !v18 )
            break;
LABEL_30:
          if ( v18 == 1 && v17->m128i_i64[0] == -2 && !v15 )
            v15 = v17;
LABEL_34:
          v24 = v36 + j;
          ++v36;
        }
        if ( v17->m128i_i64[0] != -1 )
          goto LABEL_34;
        v19 = v17;
        if ( v15 )
          v19 = v15;
LABEL_39:
        *v19 = _mm_loadu_si128(v10);
        v19[1].m128i_i64[0] = v10[1].m128i_i64[0];
        ++*(_DWORD *)(a1 + 16);
LABEL_21:
        v10 = (const __m128i *)((char *)v10 + 24);
        if ( v8 == v10 )
        {
          v7 = v33;
          return sub_C7D6A0(v7, v34, 8);
        }
      }
    }
    return sub_C7D6A0(v7, v34, 8);
  }
  else
  {
    v26 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = result + 24 * v26; k != result; result += 24 )
    {
      if ( result )
      {
        *(_QWORD *)result = -1;
        *(_DWORD *)(result + 8) = 0;
        *(_DWORD *)(result + 12) = 0;
      }
    }
  }
  return result;
}
