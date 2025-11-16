// Function: sub_2B75580
// Address: 0x2b75580
//
__int64 __fastcall sub_2B75580(const __m128i *a1, unsigned int a2)
{
  unsigned int v2; // r14d
  __int64 v4; // r12
  char v5; // dl
  unsigned __int64 v6; // rax
  unsigned int v7; // r13d
  __int64 v8; // rdi
  __int64 v9; // rax
  bool v10; // zf
  __int64 *v11; // r11
  __m128i *v12; // rax
  __int64 v13; // rdx
  __m128i *k; // rdx
  __int64 *m; // rdx
  __int64 v16; // rsi
  const __m128i *v17; // r9
  int v18; // edi
  __int64 v19; // r8
  int v20; // r14d
  const __m128i *v21; // r15
  unsigned int n; // eax
  __m128i *v23; // rcx
  __int64 v24; // r10
  __int32 v25; // edi
  __int64 result; // rax
  const __m128i *v27; // r13
  const __m128i *v28; // rsi
  const __m128i *v29; // rax
  __m128i *v30; // r12
  __int64 v31; // rax
  __int64 v32; // rcx
  __int64 i; // rcx
  __m128i *v34; // rdx
  __int64 v35; // rcx
  const __m128i *v36; // r8
  int v37; // esi
  __int64 v38; // rdi
  const __m128i *v39; // r14
  int v40; // r11d
  int j; // eax
  __m128i *v42; // r10
  __int64 v43; // r15
  int v44; // eax
  __int32 v45; // esi
  unsigned int v46; // eax
  __int64 v47; // [rsp+0h] [rbp-80h]
  _BYTE v48[112]; // [rsp+10h] [rbp-70h] BYREF

  v2 = a2;
  v4 = a1[1].m128i_i64[0];
  v5 = a1->m128i_i8[8] & 1;
  if ( a2 <= 4 )
  {
    if ( !v5 )
    {
      v7 = a1[1].m128i_u32[2];
      a1->m128i_i8[8] |= 1u;
      goto LABEL_6;
    }
    v27 = a1 + 1;
    v28 = a1 + 5;
    goto LABEL_38;
  }
  v6 = ((((((((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
          | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
          | (a2 - 1)
          | ((unsigned __int64)(a2 - 1) >> 1)) >> 8)
        | (((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
        | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
        | (a2 - 1)
        | ((unsigned __int64)(a2 - 1) >> 1)) >> 16)
      | (((((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
        | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
        | (a2 - 1)
        | ((unsigned __int64)(a2 - 1) >> 1)) >> 8)
      | (((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
      | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
      | (a2 - 1)
      | ((unsigned __int64)(a2 - 1) >> 1))
     + 1;
  v2 = v6;
  if ( (unsigned int)v6 > 0x40 )
  {
    v27 = a1 + 1;
    v28 = a1 + 5;
    if ( !v5 )
    {
      v7 = a1[1].m128i_u32[2];
      v8 = 16LL * (unsigned int)v6;
      goto LABEL_5;
    }
    goto LABEL_38;
  }
  if ( v5 )
  {
    v27 = a1 + 1;
    v28 = a1 + 5;
    v2 = 64;
LABEL_38:
    v29 = v27;
    v30 = (__m128i *)v48;
    while ( 1 )
    {
      if ( v29->m128i_i64[0] == -4096 )
      {
        if ( v29->m128i_i64[1] != -4096 )
          goto LABEL_41;
      }
      else if ( v29->m128i_i64[0] != -8192 || v29->m128i_i64[1] != -8192 )
      {
LABEL_41:
        if ( v30 )
          *v30 = _mm_loadu_si128(v29);
        ++v30;
      }
      if ( ++v29 == v28 )
      {
        if ( v2 > 4 )
        {
          a1->m128i_i8[8] &= ~1u;
          v31 = sub_C7D670(16LL * v2, 8);
          a1[1].m128i_i32[2] = v2;
          a1[1].m128i_i64[0] = v31;
        }
        v10 = (a1->m128i_i64[1] & 1) == 0;
        a1->m128i_i64[1] &= 1uLL;
        if ( v10 )
        {
          result = a1[1].m128i_i64[0];
          v32 = 16LL * a1[1].m128i_u32[2];
        }
        else
        {
          result = (__int64)v27;
          v32 = 64;
        }
        for ( i = result + v32; i != result; result += 16 )
        {
          if ( result )
          {
            *(_QWORD *)result = -4096;
            *(_QWORD *)(result + 8) = -4096;
          }
        }
        v34 = (__m128i *)v48;
        if ( v30 == (__m128i *)v48 )
          return result;
        while ( 2 )
        {
          v35 = v34->m128i_i64[0];
          if ( v34->m128i_i64[0] == -4096 )
          {
            if ( v34->m128i_i64[1] == -4096 )
              goto LABEL_69;
          }
          else if ( v35 == -8192 && v34->m128i_i64[1] == -8192 )
          {
            goto LABEL_69;
          }
          if ( (a1->m128i_i8[8] & 1) != 0 )
          {
            v36 = v27;
            v37 = 3;
          }
          else
          {
            v45 = a1[1].m128i_i32[2];
            v36 = (const __m128i *)a1[1].m128i_i64[0];
            if ( !v45 )
            {
              MEMORY[0] = v34->m128i_i64[0];
              BUG();
            }
            v37 = v45 - 1;
          }
          v38 = v34->m128i_i64[1];
          v39 = 0;
          v40 = 1;
          for ( j = v37
                  & (((0xBF58476D1CE4E5B9LL
                     * (((unsigned int)v38 >> 9) ^ ((unsigned int)v38 >> 4)
                      | ((unsigned __int64)(((unsigned int)v35 >> 9) ^ ((unsigned int)v35 >> 4)) << 32))) >> 31)
                   ^ (484763065 * (((unsigned int)v38 >> 9) ^ ((unsigned int)v38 >> 4)))); ; j = v37 & v44 )
          {
            v42 = (__m128i *)&v36[j];
            v43 = v42->m128i_i64[0];
            if ( v35 == v42->m128i_i64[0] && v42->m128i_i64[1] == v38 )
              break;
            if ( v43 == -4096 )
            {
              if ( v42->m128i_i64[1] == -4096 )
              {
                if ( v39 )
                  v42 = (__m128i *)v39;
                break;
              }
            }
            else if ( v43 == -8192 && v42->m128i_i64[1] == -8192 && !v39 )
            {
              v39 = &v36[j];
            }
            v44 = v40 + j;
            ++v40;
          }
          v42->m128i_i64[0] = v35;
          v42->m128i_i64[1] = v38;
          result = (2 * ((unsigned __int32)a1->m128i_i32[2] >> 1) + 2) | a1->m128i_i32[2] & 1;
          a1->m128i_i32[2] = result;
LABEL_69:
          if ( v30 == ++v34 )
            return result;
          continue;
        }
      }
    }
  }
  v7 = a1[1].m128i_u32[2];
  v2 = 64;
  v8 = 1024;
LABEL_5:
  v9 = sub_C7D670(v8, 8);
  a1[1].m128i_i32[2] = v2;
  a1[1].m128i_i64[0] = v9;
LABEL_6:
  v10 = (a1->m128i_i64[1] & 1) == 0;
  a1->m128i_i64[1] &= 1uLL;
  v47 = 16LL * v7;
  v11 = (__int64 *)(v4 + v47);
  if ( v10 )
  {
    v12 = (__m128i *)a1[1].m128i_i64[0];
    v13 = a1[1].m128i_u32[2];
  }
  else
  {
    v12 = (__m128i *)&a1[1];
    v13 = 4;
  }
  for ( k = &v12[v13]; k != v12; ++v12 )
  {
    if ( v12 )
    {
      v12->m128i_i64[0] = -4096;
      v12->m128i_i64[1] = -4096;
    }
  }
  for ( m = (__int64 *)v4; v11 != m; m += 2 )
  {
    v16 = *m;
    if ( *m == -4096 )
    {
      if ( m[1] != -4096 )
        goto LABEL_15;
    }
    else if ( v16 != -8192 || m[1] != -8192 )
    {
LABEL_15:
      if ( (a1->m128i_i8[8] & 1) != 0 )
      {
        v17 = a1 + 1;
        v18 = 3;
      }
      else
      {
        v25 = a1[1].m128i_i32[2];
        v17 = (const __m128i *)a1[1].m128i_i64[0];
        if ( !v25 )
        {
          MEMORY[0] = *m;
          BUG();
        }
        v18 = v25 - 1;
      }
      v19 = m[1];
      v20 = 1;
      v21 = 0;
      for ( n = v18
              & (((0xBF58476D1CE4E5B9LL
                 * (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4)
                  | ((unsigned __int64)(((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4)) << 32))) >> 31)
               ^ (484763065 * (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4)))); ; n = v18 & v46 )
      {
        v23 = (__m128i *)&v17[n];
        v24 = v23->m128i_i64[0];
        if ( v16 == v23->m128i_i64[0] && v23->m128i_i64[1] == v19 )
          break;
        if ( v24 == -4096 )
        {
          if ( v23->m128i_i64[1] == -4096 )
          {
            if ( v21 )
              v23 = (__m128i *)v21;
            break;
          }
        }
        else if ( v24 == -8192 && v23->m128i_i64[1] == -8192 && !v21 )
        {
          v21 = &v17[n];
        }
        v46 = v20 + n;
        ++v20;
      }
      v23->m128i_i64[0] = v16;
      v23->m128i_i64[1] = m[1];
      a1->m128i_i32[2] = (2 * ((unsigned __int32)a1->m128i_i32[2] >> 1) + 2) | a1->m128i_i32[2] & 1;
    }
  }
  return sub_C7D6A0(v4, v47, 8);
}
