// Function: sub_2E64C00
// Address: 0x2e64c00
//
__int64 __fastcall sub_2E64C00(const __m128i *a1, unsigned int a2)
{
  unsigned int v2; // r14d
  __int64 v4; // r12
  char v5; // dl
  unsigned __int64 v6; // rax
  unsigned int v7; // r13d
  __int64 v8; // rdi
  __int64 v9; // rax
  bool v10; // zf
  __int64 v11; // r11
  __m128i *v12; // rax
  __int64 v13; // rdx
  __m128i *k; // rdx
  __int64 m; // rdx
  __int64 v16; // rcx
  const __m128i *v17; // r9
  int v18; // edi
  __int64 v19; // r8
  int v20; // r14d
  __int64 *v21; // r15
  unsigned int n; // eax
  __int64 *v23; // rsi
  __int64 v24; // r10
  __int32 v25; // edi
  __int64 result; // rax
  const __m128i *v27; // r13
  const __m128i *v28; // rsi
  const __m128i *v29; // rax
  _BYTE *v30; // r12
  __int64 v31; // rax
  __int64 v32; // rcx
  __int64 i; // rcx
  _BYTE *v34; // rdx
  const __m128i *v35; // r8
  int v36; // esi
  __int64 v37; // rdi
  __int64 *v38; // r14
  int v39; // r11d
  unsigned int j; // eax
  __int64 *v41; // r10
  __int64 v42; // r15
  unsigned int v43; // eax
  __int32 v44; // esi
  int v45; // eax
  unsigned int v46; // eax
  __int64 v47; // [rsp+0h] [rbp-A0h]
  _BYTE v48[144]; // [rsp+10h] [rbp-90h] BYREF

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
    v28 = a1 + 7;
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
    v28 = a1 + 7;
    if ( !v5 )
    {
      v7 = a1[1].m128i_u32[2];
      v8 = 24LL * (unsigned int)v6;
      goto LABEL_5;
    }
    goto LABEL_38;
  }
  if ( v5 )
  {
    v27 = a1 + 1;
    v28 = a1 + 7;
    v2 = 64;
LABEL_38:
    v29 = v27;
    v30 = v48;
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
          *(__m128i *)v30 = _mm_loadu_si128(v29);
        v30 += 24;
        *((_DWORD *)v30 - 2) = v29[1].m128i_i32[0];
      }
      v29 = (const __m128i *)((char *)v29 + 24);
      if ( v29 == v28 )
      {
        if ( v2 > 4 )
        {
          a1->m128i_i8[8] &= ~1u;
          v31 = sub_C7D670(24LL * v2, 8);
          a1[1].m128i_i32[2] = v2;
          a1[1].m128i_i64[0] = v31;
        }
        v10 = (a1->m128i_i64[1] & 1) == 0;
        a1->m128i_i64[1] &= 1uLL;
        if ( v10 )
        {
          result = a1[1].m128i_i64[0];
          v32 = 24LL * a1[1].m128i_u32[2];
        }
        else
        {
          result = (__int64)v27;
          v32 = 96;
        }
        for ( i = result + v32; i != result; result += 24 )
        {
          if ( result )
          {
            *(_QWORD *)result = -4096;
            *(_QWORD *)(result + 8) = -4096;
          }
        }
        v34 = v48;
        if ( v30 == v48 )
          return result;
        while ( 2 )
        {
          v16 = *(_QWORD *)v34;
          if ( *(_QWORD *)v34 == -4096 )
          {
            if ( *((_QWORD *)v34 + 1) == -4096 )
              goto LABEL_69;
          }
          else if ( v16 == -8192 && *((_QWORD *)v34 + 1) == -8192 )
          {
            goto LABEL_69;
          }
          if ( (a1->m128i_i8[8] & 1) != 0 )
          {
            v35 = v27;
            v36 = 3;
          }
          else
          {
            v44 = a1[1].m128i_i32[2];
            v35 = (const __m128i *)a1[1].m128i_i64[0];
            if ( !v44 )
            {
LABEL_92:
              MEMORY[0] = v16;
              BUG();
            }
            v36 = v44 - 1;
          }
          v37 = *((_QWORD *)v34 + 1);
          v38 = 0;
          v39 = 1;
          for ( j = v36
                  & (((0xBF58476D1CE4E5B9LL
                     * (((unsigned int)v37 >> 9) ^ ((unsigned int)v37 >> 4)
                      | ((unsigned __int64)(((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4)) << 32))) >> 31)
                   ^ (484763065 * (((unsigned int)v37 >> 9) ^ ((unsigned int)v37 >> 4)))); ; j = v36 & v43 )
          {
            v41 = &v35->m128i_i64[3 * j];
            v42 = *v41;
            if ( v16 == *v41 && v41[1] == v37 )
              break;
            if ( v42 == -4096 )
            {
              if ( v41[1] == -4096 )
              {
                if ( v38 )
                  v41 = v38;
                break;
              }
            }
            else if ( v42 == -8192 && v41[1] == -8192 && !v38 )
            {
              v38 = &v35->m128i_i64[3 * j];
            }
            v43 = v39 + j;
            ++v39;
          }
          v45 = *((_DWORD *)v34 + 4);
          *v41 = v16;
          v41[1] = v37;
          *((_DWORD *)v41 + 4) = v45;
          result = (2 * ((unsigned __int32)a1->m128i_i32[2] >> 1) + 2) | a1->m128i_i32[2] & 1;
          a1->m128i_i32[2] = result;
LABEL_69:
          v34 += 24;
          if ( v30 == v34 )
            return result;
          continue;
        }
      }
    }
  }
  v7 = a1[1].m128i_u32[2];
  v2 = 64;
  v8 = 1536;
LABEL_5:
  v9 = sub_C7D670(v8, 8);
  a1[1].m128i_i32[2] = v2;
  a1[1].m128i_i64[0] = v9;
LABEL_6:
  v10 = (a1->m128i_i64[1] & 1) == 0;
  a1->m128i_i64[1] &= 1uLL;
  v47 = 24LL * v7;
  v11 = v4 + v47;
  if ( v10 )
  {
    v12 = (__m128i *)a1[1].m128i_i64[0];
    v13 = 24LL * a1[1].m128i_u32[2];
  }
  else
  {
    v12 = (__m128i *)&a1[1];
    v13 = 96;
  }
  for ( k = (__m128i *)((char *)v12 + v13); k != v12; v12 = (__m128i *)((char *)v12 + 24) )
  {
    if ( v12 )
    {
      v12->m128i_i64[0] = -4096;
      v12->m128i_i64[1] = -4096;
    }
  }
  for ( m = v4; v11 != m; m += 24 )
  {
    v16 = *(_QWORD *)m;
    if ( *(_QWORD *)m == -4096 )
    {
      if ( *(_QWORD *)(m + 8) != -4096 )
        goto LABEL_15;
    }
    else if ( v16 != -8192 || *(_QWORD *)(m + 8) != -8192 )
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
          goto LABEL_92;
        v18 = v25 - 1;
      }
      v19 = *(_QWORD *)(m + 8);
      v20 = 1;
      v21 = 0;
      for ( n = v18
              & (((0xBF58476D1CE4E5B9LL
                 * (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4)
                  | ((unsigned __int64)(((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4)) << 32))) >> 31)
               ^ (484763065 * (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4)))); ; n = v18 & v46 )
      {
        v23 = &v17->m128i_i64[3 * n];
        v24 = *v23;
        if ( v16 == *v23 && v23[1] == v19 )
          break;
        if ( v24 == -4096 )
        {
          if ( v23[1] == -4096 )
          {
            if ( v21 )
              v23 = v21;
            break;
          }
        }
        else if ( v24 == -8192 && v23[1] == -8192 && !v21 )
        {
          v21 = &v17->m128i_i64[3 * n];
        }
        v46 = v20 + n;
        ++v20;
      }
      *v23 = v16;
      v23[1] = *(_QWORD *)(m + 8);
      *((_DWORD *)v23 + 4) = *(_DWORD *)(m + 16);
      a1->m128i_i32[2] = (2 * ((unsigned __int32)a1->m128i_i32[2] >> 1) + 2) | a1->m128i_i32[2] & 1;
    }
  }
  return sub_C7D6A0(v4, v47, 8);
}
