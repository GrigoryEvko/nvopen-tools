// Function: sub_11C13E0
// Address: 0x11c13e0
//
__int64 __fastcall sub_11C13E0(const __m128i *a1, unsigned int a2)
{
  unsigned int v2; // r14d
  char v4; // dl
  unsigned __int64 v5; // rax
  unsigned int v6; // r12d
  __int64 v7; // rdi
  __int64 v8; // rax
  bool v9; // zf
  __m128i *v10; // rax
  __int64 v11; // rdx
  __m128i *k; // rdx
  const __m128i *v13; // r13
  __int64 m; // r15
  __int64 v15; // rbx
  const __m128i *v16; // rsi
  int v17; // r12d
  unsigned int v18; // r14d
  unsigned int v19; // eax
  __int64 *v20; // r10
  int v21; // r9d
  unsigned int n; // eax
  __int64 *v23; // rdi
  __int64 v24; // rdx
  int v25; // edx
  __int64 result; // rax
  const __m128i *v27; // rsi
  unsigned int *v28; // r12
  const __m128i *v29; // rax
  unsigned int *v30; // r13
  __int64 v31; // rax
  __m128i *v32; // rax
  __int64 v33; // rcx
  __m128i *i; // rcx
  __int64 v36; // rbx
  const __m128i *v37; // r8
  int v38; // r13d
  unsigned int v39; // r14d
  unsigned int v40; // eax
  int v41; // r10d
  __int64 *v42; // r9
  unsigned int j; // eax
  __int64 *v44; // rdi
  __int64 v45; // rdx
  unsigned int v46; // eax
  __int32 v47; // edx
  unsigned int v48; // eax
  unsigned int v49; // eax
  __int64 v50; // [rsp+0h] [rbp-130h]
  const __m128i *v51; // [rsp+8h] [rbp-128h]
  const __m128i *v52; // [rsp+8h] [rbp-128h]
  __int64 v53; // [rsp+10h] [rbp-120h]
  __int64 v54; // [rsp+20h] [rbp-110h]
  unsigned int *v55; // [rsp+20h] [rbp-110h]
  const __m128i *v56; // [rsp+28h] [rbp-108h]
  unsigned int v57; // [rsp+3Ch] [rbp-F4h] BYREF
  unsigned int v58[60]; // [rsp+40h] [rbp-F0h] BYREF

  v2 = a2;
  v53 = a1[1].m128i_i64[0];
  v4 = a1->m128i_i8[8] & 1;
  if ( a2 <= 8 )
  {
    if ( !v4 )
    {
      v6 = a1[1].m128i_u32[2];
      a1->m128i_i8[8] |= 1u;
      goto LABEL_6;
    }
    v27 = a1 + 13;
    v52 = a1 + 1;
    goto LABEL_38;
  }
  v5 = ((((((((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
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
  v2 = v5;
  if ( (unsigned int)v5 > 0x40 )
  {
    v52 = a1 + 1;
    v27 = a1 + 13;
    if ( !v4 )
    {
      v6 = a1[1].m128i_u32[2];
      v7 = 24LL * (unsigned int)v5;
      goto LABEL_5;
    }
    goto LABEL_38;
  }
  if ( v4 )
  {
    v27 = a1 + 13;
    v2 = 64;
    v52 = a1 + 1;
LABEL_38:
    v28 = v58;
    v29 = v52;
    v30 = v58;
    while ( 1 )
    {
      while ( v29->m128i_i64[0] == -4096 )
      {
        if ( v29->m128i_i32[2] != 100 )
          goto LABEL_40;
        v29 = (const __m128i *)((char *)v29 + 24);
        if ( v29 == v27 )
        {
LABEL_47:
          if ( v2 > 8 )
          {
            a1->m128i_i8[8] &= ~1u;
            v31 = sub_C7D670(24LL * v2, 8);
            a1[1].m128i_i32[2] = v2;
            a1[1].m128i_i64[0] = v31;
          }
          v9 = (a1->m128i_i64[1] & 1) == 0;
          a1->m128i_i64[1] &= 1uLL;
          if ( v9 )
          {
            v32 = (__m128i *)a1[1].m128i_i64[0];
            v33 = 24LL * a1[1].m128i_u32[2];
          }
          else
          {
            v32 = (__m128i *)v52;
            v33 = 192;
          }
          for ( i = (__m128i *)((char *)v32 + v33); i != v32; v32 = (__m128i *)((char *)v32 + 24) )
          {
            if ( v32 )
            {
              v32->m128i_i64[0] = -4096;
              v32->m128i_i32[2] = 100;
            }
          }
          result = (__int64)&v57;
          if ( v30 == v58 )
            return result;
          v55 = v30;
          while ( 2 )
          {
            v36 = *(_QWORD *)v28;
            if ( *(_QWORD *)v28 == -4096 )
            {
              if ( v28[2] == 100 )
                goto LABEL_74;
            }
            else if ( v36 == -8192 && v28[2] == 101 )
            {
              goto LABEL_74;
            }
            if ( (a1->m128i_i8[8] & 1) != 0 )
            {
              v37 = v52;
              v38 = 7;
            }
            else
            {
              v47 = a1[1].m128i_i32[2];
              v37 = (const __m128i *)a1[1].m128i_i64[0];
              if ( !v47 )
              {
                MEMORY[0] = *(_QWORD *)v28;
                BUG();
              }
              v38 = v47 - 1;
            }
            v39 = v28[2];
            v56 = v37;
            v57 = v39;
            v40 = sub_CF97C0(&v57);
            v41 = 1;
            v42 = 0;
            for ( j = v38
                    & (((0xBF58476D1CE4E5B9LL
                       * (v40 | ((unsigned __int64)(((unsigned int)v36 >> 9) ^ ((unsigned int)v36 >> 4)) << 32))) >> 31)
                     ^ (484763065 * v40)); ; j = v38 & v46 )
            {
              v44 = &v56->m128i_i64[3 * j];
              v45 = *v44;
              if ( v36 == *v44 && *((_DWORD *)v44 + 2) == v39 )
                break;
              if ( v45 == -4096 )
              {
                if ( *((_DWORD *)v44 + 2) == 100 )
                {
                  if ( v42 )
                    v44 = v42;
                  break;
                }
              }
              else if ( v45 == -8192 && *((_DWORD *)v44 + 2) == 101 && !v42 )
              {
                v42 = &v56->m128i_i64[3 * j];
              }
              v46 = v41 + j;
              ++v41;
            }
            v48 = v28[4];
            *v44 = v36;
            *((_DWORD *)v44 + 2) = v39;
            *((_DWORD *)v44 + 4) = v48;
            result = (2 * ((unsigned __int32)a1->m128i_i32[2] >> 1) + 2) | a1->m128i_i32[2] & 1;
            a1->m128i_i32[2] = result;
LABEL_74:
            v28 += 6;
            if ( v55 == v28 )
              return result;
            continue;
          }
        }
      }
      if ( v29->m128i_i64[0] != -8192 || v29->m128i_i32[2] != 101 )
      {
LABEL_40:
        if ( v30 )
          *(__m128i *)v30 = _mm_loadu_si128(v29);
        v30 += 6;
        *(v30 - 2) = v29[1].m128i_u32[0];
      }
      v29 = (const __m128i *)((char *)v29 + 24);
      if ( v29 == v27 )
        goto LABEL_47;
    }
  }
  v6 = a1[1].m128i_u32[2];
  v2 = 64;
  v7 = 1536;
LABEL_5:
  v8 = sub_C7D670(v7, 8);
  a1[1].m128i_i32[2] = v2;
  a1[1].m128i_i64[0] = v8;
LABEL_6:
  v9 = (a1->m128i_i64[1] & 1) == 0;
  a1->m128i_i64[1] &= 1uLL;
  v50 = 24LL * v6;
  v54 = v50 + v53;
  if ( v9 )
  {
    v10 = (__m128i *)a1[1].m128i_i64[0];
    v11 = 24LL * a1[1].m128i_u32[2];
  }
  else
  {
    v10 = (__m128i *)&a1[1];
    v11 = 192;
  }
  for ( k = (__m128i *)((char *)v10 + v11); k != v10; v10 = (__m128i *)((char *)v10 + 24) )
  {
    if ( v10 )
    {
      v10->m128i_i64[0] = -4096;
      v10->m128i_i32[2] = 100;
    }
  }
  v13 = a1;
  v51 = a1 + 1;
  for ( m = v53; v54 != m; m += 24 )
  {
    v15 = *(_QWORD *)m;
    if ( *(_QWORD *)m == -4096 )
    {
      if ( *(_DWORD *)(m + 8) != 100 )
        goto LABEL_15;
    }
    else if ( v15 != -8192 || *(_DWORD *)(m + 8) != 101 )
    {
LABEL_15:
      if ( (v13->m128i_i8[8] & 1) != 0 )
      {
        v16 = v51;
        v17 = 7;
      }
      else
      {
        v25 = v13[1].m128i_i32[2];
        v16 = (const __m128i *)v13[1].m128i_i64[0];
        if ( !v25 )
        {
          MEMORY[0] = *(_QWORD *)m;
          BUG();
        }
        v17 = v25 - 1;
      }
      v18 = *(_DWORD *)(m + 8);
      v58[0] = v18;
      v19 = sub_CF97C0(v58);
      v20 = 0;
      v21 = 1;
      for ( n = v17
              & (((0xBF58476D1CE4E5B9LL
                 * (v19 | ((unsigned __int64)(((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4)) << 32))) >> 31)
               ^ (484763065 * v19)); ; n = v17 & v49 )
      {
        v23 = &v16->m128i_i64[3 * n];
        v24 = *v23;
        if ( v15 == *v23 && *((_DWORD *)v23 + 2) == v18 )
          break;
        if ( v24 == -4096 )
        {
          if ( *((_DWORD *)v23 + 2) == 100 )
          {
            if ( v20 )
              v23 = v20;
            break;
          }
        }
        else if ( v24 == -8192 && *((_DWORD *)v23 + 2) == 101 && !v20 )
        {
          v20 = &v16->m128i_i64[3 * n];
        }
        v49 = v21 + n;
        ++v21;
      }
      *v23 = v15;
      *((_DWORD *)v23 + 2) = *(_DWORD *)(m + 8);
      *((_DWORD *)v23 + 4) = *(_DWORD *)(m + 16);
      v13->m128i_i32[2] = (2 * ((unsigned __int32)v13->m128i_i32[2] >> 1) + 2) | v13->m128i_i32[2] & 1;
    }
  }
  return sub_C7D6A0(v53, v50, 8);
}
