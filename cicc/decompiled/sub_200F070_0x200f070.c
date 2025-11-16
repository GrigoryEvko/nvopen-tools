// Function: sub_200F070
// Address: 0x200f070
//
__int64 __fastcall sub_200F070(const __m128i *a1, unsigned int a2)
{
  __int64 result; // rax
  char v4; // dl
  __int64 v5; // r12
  unsigned __int64 v6; // rax
  int v7; // r14d
  __int64 v8; // r15
  const __m128i *v9; // rsi
  const __m128i *v10; // rax
  __m128i *v11; // r12
  __int32 v12; // edx
  __int64 v13; // rax
  bool v14; // zf
  __int64 m128i_i64; // rdx
  __int64 v16; // rdi
  unsigned __int64 v17; // rcx
  const __m128i *v18; // r8
  int v19; // edi
  int v20; // r9d
  int v21; // r14d
  __int64 *v22; // r11
  unsigned int i; // edx
  __int64 *v24; // rsi
  unsigned int v25; // edx
  unsigned __int32 v26; // r13d
  __int64 v27; // rdi
  const __m128i *v28; // rax
  __int64 v29; // rdx
  const __m128i *j; // rdx
  unsigned __int64 v31; // rcx
  const __m128i *v32; // r10
  int v33; // r9d
  int v34; // r11d
  int v35; // r13d
  __int64 *v36; // r14
  unsigned int k; // edx
  __int64 *v38; // rsi
  unsigned int v39; // edx
  __int32 v40; // r9d
  __int64 v41; // rax
  __int32 v42; // edi
  int v43; // r15d
  int v44; // r10d
  _BYTE v45[240]; // [rsp+10h] [rbp-F0h] BYREF

  result = a1->m128i_u8[8];
  v4 = a1->m128i_i8[8] & 1;
  if ( a2 <= 7 )
  {
    if ( v4 )
      return result;
    v5 = a1[1].m128i_i64[0];
    v26 = a1[1].m128i_u32[2];
    a1->m128i_i8[8] = result | 1;
  }
  else
  {
    v5 = a1[1].m128i_i64[0];
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
    v7 = v6;
    if ( (unsigned int)v6 > 0x40 )
    {
      v8 = 24LL * (unsigned int)v6;
      if ( v4 )
      {
LABEL_5:
        v9 = a1 + 13;
        v10 = a1 + 1;
        v11 = (__m128i *)v45;
        do
        {
          while ( !v10->m128i_i64[0] && v10->m128i_i32[2] > 0xFFFFFFFD )
          {
            v10 = (const __m128i *)((char *)v10 + 24);
            if ( v10 == v9 )
              goto LABEL_12;
          }
          if ( v11 )
            *v11 = _mm_loadu_si128(v10);
          v12 = v10[1].m128i_i32[0];
          v10 = (const __m128i *)((char *)v10 + 24);
          v11 = (__m128i *)((char *)v11 + 24);
          v11[-1].m128i_i32[2] = v12;
        }
        while ( v10 != v9 );
LABEL_12:
        a1->m128i_i8[8] &= ~1u;
        v13 = sub_22077B0(v8);
        a1[1].m128i_i32[2] = v7;
        v14 = (a1->m128i_i64[1] & 1) == 0;
        a1->m128i_i64[1] &= 1uLL;
        m128i_i64 = v13;
        a1[1].m128i_i64[0] = v13;
        if ( !v14 )
        {
          m128i_i64 = (__int64)a1[1].m128i_i64;
          v13 = (__int64)a1[1].m128i_i64;
          v8 = 192;
        }
        v16 = v13 + v8;
        while ( 1 )
        {
          if ( m128i_i64 )
          {
            *(_QWORD *)v13 = 0;
            *(_DWORD *)(v13 + 8) = -1;
          }
          v13 += 24;
          if ( v16 == v13 )
            break;
          m128i_i64 = v13;
        }
        for ( result = (__int64)v45; v11 != (__m128i *)result; result += 24 )
        {
          v17 = *(_QWORD *)result;
          if ( *(_QWORD *)result || *(_DWORD *)(result + 8) <= 0xFFFFFFFD )
          {
            if ( (a1->m128i_i8[8] & 1) != 0 )
            {
              v18 = a1 + 1;
              v19 = 7;
            }
            else
            {
              v42 = a1[1].m128i_i32[2];
              v18 = (const __m128i *)a1[1].m128i_i64[0];
              if ( !v42 )
                goto LABEL_79;
              v19 = v42 - 1;
            }
            v20 = *(_DWORD *)(result + 8);
            v21 = 1;
            v22 = 0;
            for ( i = v19 & (v20 + ((v17 >> 9) ^ (v17 >> 4))); ; i = v19 & v25 )
            {
              v24 = &v18->m128i_i64[3 * i];
              if ( v17 == *v24 && v20 == *((_DWORD *)v24 + 2) )
                break;
              if ( !*v24 )
              {
                v44 = *((_DWORD *)v24 + 2);
                if ( v44 == -1 )
                {
                  if ( v22 )
                    v24 = v22;
                  break;
                }
                if ( v44 == -2 && !v22 )
                  v22 = &v18->m128i_i64[3 * i];
              }
              v25 = v21 + i;
              ++v21;
            }
            *v24 = *(_QWORD *)result;
            *((_DWORD *)v24 + 2) = *(_DWORD *)(result + 8);
            *((_DWORD *)v24 + 4) = *(_DWORD *)(result + 16);
            a1->m128i_i32[2] = (2 * ((unsigned __int32)a1->m128i_i32[2] >> 1) + 2) | a1->m128i_i32[2] & 1;
          }
        }
        return result;
      }
      v26 = a1[1].m128i_u32[2];
    }
    else
    {
      if ( v4 )
      {
        v8 = 1536;
        v7 = 64;
        goto LABEL_5;
      }
      v26 = a1[1].m128i_u32[2];
      v8 = 1536;
      v7 = 64;
    }
    v41 = sub_22077B0(v8);
    a1[1].m128i_i32[2] = v7;
    a1[1].m128i_i64[0] = v41;
  }
  v14 = (a1->m128i_i64[1] & 1) == 0;
  a1->m128i_i64[1] &= 1uLL;
  v27 = v5 + 24LL * v26;
  if ( v14 )
  {
    v28 = (const __m128i *)a1[1].m128i_i64[0];
    v29 = 24LL * a1[1].m128i_u32[2];
  }
  else
  {
    v28 = a1 + 1;
    v29 = 192;
  }
  for ( j = (const __m128i *)((char *)v28 + v29); j != v28; v28 = (const __m128i *)((char *)v28 + 24) )
  {
    if ( v28 )
    {
      v28->m128i_i64[0] = 0;
      v28->m128i_i32[2] = -1;
    }
  }
  for ( result = v5; v27 != result; result += 24 )
  {
    v31 = *(_QWORD *)result;
    if ( *(_QWORD *)result || *(_DWORD *)(result + 8) <= 0xFFFFFFFD )
    {
      if ( (a1->m128i_i8[8] & 1) != 0 )
      {
        v32 = a1 + 1;
        v33 = 7;
      }
      else
      {
        v40 = a1[1].m128i_i32[2];
        v32 = (const __m128i *)a1[1].m128i_i64[0];
        if ( !v40 )
        {
LABEL_79:
          MEMORY[0] = *(_QWORD *)result;
          MEMORY[8] = *(_DWORD *)(result + 8);
          BUG();
        }
        v33 = v40 - 1;
      }
      v34 = *(_DWORD *)(result + 8);
      v35 = 1;
      v36 = 0;
      for ( k = v33 & (v34 + ((v31 >> 9) ^ (v31 >> 4))); ; k = v33 & v39 )
      {
        v38 = &v32->m128i_i64[3 * k];
        if ( v31 == *v38 && v34 == *((_DWORD *)v38 + 2) )
          break;
        if ( !*v38 )
        {
          v43 = *((_DWORD *)v38 + 2);
          if ( v43 == -1 )
          {
            if ( v36 )
              v38 = v36;
            break;
          }
          if ( !v36 && v43 == -2 )
            v36 = &v32->m128i_i64[3 * k];
        }
        v39 = v35 + k;
        ++v35;
      }
      *v38 = *(_QWORD *)result;
      *((_DWORD *)v38 + 2) = *(_DWORD *)(result + 8);
      *((_DWORD *)v38 + 4) = *(_DWORD *)(result + 16);
      a1->m128i_i32[2] = (2 * ((unsigned __int32)a1->m128i_i32[2] >> 1) + 2) | a1->m128i_i32[2] & 1;
    }
  }
  return j___libc_free_0(v5);
}
