// Function: sub_19B90A0
// Address: 0x19b90a0
//
__int64 __fastcall sub_19B90A0(const __m128i *a1, unsigned int a2)
{
  __int64 result; // rax
  char v4; // dl
  __int64 *v5; // r12
  unsigned __int64 v6; // rax
  int v7; // r14d
  __int64 v8; // r15
  const __m128i *v9; // rax
  __m128i *v10; // r12
  __int64 v11; // r13
  bool v12; // zf
  __int64 *v13; // r8
  __m128i *v14; // rax
  __int64 v15; // rdx
  __m128i *j; // rdx
  __int64 *k; // rdx
  __int64 v18; // rcx
  const __m128i *v19; // r9
  int v20; // esi
  __int64 v21; // rdi
  const __m128i *v22; // r14
  __int64 v23; // r11
  int v24; // r13d
  unsigned __int64 v25; // r11
  unsigned __int64 v26; // r11
  int m; // eax
  __m128i *v28; // r11
  __int64 v29; // r15
  __int32 v30; // esi
  __int64 v31; // rax
  int v32; // eax
  __m128i *v33; // rdx
  __int64 m128i_i64; // rcx
  __int64 v35; // rdi
  const __m128i *v36; // r8
  int v37; // esi
  __int64 v38; // rdi
  int v39; // r11d
  __int64 v40; // r9
  const __m128i *v41; // r10
  unsigned __int64 v42; // r9
  unsigned __int64 v43; // r9
  int i; // eax
  __m128i *v45; // r9
  __int64 v46; // r14
  int v47; // eax
  __int32 v48; // esi
  _BYTE v49[112]; // [rsp+10h] [rbp-70h] BYREF

  result = a1->m128i_u8[8];
  v4 = a1->m128i_i8[8] & 1;
  if ( a2 <= 3 )
  {
    if ( v4 )
      return result;
    v5 = (__int64 *)a1[1].m128i_i64[0];
    v11 = a1[1].m128i_u32[2];
    a1->m128i_i8[8] = result | 1;
  }
  else
  {
    v5 = (__int64 *)a1[1].m128i_i64[0];
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
      v8 = 16LL * (unsigned int)v6;
      if ( v4 )
        goto LABEL_5;
      v11 = a1[1].m128i_u32[2];
    }
    else
    {
      if ( v4 )
      {
        v8 = 1024;
        v7 = 64;
LABEL_5:
        v9 = a1 + 1;
        v10 = (__m128i *)v49;
        while ( 1 )
        {
          if ( v9->m128i_i64[0] == -8 )
          {
            if ( v9->m128i_i64[1] != -8 )
              goto LABEL_7;
          }
          else if ( v9->m128i_i64[0] != -16 || v9->m128i_i64[1] != -16 )
          {
LABEL_7:
            if ( v10 )
              *v10 = _mm_loadu_si128(v9);
            ++v10;
          }
          if ( ++v9 == &a1[5] )
          {
            a1->m128i_i8[8] &= ~1u;
            result = sub_22077B0(v8);
            v12 = (a1->m128i_i64[1] & 1) == 0;
            a1->m128i_i64[1] &= 1uLL;
            v33 = (__m128i *)v49;
            a1[1].m128i_i64[0] = result;
            m128i_i64 = result;
            a1[1].m128i_i32[2] = v7;
            if ( !v12 )
            {
              m128i_i64 = (__int64)a1[1].m128i_i64;
              result = (__int64)a1[1].m128i_i64;
              v8 = 64;
            }
            v35 = result + v8;
            while ( 1 )
            {
              if ( m128i_i64 )
              {
                *(_QWORD *)result = -8;
                *(_QWORD *)(result + 8) = -8;
              }
              result += 16;
              if ( v35 == result )
                break;
              m128i_i64 = result;
            }
            while ( 2 )
            {
              if ( v10 == v33 )
                return result;
              v18 = v33->m128i_i64[0];
              if ( v33->m128i_i64[0] == -8 )
              {
                if ( v33->m128i_i64[1] == -8 )
                {
LABEL_77:
                  ++v33;
                  continue;
                }
              }
              else if ( v18 == -16 && v33->m128i_i64[1] == -16 )
              {
                goto LABEL_77;
              }
              break;
            }
            if ( (a1->m128i_i8[8] & 1) != 0 )
            {
              v36 = a1 + 1;
              v37 = 3;
            }
            else
            {
              v48 = a1[1].m128i_i32[2];
              v36 = (const __m128i *)a1[1].m128i_i64[0];
              if ( !v48 )
                goto LABEL_89;
              v37 = v48 - 1;
            }
            v38 = v33->m128i_i64[1];
            v39 = 1;
            v40 = ((unsigned int)v38 >> 9) ^ ((unsigned int)v38 >> 4);
            v41 = 0;
            v42 = (((v40 | ((unsigned __int64)(((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4)) << 32))
                  - 1
                  - (v40 << 32)) >> 22)
                ^ ((v40 | ((unsigned __int64)(((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4)) << 32))
                 - 1
                 - (v40 << 32));
            v43 = ((9 * (((v42 - 1 - (v42 << 13)) >> 8) ^ (v42 - 1 - (v42 << 13)))) >> 15)
                ^ (9 * (((v42 - 1 - (v42 << 13)) >> 8) ^ (v42 - 1 - (v42 << 13))));
            for ( i = v37 & (((v43 - 1 - (v43 << 27)) >> 31) ^ (v43 - 1 - ((_DWORD)v43 << 27))); ; i = v37 & v47 )
            {
              v45 = (__m128i *)&v36[i];
              v46 = v45->m128i_i64[0];
              if ( v18 == v45->m128i_i64[0] && v45->m128i_i64[1] == v38 )
                break;
              if ( v46 == -8 )
              {
                if ( v45->m128i_i64[1] == -8 )
                {
                  if ( v41 )
                    v45 = (__m128i *)v41;
                  break;
                }
              }
              else if ( v46 == -16 && v45->m128i_i64[1] == -16 && !v41 )
              {
                v41 = &v36[i];
              }
              v47 = v39 + i;
              ++v39;
            }
            v45->m128i_i64[0] = v18;
            v45->m128i_i64[1] = v38;
            result = (2 * ((unsigned __int32)a1->m128i_i32[2] >> 1) + 2) | a1->m128i_i32[2] & 1;
            a1->m128i_i32[2] = result;
            goto LABEL_77;
          }
        }
      }
      v11 = a1[1].m128i_u32[2];
      v8 = 1024;
      v7 = 64;
    }
    v31 = sub_22077B0(v8);
    a1[1].m128i_i32[2] = v7;
    a1[1].m128i_i64[0] = v31;
  }
  v12 = (a1->m128i_i64[1] & 1) == 0;
  a1->m128i_i64[1] &= 1uLL;
  v13 = &v5[2 * v11];
  if ( v12 )
  {
    v14 = (__m128i *)a1[1].m128i_i64[0];
    v15 = a1[1].m128i_u32[2];
  }
  else
  {
    v14 = (__m128i *)&a1[1];
    v15 = 4;
  }
  for ( j = &v14[v15]; j != v14; ++v14 )
  {
    if ( v14 )
    {
      v14->m128i_i64[0] = -8;
      v14->m128i_i64[1] = -8;
    }
  }
  for ( k = v5; v13 != k; k += 2 )
  {
    v18 = *k;
    if ( *k == -8 )
    {
      if ( k[1] != -8 )
        goto LABEL_26;
    }
    else if ( v18 != -16 || k[1] != -16 )
    {
LABEL_26:
      if ( (a1->m128i_i8[8] & 1) != 0 )
      {
        v19 = a1 + 1;
        v20 = 3;
      }
      else
      {
        v30 = a1[1].m128i_i32[2];
        v19 = (const __m128i *)a1[1].m128i_i64[0];
        if ( !v30 )
        {
LABEL_89:
          MEMORY[0] = v18;
          BUG();
        }
        v20 = v30 - 1;
      }
      v21 = k[1];
      v22 = 0;
      v23 = ((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4);
      v24 = 1;
      v25 = (((v23 | ((unsigned __int64)(((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4)) << 32)) - 1 - (v23 << 32)) >> 22)
          ^ ((v23 | ((unsigned __int64)(((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4)) << 32)) - 1 - (v23 << 32));
      v26 = ((9 * (((v25 - 1 - (v25 << 13)) >> 8) ^ (v25 - 1 - (v25 << 13)))) >> 15)
          ^ (9 * (((v25 - 1 - (v25 << 13)) >> 8) ^ (v25 - 1 - (v25 << 13))));
      for ( m = v20 & (((v26 - 1 - (v26 << 27)) >> 31) ^ (v26 - 1 - ((_DWORD)v26 << 27))); ; m = v20 & v32 )
      {
        v28 = (__m128i *)&v19[m];
        v29 = v28->m128i_i64[0];
        if ( v18 == v28->m128i_i64[0] && v28->m128i_i64[1] == v21 )
          break;
        if ( v29 == -8 )
        {
          if ( v28->m128i_i64[1] == -8 )
          {
            if ( v22 )
              v28 = (__m128i *)v22;
            break;
          }
        }
        else if ( v29 == -16 && v28->m128i_i64[1] == -16 && !v22 )
        {
          v22 = &v19[m];
        }
        v32 = v24 + m;
        ++v24;
      }
      v28->m128i_i64[0] = v18;
      v28->m128i_i64[1] = k[1];
      a1->m128i_i32[2] = (2 * ((unsigned __int32)a1->m128i_i32[2] >> 1) + 2) | a1->m128i_i32[2] & 1;
    }
  }
  return j___libc_free_0(v5);
}
