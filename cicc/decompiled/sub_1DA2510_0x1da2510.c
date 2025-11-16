// Function: sub_1DA2510
// Address: 0x1da2510
//
__int64 __fastcall sub_1DA2510(const __m128i *a1, unsigned int a2)
{
  __int64 result; // rax
  char v4; // dl
  __int64 v5; // r12
  unsigned __int64 v6; // rax
  int v7; // r15d
  __int64 v8; // r14
  const __m128i *v9; // rsi
  const __m128i *v10; // rax
  __m128i *v11; // r12
  __m128i *v12; // rdx
  bool v13; // zf
  __int64 m128i_i64; // rcx
  __int64 v15; // r14
  __int64 v16; // rcx
  const __m128i *v17; // r8
  int v18; // esi
  __int64 v19; // rdi
  int v20; // r14d
  __int64 *v21; // r11
  __int64 v22; // r9
  unsigned __int64 v23; // r9
  unsigned __int64 v24; // r9
  unsigned int i; // eax
  __int64 *v26; // r9
  __int64 v27; // r10
  unsigned int v28; // eax
  unsigned __int32 v29; // r13d
  __int64 v30; // r9
  __m128i *v31; // rax
  __int64 v32; // rdx
  __m128i *j; // rdx
  __int64 k; // rdx
  const __m128i *v35; // r8
  int v36; // esi
  __int64 v37; // rdi
  __int64 *v38; // r14
  __int64 v39; // r11
  int v40; // r13d
  unsigned __int64 v41; // r11
  unsigned __int64 v42; // r11
  unsigned int m; // eax
  __int64 *v44; // r11
  __int64 v45; // r15
  __int32 v46; // esi
  __int64 v47; // rax
  unsigned int v48; // eax
  __int32 v49; // esi
  __int32 v50; // eax
  _BYTE v51[240]; // [rsp+10h] [rbp-F0h] BYREF

  result = a1->m128i_u8[8];
  v4 = a1->m128i_i8[8] & 1;
  if ( a2 <= 7 )
  {
    if ( v4 )
      return result;
    v5 = a1[1].m128i_i64[0];
    v29 = a1[1].m128i_u32[2];
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
        goto LABEL_5;
      v29 = a1[1].m128i_u32[2];
    }
    else
    {
      if ( v4 )
      {
        v8 = 1536;
        v7 = 64;
LABEL_5:
        v9 = a1 + 13;
        v10 = a1 + 1;
        v11 = (__m128i *)v51;
        while ( 1 )
        {
          while ( v10->m128i_i64[0] == -8 )
          {
            if ( v10->m128i_i64[1] != -8 )
              goto LABEL_7;
            v10 = (const __m128i *)((char *)v10 + 24);
            if ( v10 == v9 )
            {
LABEL_14:
              a1->m128i_i8[8] &= ~1u;
              result = sub_22077B0(v8);
              a1[1].m128i_i32[2] = v7;
              v12 = (__m128i *)v51;
              v13 = (a1->m128i_i64[1] & 1) == 0;
              a1->m128i_i64[1] &= 1uLL;
              m128i_i64 = result;
              a1[1].m128i_i64[0] = result;
              if ( !v13 )
              {
                m128i_i64 = (__int64)a1[1].m128i_i64;
                result = (__int64)a1[1].m128i_i64;
                v8 = 192;
              }
              v15 = result + v8;
              while ( 1 )
              {
                if ( m128i_i64 )
                {
                  *(_QWORD *)result = -8;
                  *(_QWORD *)(result + 8) = -8;
                }
                result += 24;
                if ( v15 == result )
                  break;
                m128i_i64 = result;
              }
              while ( 2 )
              {
                if ( v11 == v12 )
                  return result;
                v16 = v12->m128i_i64[0];
                if ( v12->m128i_i64[0] == -8 )
                {
                  if ( v12->m128i_i64[1] == -8 )
                  {
LABEL_77:
                    v12 = (__m128i *)((char *)v12 + 24);
                    continue;
                  }
                }
                else if ( v16 == -16 && v12->m128i_i64[1] == -16 )
                {
                  goto LABEL_77;
                }
                break;
              }
              if ( (a1->m128i_i8[8] & 1) != 0 )
              {
                v17 = a1 + 1;
                v18 = 7;
              }
              else
              {
                v49 = a1[1].m128i_i32[2];
                v17 = (const __m128i *)a1[1].m128i_i64[0];
                if ( !v49 )
                  goto LABEL_89;
                v18 = v49 - 1;
              }
              v19 = v12->m128i_i64[1];
              v20 = 1;
              v21 = 0;
              v22 = ((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4);
              v23 = (((v22 | ((unsigned __int64)(((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4)) << 32))
                    - 1
                    - (v22 << 32)) >> 22)
                  ^ ((v22 | ((unsigned __int64)(((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4)) << 32))
                   - 1
                   - (v22 << 32));
              v24 = ((9 * (((v23 - 1 - (v23 << 13)) >> 8) ^ (v23 - 1 - (v23 << 13)))) >> 15)
                  ^ (9 * (((v23 - 1 - (v23 << 13)) >> 8) ^ (v23 - 1 - (v23 << 13))));
              for ( i = v18 & (((v24 - 1 - (v24 << 27)) >> 31) ^ (v24 - 1 - ((_DWORD)v24 << 27))); ; i = v18 & v28 )
              {
                v26 = &v17->m128i_i64[3 * i];
                v27 = *v26;
                if ( v16 == *v26 && v26[1] == v19 )
                  break;
                if ( v27 == -8 )
                {
                  if ( v26[1] == -8 )
                  {
                    if ( v21 )
                      v26 = v21;
                    break;
                  }
                }
                else if ( v27 == -16 && v26[1] == -16 && !v21 )
                {
                  v21 = &v17->m128i_i64[3 * i];
                }
                v28 = v20 + i;
                ++v20;
              }
              v50 = v12[1].m128i_i32[0];
              *v26 = v16;
              v26[1] = v19;
              *((_DWORD *)v26 + 4) = v50;
              result = (2 * ((unsigned __int32)a1->m128i_i32[2] >> 1) + 2) | a1->m128i_i32[2] & 1;
              a1->m128i_i32[2] = result;
              goto LABEL_77;
            }
          }
          if ( v10->m128i_i64[0] != -16 || v10->m128i_i64[1] != -16 )
          {
LABEL_7:
            if ( v11 )
              *v11 = _mm_loadu_si128(v10);
            v11 = (__m128i *)((char *)v11 + 24);
            v11[-1].m128i_i32[2] = v10[1].m128i_i32[0];
          }
          v10 = (const __m128i *)((char *)v10 + 24);
          if ( v10 == v9 )
            goto LABEL_14;
        }
      }
      v29 = a1[1].m128i_u32[2];
      v8 = 1536;
      v7 = 64;
    }
    v47 = sub_22077B0(v8);
    a1[1].m128i_i32[2] = v7;
    a1[1].m128i_i64[0] = v47;
  }
  v13 = (a1->m128i_i64[1] & 1) == 0;
  a1->m128i_i64[1] &= 1uLL;
  v30 = v5 + 24LL * v29;
  if ( v13 )
  {
    v31 = (__m128i *)a1[1].m128i_i64[0];
    v32 = 24LL * a1[1].m128i_u32[2];
  }
  else
  {
    v31 = (__m128i *)&a1[1];
    v32 = 192;
  }
  for ( j = (__m128i *)((char *)v31 + v32); j != v31; v31 = (__m128i *)((char *)v31 + 24) )
  {
    if ( v31 )
    {
      v31->m128i_i64[0] = -8;
      v31->m128i_i64[1] = -8;
    }
  }
  for ( k = v5; v30 != k; k += 24 )
  {
    v16 = *(_QWORD *)k;
    if ( *(_QWORD *)k == -8 )
    {
      if ( *(_QWORD *)(k + 8) != -8 )
        goto LABEL_46;
    }
    else if ( v16 != -16 || *(_QWORD *)(k + 8) != -16 )
    {
LABEL_46:
      if ( (a1->m128i_i8[8] & 1) != 0 )
      {
        v35 = a1 + 1;
        v36 = 7;
      }
      else
      {
        v46 = a1[1].m128i_i32[2];
        v35 = (const __m128i *)a1[1].m128i_i64[0];
        if ( !v46 )
        {
LABEL_89:
          MEMORY[0] = v16;
          BUG();
        }
        v36 = v46 - 1;
      }
      v37 = *(_QWORD *)(k + 8);
      v38 = 0;
      v39 = ((unsigned int)v37 >> 9) ^ ((unsigned int)v37 >> 4);
      v40 = 1;
      v41 = (((v39 | ((unsigned __int64)(((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4)) << 32)) - 1 - (v39 << 32)) >> 22)
          ^ ((v39 | ((unsigned __int64)(((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4)) << 32)) - 1 - (v39 << 32));
      v42 = ((9 * (((v41 - 1 - (v41 << 13)) >> 8) ^ (v41 - 1 - (v41 << 13)))) >> 15)
          ^ (9 * (((v41 - 1 - (v41 << 13)) >> 8) ^ (v41 - 1 - (v41 << 13))));
      for ( m = v36 & (((v42 - 1 - (v42 << 27)) >> 31) ^ (v42 - 1 - ((_DWORD)v42 << 27))); ; m = v36 & v48 )
      {
        v44 = &v35->m128i_i64[3 * m];
        v45 = *v44;
        if ( v16 == *v44 && v44[1] == v37 )
          break;
        if ( v45 == -8 )
        {
          if ( v44[1] == -8 )
          {
            if ( v38 )
              v44 = v38;
            break;
          }
        }
        else if ( v45 == -16 && v44[1] == -16 && !v38 )
        {
          v38 = &v35->m128i_i64[3 * m];
        }
        v48 = v40 + m;
        ++v40;
      }
      *v44 = v16;
      v44[1] = *(_QWORD *)(k + 8);
      *((_DWORD *)v44 + 4) = *(_DWORD *)(k + 16);
      a1->m128i_i32[2] = (2 * ((unsigned __int32)a1->m128i_i32[2] >> 1) + 2) | a1->m128i_i32[2] & 1;
    }
  }
  return j___libc_free_0(v5);
}
