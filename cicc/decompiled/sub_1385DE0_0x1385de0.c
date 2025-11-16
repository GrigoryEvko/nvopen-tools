// Function: sub_1385DE0
// Address: 0x1385de0
//
__int64 __fastcall sub_1385DE0(__int64 a1, int a2)
{
  __int64 v3; // rbx
  const __m128i *v4; // r12
  unsigned __int64 v5; // rax
  __int64 result; // rax
  __int64 v7; // rdx
  const __m128i *v8; // rbx
  __int64 i; // rdx
  const __m128i *v10; // r13
  __int64 v11; // rdx
  int v12; // ecx
  __int32 v13; // esi
  int v14; // ecx
  __int64 v15; // rdi
  __int64 *v16; // r11
  int v17; // r10d
  unsigned __int64 v18; // r8
  unsigned __int64 v19; // r8
  unsigned int j; // r9d
  __int64 *v21; // rax
  __int64 v22; // r8
  __m128i v23; // xmm1
  __int64 v24; // rcx
  __int64 v25; // rdx
  __int64 v26; // rdx
  __int64 k; // rdx
  unsigned int v28; // eax

  v3 = *(unsigned int *)(a1 + 24);
  v4 = *(const __m128i **)(a1 + 8);
  v5 = ((((((((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
            | (unsigned int)(a2 - 1)
            | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
          | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
          | (unsigned int)(a2 - 1)
          | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 8)
        | (((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
          | (unsigned int)(a2 - 1)
          | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
        | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
        | (unsigned int)(a2 - 1)
        | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 16)
      | (((((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
          | (unsigned int)(a2 - 1)
          | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
        | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
        | (unsigned int)(a2 - 1)
        | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 8)
      | (((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
        | (unsigned int)(a2 - 1)
        | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
      | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
      | (unsigned int)(a2 - 1)
      | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1))
     + 1;
  if ( (unsigned int)v5 < 0x40 )
    LODWORD(v5) = 64;
  *(_DWORD *)(a1 + 24) = v5;
  result = sub_22077B0(48LL * (unsigned int)v5);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    v7 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v8 = &v4[3 * v3];
    for ( i = result + 48 * v7; i != result; result += 48 )
    {
      if ( result )
      {
        *(_QWORD *)result = -8;
        *(_DWORD *)(result + 8) = -1;
      }
    }
    if ( v8 != v4 )
    {
      v10 = v4;
      while ( 1 )
      {
        v11 = v10->m128i_i64[0];
        if ( v10->m128i_i64[0] == -8 )
        {
          if ( v10->m128i_i32[2] != -1 )
            goto LABEL_12;
          v10 += 3;
          if ( v8 == v10 )
            return j___libc_free_0(v4);
        }
        else if ( v11 == -16 && v10->m128i_i32[2] == -2 )
        {
          v10 += 3;
          if ( v8 == v10 )
            return j___libc_free_0(v4);
        }
        else
        {
LABEL_12:
          v12 = *(_DWORD *)(a1 + 24);
          if ( !v12 )
          {
            MEMORY[0] = _mm_loadu_si128(v10);
            BUG();
          }
          v13 = v10->m128i_i32[2];
          v14 = v12 - 1;
          v16 = 0;
          v17 = 1;
          v18 = ((((unsigned int)(37 * v13)
                 | ((unsigned __int64)(((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4)) << 32))
                - 1
                - ((unsigned __int64)(unsigned int)(37 * v13) << 32)) >> 22)
              ^ (((unsigned int)(37 * v13)
                | ((unsigned __int64)(((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4)) << 32))
               - 1
               - ((unsigned __int64)(unsigned int)(37 * v13) << 32));
          v19 = ((9 * (((v18 - 1 - (v18 << 13)) >> 8) ^ (v18 - 1 - (v18 << 13)))) >> 15)
              ^ (9 * (((v18 - 1 - (v18 << 13)) >> 8) ^ (v18 - 1 - (v18 << 13))));
          for ( j = v14 & (((v19 - 1 - (v19 << 27)) >> 31) ^ (v19 - 1 - ((_DWORD)v19 << 27))); ; j = v14 & v28 )
          {
            v15 = *(_QWORD *)(a1 + 8);
            v21 = (__int64 *)(v15 + 48LL * j);
            v22 = *v21;
            if ( v11 == *v21 && v13 == *((_DWORD *)v21 + 2) )
              break;
            if ( v22 == -8 )
            {
              if ( *((_DWORD *)v21 + 2) == -1 )
              {
                if ( v16 )
                  v21 = v16;
                break;
              }
            }
            else if ( v22 == -16 && *((_DWORD *)v21 + 2) == -2 && !v16 )
            {
              v16 = (__int64 *)(v15 + 48LL * j);
            }
            v28 = j + v17++;
          }
          v23 = _mm_loadu_si128(v10);
          v21[4] = 0;
          v21[3] = 0;
          *((_DWORD *)v21 + 10) = 0;
          v21[2] = 1;
          *(__m128i *)v21 = v23;
          v24 = v10[1].m128i_i64[1];
          ++v10[1].m128i_i64[0];
          v25 = v21[3];
          v10 += 3;
          v21[3] = v24;
          LODWORD(v24) = v10[-1].m128i_i32[0];
          v10[-2].m128i_i64[1] = v25;
          LODWORD(v25) = *((_DWORD *)v21 + 8);
          *((_DWORD *)v21 + 8) = v24;
          LODWORD(v24) = v10[-1].m128i_i32[1];
          v10[-1].m128i_i32[0] = v25;
          LODWORD(v25) = *((_DWORD *)v21 + 9);
          *((_DWORD *)v21 + 9) = v24;
          LODWORD(v24) = v10[-1].m128i_i32[2];
          v10[-1].m128i_i32[1] = v25;
          LODWORD(v25) = *((_DWORD *)v21 + 10);
          *((_DWORD *)v21 + 10) = v24;
          v10[-1].m128i_i32[2] = v25;
          ++*(_DWORD *)(a1 + 16);
          j___libc_free_0(v10[-2].m128i_i64[1]);
          if ( v8 == v10 )
            return j___libc_free_0(v4);
        }
      }
    }
    return j___libc_free_0(v4);
  }
  else
  {
    v26 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = result + 48 * v26; k != result; result += 48 )
    {
      if ( result )
      {
        *(_QWORD *)result = -8;
        *(_DWORD *)(result + 8) = -1;
      }
    }
  }
  return result;
}
