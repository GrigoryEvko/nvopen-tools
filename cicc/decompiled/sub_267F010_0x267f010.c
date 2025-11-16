// Function: sub_267F010
// Address: 0x267f010
//
_QWORD *__fastcall sub_267F010(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v4; // rbx
  __int64 v5; // r14
  unsigned int v6; // eax
  _QWORD *result; // rax
  __m128i *v8; // r13
  _QWORD *i; // rdx
  __m128i *v10; // r14
  __int64 v11; // rax
  int v12; // edx
  __int64 v13; // rcx
  __int64 v14; // r8
  int v15; // r9d
  unsigned int v16; // edx
  __int64 *v17; // rdi
  __int64 *v18; // r15
  __int64 v19; // rsi
  __m128i *v20; // rbx
  __int64 v21; // r9
  __int64 v22; // rbx
  unsigned __int64 v23; // r15
  void (__fastcall *v24)(unsigned __int64, unsigned __int64, __int64, __int64, __int64, __int64); // rax
  __m128i *v25; // rax
  __m128i *v26; // rax
  __int64 v27; // rcx
  __m128i *v28; // rsi
  __m128i v29; // xmm0
  __int64 v30; // rdx
  __int64 v31; // rdx
  __int64 v32; // r15
  __int64 v33; // rbx
  void (__fastcall *v34)(__int64, __int64, __int64); // rax
  unsigned __int64 v35; // rdi
  int v36; // eax
  __m128i *v37; // rdx
  _QWORD *j; // rdx
  __int64 v39; // [rsp+8h] [rbp-68h]
  unsigned __int32 v40; // [rsp+10h] [rbp-60h]
  int v41; // [rsp+10h] [rbp-60h]
  __m128i *v42; // [rsp+18h] [rbp-58h]
  __m128i *v43; // [rsp+20h] [rbp-50h]
  unsigned int v44; // [rsp+20h] [rbp-50h]
  __int64 v45; // [rsp+28h] [rbp-48h]
  unsigned __int64 v46[7]; // [rsp+38h] [rbp-38h] BYREF

  v2 = (unsigned int)(a2 - 1);
  v4 = *(unsigned int *)(a1 + 24);
  v5 = *(_QWORD *)(a1 + 8);
  v45 = v5;
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
  result = (_QWORD *)sub_C7D670(56LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v39 = 56 * v4;
    v8 = (__m128i *)(56 * v4 + v5);
    for ( i = &result[7 * *(unsigned int *)(a1 + 24)]; i != result; result += 7 )
    {
      if ( result )
        *result = -4096;
    }
    v10 = (__m128i *)(v5 + 24);
    if ( v8 != (__m128i *)v45 )
    {
      while ( 1 )
      {
        v11 = v10[-2].m128i_i64[1];
        if ( v11 != -8192 && v11 != -4096 )
        {
          v12 = *(_DWORD *)(a1 + 24);
          if ( !v12 )
          {
            MEMORY[0] = v10[-2].m128i_i64[1];
            BUG();
          }
          v13 = (unsigned int)(v12 - 1);
          v14 = *(_QWORD *)(a1 + 8);
          v15 = 1;
          v16 = v13 & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
          v17 = 0;
          v18 = (__int64 *)(v14 + 56LL * v16);
          v19 = *v18;
          if ( v11 != *v18 )
          {
            while ( v19 != -4096 )
            {
              if ( !v17 && v19 == -8192 )
                v17 = v18;
              v16 = v13 & (v15 + v16);
              v18 = (__int64 *)(v14 + 56LL * v16);
              v19 = *v18;
              if ( v11 == *v18 )
                goto LABEL_13;
              ++v15;
            }
            if ( v17 )
              v18 = v17;
          }
LABEL_13:
          *v18 = v11;
          v20 = (__m128i *)(v18 + 3);
          v18[2] = 0x100000000LL;
          v18[1] = (__int64)(v18 + 3);
          v21 = v10[-1].m128i_u32[2];
          if ( v18 + 1 != (__int64 *)&v10[-1] && (_DWORD)v21 )
          {
            v25 = (__m128i *)v10[-1].m128i_i64[0];
            if ( v25 == v10 )
            {
              v26 = v10;
              v27 = 1;
              if ( (_DWORD)v21 != 1 )
              {
                v40 = v10[-1].m128i_u32[2];
                v43 = (__m128i *)sub_C8D7D0((__int64)(v18 + 1), (__int64)(v18 + 3), (unsigned int)v21, 0x20u, v46, v21);
                sub_26780F0((__int64)(v18 + 1), v43);
                v35 = v18[1];
                v36 = v46[0];
                v37 = v43;
                v21 = v40;
                if ( v20 != (__m128i *)v35 )
                {
                  v41 = v46[0];
                  v42 = v43;
                  v44 = v21;
                  _libc_free(v35);
                  v36 = v41;
                  v37 = v42;
                  v21 = v44;
                }
                v18[1] = (__int64)v37;
                v20 = v37;
                *((_DWORD *)v18 + 5) = v36;
                v26 = (__m128i *)v10[-1].m128i_i64[0];
                v27 = v10[-1].m128i_u32[2];
              }
              v13 = 32 * v27;
              v28 = (__m128i *)((char *)v20 + v13);
              if ( v13 )
              {
                do
                {
                  if ( v20 )
                  {
                    v20[1].m128i_i64[0] = 0;
                    v29 = _mm_loadu_si128(v26);
                    *v26 = _mm_loadu_si128(v20);
                    *v20 = v29;
                    v30 = v26[1].m128i_i64[0];
                    v26[1].m128i_i64[0] = 0;
                    v13 = v20[1].m128i_i64[1];
                    v20[1].m128i_i64[0] = v30;
                    v31 = v26[1].m128i_i64[1];
                    v26[1].m128i_i64[1] = v13;
                    v20[1].m128i_i64[1] = v31;
                  }
                  v20 += 2;
                  v26 += 2;
                }
                while ( v20 != v28 );
              }
              *((_DWORD *)v18 + 4) = v21;
              v32 = v10[-1].m128i_i64[0];
              v33 = v32 + 32LL * v10[-1].m128i_u32[2];
              while ( v32 != v33 )
              {
                v34 = *(void (__fastcall **)(__int64, __int64, __int64))(v33 - 16);
                v33 -= 32;
                if ( v34 )
                  v34(v33, v33, 3);
              }
              v10[-1].m128i_i32[2] = 0;
            }
            else
            {
              v18[1] = (__int64)v25;
              *((_DWORD *)v18 + 4) = v10[-1].m128i_i32[2];
              *((_DWORD *)v18 + 5) = v10[-1].m128i_i32[3];
              v10[-1].m128i_i64[0] = (__int64)v10;
              v10[-1].m128i_i32[3] = 0;
              v10[-1].m128i_i32[2] = 0;
            }
          }
          ++*(_DWORD *)(a1 + 16);
          v22 = v10[-1].m128i_i64[0];
          v23 = v22 + 32LL * v10[-1].m128i_u32[2];
          if ( v22 != v23 )
          {
            do
            {
              v24 = *(void (__fastcall **)(unsigned __int64, unsigned __int64, __int64, __int64, __int64, __int64))(v23 - 16);
              v23 -= 32LL;
              if ( v24 )
                v24(v23, v23, 3, v13, v14, v21);
            }
            while ( v22 != v23 );
            v23 = v10[-1].m128i_u64[0];
          }
          if ( v10 != (__m128i *)v23 )
            _libc_free(v23);
        }
        if ( v8 == &v10[2] )
          break;
        v10 = (__m128i *)((char *)v10 + 56);
      }
    }
    return (_QWORD *)sub_C7D6A0(v45, v39, 8);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[7 * *(unsigned int *)(a1 + 24)]; j != result; result += 7 )
    {
      if ( result )
        *result = -4096;
    }
  }
  return result;
}
