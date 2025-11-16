// Function: sub_1EF8700
// Address: 0x1ef8700
//
const __m128i *__fastcall sub_1EF8700(const __m128i *a1, const __m128i *a2, const __m128i *a3)
{
  __int64 v3; // rbx
  __int64 v4; // r13
  __int64 v5; // r14
  const __m128i *v6; // r15
  __m128i *v7; // r12
  __int64 v8; // rcx
  __int64 v9; // r13
  __int32 v10; // ebx
  __int64 v11; // r8
  __int32 v12; // edi
  __int32 v13; // esi
  __int64 *v14; // rax
  const __m128i *v15; // r15
  __int64 v16; // r12
  __int64 v17; // r14
  __int64 v18; // rcx
  __int64 v19; // rbx
  __int32 v20; // r13d
  __int64 v21; // r8
  __int32 v22; // edi
  __int32 v23; // esi
  unsigned __int64 v24; // rdi
  const __m128i *v26; // r12
  __m128i *v27; // rbx
  __int64 v28; // r15
  __int64 v29; // r14
  __int32 v30; // r13d
  __int64 v31; // rsi
  __int32 v32; // ecx
  __int32 v33; // edx
  __int64 v34; // [rsp+0h] [rbp-60h]
  __int64 v35; // [rsp+8h] [rbp-58h]
  __int64 v36; // [rsp+8h] [rbp-58h]
  __m128i *v37; // [rsp+10h] [rbp-50h]
  signed __int64 v38; // [rsp+18h] [rbp-48h]
  __int64 v39; // [rsp+20h] [rbp-40h]
  __int64 v40; // [rsp+20h] [rbp-40h]
  unsigned __int64 v41; // [rsp+20h] [rbp-40h]
  __int64 v42; // [rsp+28h] [rbp-38h]
  __int64 v43; // [rsp+28h] [rbp-38h]
  __int64 v44; // [rsp+28h] [rbp-38h]

  v37 = (__m128i *)a1;
  if ( a1 == a2 )
    return a3;
  if ( a2 == a3 )
    return a1;
  v34 = (__int64)a1->m128i_i64 + (char *)a3 - (char *)a2;
  v3 = 0xCCCCCCCCCCCCCCCDLL * (((char *)a2 - (char *)a1) >> 3);
  v38 = 0xCCCCCCCCCCCCCCCDLL * (((char *)a3 - (char *)a1) >> 3);
  if ( v3 == v38 - v3 )
  {
    v26 = a2 + 1;
    v27 = (__m128i *)&a1[1];
    v41 = (0xCCCCCCCCCCCCCCDLL * ((unsigned __int64)((char *)&a2[-3].m128i_u64[1] - (char *)a1) >> 3))
        & 0x1FFFFFFFFFFFFFFFLL;
    v44 = (__int64)&a2[3].m128i_i64[5 * v41 + 1];
    do
    {
      v28 = v27->m128i_i64[0];
      v29 = v27->m128i_i64[1];
      v27->m128i_i64[0] = 0;
      v30 = v27[1].m128i_i32[0];
      v31 = v27[-1].m128i_i64[0];
      v27->m128i_i64[1] = 0;
      v32 = v27[-1].m128i_i32[2];
      v33 = v27[-1].m128i_i32[3];
      v27[1].m128i_i32[0] = 0;
      v27[-1].m128i_i64[0] = v26[-1].m128i_i64[0];
      v27[-1].m128i_i32[2] = v26[-1].m128i_i32[2];
      v27[-1].m128i_i32[3] = v26[-1].m128i_i32[3];
      if ( v27 != v26 )
      {
        *v27 = _mm_loadu_si128(v26);
        v27[1].m128i_i32[0] = v26[1].m128i_i32[0];
      }
      v26[-1].m128i_i64[0] = v31;
      v26 = (const __m128i *)((char *)v26 + 40);
      v27 = (__m128i *)((char *)v27 + 40);
      v26[-3].m128i_i32[0] = v32;
      v26[-3].m128i_i32[1] = v33;
      _libc_free(0);
      v26[-3].m128i_i64[1] = v28;
      v26[-2].m128i_i64[0] = v29;
      v26[-2].m128i_i32[2] = v30;
    }
    while ( v26 != (const __m128i *)v44 );
    return (const __m128i *)((char *)a1 + 40 * v41 + 40);
  }
  else
  {
    v4 = v38 - v3;
    if ( v3 >= v38 - v3 )
      goto LABEL_14;
    while ( 1 )
    {
      if ( v4 > 0 )
      {
        v39 = v4;
        v5 = 0;
        v35 = v3;
        v6 = (__m128i *)((char *)v37 + 40 * v3 + 16);
        v7 = v37 + 1;
        do
        {
          v8 = v7->m128i_i64[0];
          v9 = v7->m128i_i64[1];
          v7->m128i_i64[0] = 0;
          v7->m128i_i64[1] = 0;
          v10 = v7[1].m128i_i32[0];
          v7[1].m128i_i32[0] = 0;
          v11 = v7[-1].m128i_i64[0];
          v12 = v7[-1].m128i_i32[2];
          v7[-1].m128i_i64[0] = v6[-1].m128i_i64[0];
          v13 = v7[-1].m128i_i32[3];
          v7[-1].m128i_i32[2] = v6[-1].m128i_i32[2];
          v7[-1].m128i_i32[3] = v6[-1].m128i_i32[3];
          if ( v7 != v6 )
          {
            *v7 = _mm_loadu_si128(v6);
            v7[1].m128i_i32[0] = v6[1].m128i_i32[0];
          }
          v6[-1].m128i_i32[2] = v12;
          ++v5;
          v6 = (const __m128i *)((char *)v6 + 40);
          v6[-4].m128i_i64[1] = v11;
          v7 = (__m128i *)((char *)v7 + 40);
          v6[-3].m128i_i32[1] = v13;
          v42 = v8;
          _libc_free(0);
          v6[-2].m128i_i64[0] = v9;
          v6[-2].m128i_i32[2] = v10;
          v6[-3].m128i_i64[1] = v42;
        }
        while ( v39 != v5 );
        v3 = v35;
        v37 = (__m128i *)((char *)v37 + 40 * v39);
      }
      if ( !(v38 % v3) )
        break;
      v4 = v3;
      v3 -= v38 % v3;
      while ( 1 )
      {
        v38 = v4;
        v4 -= v3;
        if ( v3 < v4 )
          break;
LABEL_14:
        v14 = &v37->m128i_i64[5 * v38];
        v37 = (__m128i *)&v14[-5 * v4];
        if ( v3 > 0 )
        {
          v36 = v4;
          v15 = (const __m128i *)(v14 - 3);
          v16 = (__int64)&v14[-5 * v4 - 3];
          v17 = 0;
          v40 = v3;
          do
          {
            v18 = *(_QWORD *)v16;
            v19 = *(_QWORD *)(v16 + 8);
            *(_QWORD *)v16 = 0;
            *(_QWORD *)(v16 + 8) = 0;
            v20 = *(_DWORD *)(v16 + 16);
            *(_DWORD *)(v16 + 16) = 0;
            v21 = *(_QWORD *)(v16 - 16);
            v22 = *(_DWORD *)(v16 - 8);
            *(_QWORD *)(v16 - 16) = v15[-1].m128i_i64[0];
            v23 = *(_DWORD *)(v16 - 4);
            *(_DWORD *)(v16 - 8) = v15[-1].m128i_i32[2];
            *(_DWORD *)(v16 - 4) = v15[-1].m128i_i32[3];
            if ( (const __m128i *)v16 != v15 )
            {
              *(__m128i *)v16 = _mm_loadu_si128(v15);
              *(_DWORD *)(v16 + 16) = v15[1].m128i_i32[0];
              v15->m128i_i64[0] = 0;
            }
            v15[-1].m128i_i32[2] = v22;
            v24 = v15->m128i_i64[0];
            ++v17;
            v15 = (const __m128i *)((char *)v15 - 40);
            v15[1].m128i_i64[1] = v21;
            v16 -= 40;
            v15[2].m128i_i32[1] = v23;
            v43 = v18;
            _libc_free(v24);
            v15[3].m128i_i64[0] = v19;
            v15[3].m128i_i32[2] = v20;
            v15[2].m128i_i64[1] = v43;
          }
          while ( v40 != v17 );
          v4 = v36;
          v37 = (__m128i *)((char *)v37 - 40 * v40);
        }
        v3 = v38 % v4;
        if ( !(v38 % v4) )
          return (const __m128i *)v34;
      }
    }
  }
  return (const __m128i *)v34;
}
