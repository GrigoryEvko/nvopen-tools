// Function: sub_2F76630
// Address: 0x2f76630
//
__int64 __fastcall sub_2F76630(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 *v6; // r12
  const __m128i *v7; // r12
  __int64 v9; // rax
  __int64 v10; // r8
  __int64 v11; // rcx
  __int64 v12; // rdi
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // r9
  __int64 v17; // rsi
  __int64 v18; // rcx
  __int64 v19; // r9
  __int64 v20; // rax
  __int64 v21; // rsi
  __int64 v22; // rax
  __int64 v23; // rcx
  unsigned __int64 v24; // rdx
  const __m128i *v25; // rax
  __int32 v26; // ecx
  __m128i v27; // xmm0
  __int64 v28; // rax
  __int64 v29; // rbx
  __int64 result; // rax
  unsigned __int64 v31; // r8
  __int64 v32; // rdx
  int *v33; // r14
  int *v34; // r15
  int *v35; // rbx
  __int64 v36; // rax
  __int64 v37; // rdx
  __int64 v38; // [rsp+20h] [rbp-90h]
  __int64 v41; // [rsp+38h] [rbp-78h]
  __int64 v42; // [rsp+40h] [rbp-70h]
  unsigned int v43; // [rsp+40h] [rbp-70h]
  bool v44; // [rsp+48h] [rbp-68h]
  __int64 v45; // [rsp+48h] [rbp-68h]
  unsigned __int64 v46; // [rsp+50h] [rbp-60h]
  unsigned __int64 v47; // [rsp+50h] [rbp-60h]
  int v48; // [rsp+50h] [rbp-60h]

  v6 = a1;
  if ( 3LL * *((unsigned int *)a1 + 54) )
  {
    v7 = (const __m128i *)a1[26];
    v46 = a4 & 0xFFFFFFFFFFFFFFF8LL | 6;
    v44 = a5 != 0;
    do
    {
      v9 = sub_2F73FB0(
             a2,
             a3,
             1,
             v7->m128i_i32[0],
             v46,
             (unsigned __int8 (__fastcall *)(__int64, __int64, __int64))sub_2F74570,
             -1,
             -1);
      v10 = v7->m128i_u32[0];
      v11 = v9;
      v12 = v7[1].m128i_i64[0];
      v14 = v13;
      v15 = v7->m128i_i64[1];
      if ( (int)v10 < 0 && v44 && !(v14 & ~v12 | v11 & ~v15) )
      {
        v41 = v11;
        v38 = v14;
        v43 = v7->m128i_i32[0];
        sub_2E8D840(a5, v10, 1);
        v15 = v7->m128i_i64[1];
        v12 = v7[1].m128i_i64[0];
        v10 = v43;
        v11 = v41;
        v14 = v38;
      }
      v16 = v11;
      v17 = v14 & v12;
      v18 = v15 & ~v11;
      v19 = v15 & v16;
      v20 = v12 & ~v14;
      if ( v20 | v18 )
      {
        v42 = v19;
        sub_2F747D0((__int64)(a1 + 52), v17, v15, v18, v10, v19, v10, v18, v20);
        v19 = v42;
      }
      if ( v17 | v19 )
      {
        v7->m128i_i64[1] = v19;
        v7[1].m128i_i64[0] = v17;
        v7 = (const __m128i *)((char *)v7 + 24);
        v28 = *((unsigned int *)a1 + 54);
        v21 = a1[26];
      }
      else
      {
        v21 = a1[26];
        v22 = *((unsigned int *)a1 + 54);
        v23 = v21 + 24 * v22 - ((_QWORD)v7 + 24);
        v24 = 0xAAAAAAAAAAAAAAABLL * (v23 >> 3);
        if ( v23 > 0 )
        {
          v25 = v7;
          do
          {
            v26 = v25[1].m128i_i32[2];
            v27 = _mm_loadu_si128(v25 + 2);
            v25 = (const __m128i *)((char *)v25 + 24);
            v25[-2].m128i_i32[2] = v26;
            v25[-1] = v27;
            --v24;
          }
          while ( v24 );
          LODWORD(v22) = *((_DWORD *)a1 + 54);
          v21 = a1[26];
        }
        v28 = (unsigned int)(v22 - 1);
        *((_DWORD *)a1 + 54) = v28;
      }
    }
    while ( v7 != (const __m128i *)(v21 + 24 * v28) );
    v6 = a1;
  }
  v29 = *v6;
  result = *v6 + 24LL * *((unsigned int *)v6 + 2);
  v45 = result;
  if ( result != *v6 )
  {
    v31 = a4 & 0xFFFFFFFFFFFFFFF8LL;
    do
    {
      v47 = v31;
      result = sub_2F73FB0(
                 a2,
                 a3,
                 1,
                 *(_DWORD *)v29,
                 v31,
                 (unsigned __int8 (__fastcall *)(__int64, __int64, __int64))sub_2F74570,
                 -1,
                 -1);
      *(_QWORD *)(v29 + 8) &= result;
      *(_QWORD *)(v29 + 16) &= v32;
      v29 += 24;
      v31 = v47;
    }
    while ( v45 != v29 );
  }
  if ( a5 )
  {
    v33 = (int *)v6[52];
    result = 3LL * *((unsigned int *)v6 + 106);
    v34 = &v33[6 * *((unsigned int *)v6 + 106)];
    if ( v34 != v33 )
    {
      v35 = (int *)v6[52];
      do
      {
        while ( 1 )
        {
          if ( *v35 < 0 )
          {
            v48 = *v35;
            v36 = sub_2F73FB0(
                    a2,
                    a3,
                    1,
                    *v35,
                    a4 & 0xFFFFFFFFFFFFFFF8LL | 6,
                    (unsigned __int8 (__fastcall *)(__int64, __int64, __int64))sub_2F74570,
                    -1,
                    -1);
            result = v37 | v36;
            if ( !result )
              break;
          }
          v35 += 6;
          if ( v34 == v35 )
            return result;
        }
        v35 += 6;
        result = sub_2E8D840(a5, v48, 1);
      }
      while ( v34 != v35 );
    }
  }
  return result;
}
