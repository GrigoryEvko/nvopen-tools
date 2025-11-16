// Function: sub_262D6A0
// Address: 0x262d6a0
//
void __fastcall sub_262D6A0(__int64 a1, unsigned __int64 a2)
{
  const __m128i *v3; // r15
  __int64 v4; // rbx
  char *v5; // rdx
  __int64 v6; // rax
  bool v7; // cf
  unsigned __int64 v8; // rax
  __int8 *v9; // rbx
  __m128i *v10; // rbx
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // r12
  unsigned __int64 v17; // r13
  unsigned __int64 v18; // rdi
  unsigned __int64 v19; // rdi
  unsigned __int64 v20; // rdi
  unsigned __int64 v21; // rdi
  unsigned __int64 v22; // rdi
  __int64 v23; // r12
  unsigned __int64 v24; // r13
  unsigned __int64 v25; // rdi
  unsigned __int64 v26; // r12
  __int64 v27; // rax
  unsigned __int64 v28; // [rsp+8h] [rbp-58h]
  unsigned __int64 v29; // [rsp+10h] [rbp-50h]
  __m128i *v30; // [rsp+18h] [rbp-48h]
  unsigned __int64 v31; // [rsp+20h] [rbp-40h]
  const __m128i *v32; // [rsp+28h] [rbp-38h]

  v31 = a2;
  if ( !a2 )
    return;
  v3 = *(const __m128i **)a1;
  v32 = *(const __m128i **)(a1 + 8);
  v4 = (__int64)v32->m128i_i64 - *(_QWORD *)a1;
  v29 = 0x2E8BA2E8BA2E8BA3LL * (v4 >> 4);
  if ( a2 <= 0x2E8BA2E8BA2E8BA3LL * ((__int64)(*(_QWORD *)(a1 + 16) - (_QWORD)v32) >> 4) )
  {
    v5 = *(char **)(a1 + 8);
    do
    {
      if ( v5 )
        memset(v5, 0, 0xB0u);
      v5 += 176;
      --a2;
    }
    while ( a2 );
    *(_QWORD *)(a1 + 8) = &v32[11 * v31];
    return;
  }
  if ( 0xBA2E8BA2E8BA2ELL - v29 < a2 )
    sub_4262D8((__int64)"vector::_M_default_append");
  v6 = 0x2E8BA2E8BA2E8BA3LL * (((__int64)v32->m128i_i64 - *(_QWORD *)a1) >> 4);
  if ( a2 >= v29 )
    v6 = a2;
  v7 = __CFADD__(v29, v6);
  v8 = v29 + v6;
  if ( v7 )
  {
    v26 = 0x7FFFFFFFFFFFFFA0LL;
  }
  else
  {
    if ( !v8 )
    {
      v28 = 0;
      v30 = 0;
      goto LABEL_15;
    }
    if ( v8 > 0xBA2E8BA2E8BA2ELL )
      v8 = 0xBA2E8BA2E8BA2ELL;
    v26 = 176 * v8;
  }
  v27 = sub_22077B0(v26);
  v3 = *(const __m128i **)a1;
  v30 = (__m128i *)v27;
  v32 = *(const __m128i **)(a1 + 8);
  v28 = v27 + v26;
LABEL_15:
  v9 = &v30->m128i_i8[v4];
  do
  {
    if ( v9 )
      memset(v9, 0, 0xB0u);
    v9 += 176;
    --a2;
  }
  while ( a2 );
  if ( v32 != v3 )
  {
    v10 = v30;
    do
    {
      if ( v10 )
      {
        v10->m128i_i32[0] = v3->m128i_i32[0];
        v10->m128i_i32[1] = v3->m128i_i32[1];
        v10->m128i_i8[8] = v3->m128i_i8[8];
        v10->m128i_i8[9] = v3->m128i_i8[9];
        v10->m128i_i8[10] = v3->m128i_i8[10];
        v10->m128i_i8[11] = v3->m128i_i8[11];
        v10->m128i_i32[3] = v3->m128i_i32[3];
        v10[1] = _mm_loadu_si128(v3 + 1);
        v10[2].m128i_i64[0] = v3[2].m128i_i64[0];
        v10[2].m128i_i64[1] = v3[2].m128i_i64[1];
        v10[3].m128i_i64[0] = v3[3].m128i_i64[0];
        v11 = v3[3].m128i_i64[1];
        v3[3].m128i_i64[0] = 0;
        v3[2].m128i_i64[1] = 0;
        v3[2].m128i_i64[0] = 0;
        v10[3].m128i_i64[1] = v11;
        v10[4].m128i_i64[0] = v3[4].m128i_i64[0];
        v10[4].m128i_i64[1] = v3[4].m128i_i64[1];
        v3[4].m128i_i64[1] = 0;
        v3[4].m128i_i64[0] = 0;
        v12 = v3[5].m128i_i64[0];
        v3[3].m128i_i64[1] = 0;
        v10[5].m128i_i64[0] = v12;
        v10[5].m128i_i64[1] = v3[5].m128i_i64[1];
        v10[6].m128i_i64[0] = v3[6].m128i_i64[0];
        v13 = v3[6].m128i_i64[1];
        v3[6].m128i_i64[0] = 0;
        v3[5].m128i_i64[1] = 0;
        v3[5].m128i_i64[0] = 0;
        v10[6].m128i_i64[1] = v13;
        v10[7].m128i_i64[0] = v3[7].m128i_i64[0];
        v10[7].m128i_i64[1] = v3[7].m128i_i64[1];
        v14 = v3[8].m128i_i64[0];
        v3[7].m128i_i64[1] = 0;
        v3[7].m128i_i64[0] = 0;
        v3[6].m128i_i64[1] = 0;
        v10[8].m128i_i64[0] = v14;
        v10[8].m128i_i64[1] = v3[8].m128i_i64[1];
        v10[9].m128i_i64[0] = v3[9].m128i_i64[0];
        v15 = v3[9].m128i_i64[1];
        v3[9].m128i_i64[0] = 0;
        v3[8].m128i_i64[1] = 0;
        v3[8].m128i_i64[0] = 0;
        v10[9].m128i_i64[1] = v15;
        v10[10].m128i_i64[0] = v3[10].m128i_i64[0];
        v10[10].m128i_i64[1] = v3[10].m128i_i64[1];
        v3[10].m128i_i64[1] = 0;
        v3[10].m128i_i64[0] = 0;
        v3[9].m128i_i64[1] = 0;
      }
      else
      {
        v23 = v3[10].m128i_i64[0];
        v24 = v3[9].m128i_u64[1];
        if ( v23 != v24 )
        {
          do
          {
            v25 = *(_QWORD *)(v24 + 16);
            if ( v25 )
              j_j___libc_free_0(v25);
            v24 += 40LL;
          }
          while ( v23 != v24 );
          v24 = v3[9].m128i_u64[1];
        }
        if ( v24 )
          j_j___libc_free_0(v24);
      }
      v16 = v3[8].m128i_i64[1];
      v17 = v3[8].m128i_u64[0];
      if ( v16 != v17 )
      {
        do
        {
          v18 = *(_QWORD *)(v17 + 16);
          if ( v18 )
            j_j___libc_free_0(v18);
          v17 += 40LL;
        }
        while ( v16 != v17 );
        v17 = v3[8].m128i_u64[0];
      }
      if ( v17 )
        j_j___libc_free_0(v17);
      v19 = v3[6].m128i_u64[1];
      if ( v19 )
        j_j___libc_free_0(v19);
      v20 = v3[5].m128i_u64[0];
      if ( v20 )
        j_j___libc_free_0(v20);
      v21 = v3[3].m128i_u64[1];
      if ( v21 )
        j_j___libc_free_0(v21);
      v22 = v3[2].m128i_u64[0];
      if ( v22 )
        j_j___libc_free_0(v22);
      v3 += 11;
      v10 += 11;
    }
    while ( v3 != v32 );
    v3 = *(const __m128i **)a1;
  }
  if ( v3 )
    j_j___libc_free_0((unsigned __int64)v3);
  *(_QWORD *)a1 = v30;
  *(_QWORD *)(a1 + 8) = &v30[11 * v29 + 11 * v31];
  *(_QWORD *)(a1 + 16) = v28;
}
