// Function: sub_FE83C0
// Address: 0xfe83c0
//
signed __int64 __fastcall sub_FE83C0(__m128i *a1, unsigned __int64 *a2, __int64 a3)
{
  signed __int64 result; // rax
  __m128i *v4; // r8
  __int64 v5; // r15
  __m128i *v7; // r12
  unsigned __int32 v8; // esi
  unsigned __int32 v9; // edi
  __int32 v10; // r10d
  unsigned __int32 v11; // ecx
  __m128i *v12; // rax
  __int64 v13; // r9
  unsigned __int32 v14; // edx
  unsigned __int32 v15; // esi
  unsigned __int32 v16; // edx
  unsigned __int64 v17; // rbx
  __m128i *v18; // rax
  unsigned __int64 *v19; // r13
  __int32 v20; // esi
  __int64 v21; // rdx
  __m128i v22; // xmm3
  __int64 v23; // rbx
  __int64 v24; // rsi
  unsigned __int64 v25; // rcx
  __int64 v26; // r8
  __m128i v27; // xmm5

  result = (char *)a2 - (char *)a1;
  if ( (char *)a2 - (char *)a1 <= 256 )
    return result;
  v4 = (__m128i *)a2;
  v5 = a3;
  v7 = a1 + 1;
  if ( !a3 )
  {
    v19 = a2;
    goto LABEL_23;
  }
  while ( 2 )
  {
    v8 = a1[1].m128i_u32[1];
    v9 = v4[-1].m128i_u32[1];
    --v5;
    v10 = a1->m128i_i32[0];
    v11 = a1->m128i_u32[1];
    v12 = &a1[result >> 5];
    v13 = a1->m128i_i64[1];
    v14 = v12->m128i_u32[1];
    if ( v8 < v14 )
    {
      if ( v14 < v9 )
        goto LABEL_5;
      if ( v8 >= v9 )
      {
        v27 = _mm_loadu_si128(a1 + 1);
        a1[1].m128i_i32[0] = v10;
        a1[1].m128i_i32[1] = v11;
        a1[1].m128i_i64[1] = v13;
        *a1 = v27;
        v15 = v4[-1].m128i_u32[1];
        goto LABEL_6;
      }
LABEL_21:
      *a1 = _mm_loadu_si128(v4 - 1);
      v4[-1].m128i_i32[0] = v10;
      v15 = v11;
      v4[-1].m128i_i32[1] = v11;
      v4[-1].m128i_i64[1] = v13;
      v11 = a1[1].m128i_u32[1];
      goto LABEL_6;
    }
    if ( v8 < v9 )
    {
      v22 = _mm_loadu_si128(a1 + 1);
      a1[1].m128i_i32[0] = v10;
      a1[1].m128i_i32[1] = v11;
      a1[1].m128i_i64[1] = v13;
      *a1 = v22;
      v15 = v4[-1].m128i_u32[1];
      goto LABEL_6;
    }
    if ( v14 < v9 )
      goto LABEL_21;
LABEL_5:
    *a1 = _mm_loadu_si128(v12);
    v12->m128i_i32[0] = v10;
    v12->m128i_i32[1] = v11;
    v12->m128i_i64[1] = v13;
    v15 = v4[-1].m128i_u32[1];
    v11 = a1[1].m128i_u32[1];
LABEL_6:
    v16 = a1->m128i_u32[1];
    v17 = (unsigned __int64)v7;
    v18 = v4;
    while ( 1 )
    {
      v19 = (unsigned __int64 *)v17;
      if ( v11 < v16 )
        goto LABEL_12;
      --v18;
      if ( v15 > v16 )
      {
        do
          --v18;
        while ( v18->m128i_i32[1] > v16 );
      }
      if ( (unsigned __int64)v18 <= v17 )
        break;
      v20 = *(_DWORD *)v17;
      v21 = *(_QWORD *)(v17 + 8);
      *(__m128i *)v17 = _mm_loadu_si128(v18);
      v18->m128i_i32[0] = v20;
      v15 = v18[-1].m128i_u32[1];
      v18->m128i_i32[1] = v11;
      v18->m128i_i64[1] = v21;
      v16 = a1->m128i_u32[1];
LABEL_12:
      v11 = *(_DWORD *)(v17 + 20);
      v17 += 16LL;
    }
    sub_FE83C0(v17, v4, v5);
    result = v17 - (_QWORD)a1;
    if ( (__int64)(v17 - (_QWORD)a1) > 256 )
    {
      if ( v5 )
      {
        v4 = (__m128i *)v17;
        continue;
      }
LABEL_23:
      v23 = result >> 4;
      v24 = ((result >> 4) - 2) >> 1;
      sub_FE8140((__int64)a1, v24, result >> 4, a1[v24].m128i_u64[0], a1[v24].m128i_i64[1]);
      do
      {
        --v24;
        sub_FE8140((__int64)a1, v24, v23, a1[v24].m128i_u64[0], a1[v24].m128i_i64[1]);
      }
      while ( v24 );
      do
      {
        v19 -= 2;
        v25 = *v19;
        v26 = v19[1];
        *(__m128i *)v19 = _mm_loadu_si128(a1);
        result = (signed __int64)sub_FE8140((__int64)a1, 0, ((char *)v19 - (char *)a1) >> 4, v25, v26);
      }
      while ( (char *)v19 - (char *)a1 > 16 );
    }
    return result;
  }
}
