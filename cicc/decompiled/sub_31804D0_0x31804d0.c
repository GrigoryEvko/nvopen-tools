// Function: sub_31804D0
// Address: 0x31804d0
//
__int64 __fastcall sub_31804D0(const __m128i *a1, __int64 a2, _QWORD *a3)
{
  __int64 v5; // r14
  __int32 v6; // eax
  __int64 v7; // rdi
  const __m128i *v8; // r12
  __int64 v9; // rbx
  __int64 v10; // rax
  unsigned __int64 v11; // r15
  unsigned __int64 v12; // r14
  _QWORD *v13; // rdi
  __int64 v14; // rax
  const __m128i *v15; // rdi
  __m128i *v16; // rax
  __m128i *v17; // rcx
  __m128i *v18; // rdx
  __m128i *v19; // rax
  __m128i *v20; // rdx
  __int64 v21; // rax
  __m128i v22; // xmm0
  __int32 v23; // eax
  __int64 v24; // rdi
  __int64 v25; // rax
  const __m128i *v26; // rdi
  __m128i *v27; // rax
  __m128i *v28; // rcx
  __m128i *v29; // rdx
  __m128i *v30; // rax
  __m128i *v31; // rdx
  __int64 v32; // rax
  __m128i v33; // xmm1
  __int64 v35; // rdx
  __int64 i; // rax
  __int64 v37; // rax
  __int64 v38; // [rsp+0h] [rbp-40h]
  __int64 v39; // [rsp+8h] [rbp-38h]

  v5 = sub_31802B0(a3, a1 + 2);
  v38 = v5;
  v6 = a1->m128i_i32[0];
  *(_QWORD *)(v5 + 16) = 0;
  *(_DWORD *)v5 = v6;
  *(_QWORD *)(v5 + 24) = 0;
  *(_QWORD *)(v5 + 8) = a2;
  v7 = a1[1].m128i_i64[1];
  if ( v7 )
    *(_QWORD *)(v5 + 24) = sub_31804D0(v7, v5, a3);
  v8 = (const __m128i *)a1[1].m128i_i64[0];
  if ( v8 )
  {
    v9 = a3[1];
    v39 = v5;
    if ( !v9 )
      goto LABEL_21;
LABEL_5:
    v10 = *(_QWORD *)(v9 + 8);
    a3[1] = v10;
    if ( v10 )
    {
      if ( v9 == *(_QWORD *)(v10 + 24) )
      {
        *(_QWORD *)(v10 + 24) = 0;
        v35 = *(_QWORD *)(a3[1] + 16LL);
        if ( v35 )
        {
          a3[1] = v35;
          for ( i = *(_QWORD *)(v35 + 24); i; i = *(_QWORD *)(i + 24) )
          {
            a3[1] = i;
            v35 = i;
          }
          v37 = *(_QWORD *)(v35 + 16);
          if ( v37 )
            a3[1] = v37;
        }
      }
      else
      {
        *(_QWORD *)(v10 + 16) = 0;
      }
    }
    else
    {
      *a3 = 0;
    }
    v11 = *(_QWORD *)(v9 + 56);
    while ( v11 )
    {
      v12 = v11;
      sub_317D930(*(_QWORD **)(v11 + 24));
      v13 = *(_QWORD **)(v11 + 56);
      v11 = *(_QWORD *)(v11 + 16);
      sub_317D930(v13);
      j_j___libc_free_0(v12);
    }
    v14 = v8[2].m128i_i64[0];
    *(_DWORD *)(v9 + 48) = 0;
    *(_QWORD *)(v9 + 56) = 0;
    *(_QWORD *)(v9 + 32) = v14;
    *(_QWORD *)(v9 + 64) = v9 + 48;
    *(_QWORD *)(v9 + 72) = v9 + 48;
    *(_QWORD *)(v9 + 80) = 0;
    v15 = (const __m128i *)v8[3].m128i_i64[1];
    if ( v15 )
    {
      v16 = sub_317D720(v15, v9 + 48);
      v17 = v16;
      do
      {
        v18 = v16;
        v16 = (__m128i *)v16[1].m128i_i64[0];
      }
      while ( v16 );
      *(_QWORD *)(v9 + 64) = v18;
      v19 = v17;
      do
      {
        v20 = v19;
        v19 = (__m128i *)v19[1].m128i_i64[1];
      }
      while ( v19 );
      *(_QWORD *)(v9 + 72) = v20;
      v21 = v8[5].m128i_i64[0];
      *(_QWORD *)(v9 + 56) = v17;
      *(_QWORD *)(v9 + 80) = v21;
    }
    v22 = _mm_loadu_si128(v8 + 6);
    *(_QWORD *)(v9 + 88) = v8[5].m128i_i64[1];
    for ( *(__m128i *)(v9 + 96) = v22; ; *(__m128i *)(v9 + 96) = v33 )
    {
      *(_QWORD *)(v9 + 112) = v8[7].m128i_i64[0];
      *(_QWORD *)(v9 + 120) = v8[7].m128i_i64[1];
      *(_QWORD *)(v9 + 128) = v8[8].m128i_i64[0];
      v23 = v8->m128i_i32[0];
      *(_QWORD *)(v9 + 16) = 0;
      *(_DWORD *)v9 = v23;
      *(_QWORD *)(v9 + 24) = 0;
      *(_QWORD *)(v39 + 16) = v9;
      *(_QWORD *)(v9 + 8) = v39;
      v24 = v8[1].m128i_i64[1];
      if ( v24 )
        *(_QWORD *)(v9 + 24) = sub_31804D0(v24, v9, a3);
      v8 = (const __m128i *)v8[1].m128i_i64[0];
      if ( !v8 )
        break;
      v39 = v9;
      v9 = a3[1];
      if ( v9 )
        goto LABEL_5;
LABEL_21:
      v9 = sub_22077B0(0x88u);
      v25 = v8[2].m128i_i64[0];
      *(_DWORD *)(v9 + 48) = 0;
      *(_QWORD *)(v9 + 32) = v25;
      *(_QWORD *)(v9 + 56) = 0;
      *(_QWORD *)(v9 + 64) = v9 + 48;
      *(_QWORD *)(v9 + 72) = v9 + 48;
      *(_QWORD *)(v9 + 80) = 0;
      v26 = (const __m128i *)v8[3].m128i_i64[1];
      if ( v26 )
      {
        v27 = sub_317D720(v26, v9 + 48);
        v28 = v27;
        do
        {
          v29 = v27;
          v27 = (__m128i *)v27[1].m128i_i64[0];
        }
        while ( v27 );
        *(_QWORD *)(v9 + 64) = v29;
        v30 = v28;
        do
        {
          v31 = v30;
          v30 = (__m128i *)v30[1].m128i_i64[1];
        }
        while ( v30 );
        *(_QWORD *)(v9 + 72) = v31;
        v32 = v8[5].m128i_i64[0];
        *(_QWORD *)(v9 + 56) = v28;
        *(_QWORD *)(v9 + 80) = v32;
      }
      v33 = _mm_loadu_si128(v8 + 6);
      *(_QWORD *)(v9 + 88) = v8[5].m128i_i64[1];
    }
  }
  return v38;
}
