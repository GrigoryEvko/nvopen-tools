// Function: sub_38D78D0
// Address: 0x38d78d0
//
__int64 *__fastcall sub_38D78D0(__int64 a1, unsigned __int32 a2)
{
  __int64 v3; // rax
  unsigned __int64 v5; // rbx
  __int64 v6; // rax
  unsigned __int32 v7; // r12d
  unsigned __int64 v8; // rsi
  __int64 i; // rax
  __int64 v10; // rcx
  unsigned __int32 *v11; // rdx
  int v12; // eax
  __int64 *v13; // r14
  bool v14; // al
  __int64 v15; // rax
  __int64 v16; // r15
  __int64 v17; // r8
  unsigned __int64 v18; // r10
  __int64 v19; // rcx
  __int64 v20; // rdi
  unsigned __int64 v21; // r9
  __int64 v22; // rsi
  __int64 v23; // rdx
  __m128i *v24; // rax
  const __m128i *v25; // rdx
  __int64 v26; // rcx
  __int32 v27; // esi
  unsigned int v28; // esi
  __int64 v29; // rcx
  __int64 v30; // rdx
  __int64 v31; // rax
  __int64 v32; // rbx
  __int64 v33; // [rsp+8h] [rbp-48h]
  __int64 v34; // [rsp+8h] [rbp-48h]
  __int64 v35; // [rsp+8h] [rbp-48h]
  __m128i v36; // [rsp+10h] [rbp-40h] BYREF
  unsigned __int32 v37; // [rsp+20h] [rbp-30h]
  __int64 v38; // [rsp+28h] [rbp-28h]

  v3 = *(unsigned int *)(a1 + 120);
  if ( !((unsigned int)v3 | a2) )
    return (__int64 *)(a1 + 96);
  v5 = *(_QWORD *)(a1 + 112);
  v6 = 16 * v3;
  v7 = a2;
  v8 = v5 + v6;
  for ( i = v6 >> 4; i > 0; i = i - v10 - 1 )
  {
    while ( 1 )
    {
      v10 = i >> 1;
      v11 = (unsigned __int32 *)(v5 + 16 * (i >> 1));
      if ( v7 > *v11 )
        break;
      i >>= 1;
      if ( v10 <= 0 )
        goto LABEL_7;
    }
    v5 = (unsigned __int64)(v11 + 4);
  }
LABEL_7:
  if ( v8 == v5 )
  {
    v14 = 1;
  }
  else
  {
    v12 = *(_DWORD *)v5;
    if ( *(_DWORD *)v5 != v7 || (v5 += 16LL, v5 != v8) )
    {
      v13 = *(__int64 **)(v5 + 8);
      v14 = v12 != v7;
      goto LABEL_10;
    }
    v14 = 0;
  }
  v13 = (__int64 *)(a1 + 96);
LABEL_10:
  if ( v7 && v14 )
  {
    v15 = sub_22077B0(0xE0u);
    v16 = v15;
    if ( v15 )
    {
      v33 = v15;
      sub_38CF760(v15, 1, 0, 0);
      v17 = v33;
      *(_QWORD *)(v16 + 56) = 0;
      *(_WORD *)(v16 + 48) = 0;
      *(_QWORD *)(v16 + 64) = v16 + 80;
      *(_QWORD *)(v16 + 72) = 0x2000000000LL;
      *(_QWORD *)(v16 + 112) = v16 + 128;
      *(_QWORD *)(v16 + 120) = 0x400000000LL;
    }
    else
    {
      v17 = 0;
    }
    v18 = *(unsigned int *)(a1 + 120);
    v19 = *(_QWORD *)(a1 + 112);
    v36.m128i_i32[0] = v7;
    v20 = a1 + 112;
    v36.m128i_i64[1] = v16;
    v21 = *(unsigned int *)(a1 + 124);
    LODWORD(v22) = v18;
    v23 = 16 * v18;
    v24 = (__m128i *)(v19 + 16 * v18);
    if ( v24 == (__m128i *)v5 )
    {
      if ( (unsigned int)v18 >= (unsigned int)v21 )
      {
        v35 = v17;
        sub_16CD150(v20, (const void *)(a1 + 128), 0, 16, v17, v21);
        v17 = v35;
        v5 = *(_QWORD *)(a1 + 112) + 16LL * *(unsigned int *)(a1 + 120);
      }
      *(__m128i *)v5 = _mm_load_si128(&v36);
      ++*(_DWORD *)(a1 + 120);
    }
    else
    {
      if ( v18 >= v21 )
      {
        v32 = v5 - v19;
        v34 = v17;
        sub_16CD150(v20, (const void *)(a1 + 128), 0, 16, v17, v21);
        v19 = *(_QWORD *)(a1 + 112);
        v17 = v34;
        v22 = *(unsigned int *)(a1 + 120);
        v23 = 16 * v22;
        v5 = v19 + v32;
        v24 = (__m128i *)(v19 + 16 * v22);
      }
      v25 = (const __m128i *)(v19 + v23 - 16);
      if ( v24 )
      {
        *v24 = _mm_loadu_si128(v25);
        v22 = *(unsigned int *)(a1 + 120);
        v24 = (__m128i *)(*(_QWORD *)(a1 + 112) + 16 * v22);
        v25 = v24 - 1;
      }
      v26 = (__int64)((__int64)v25->m128i_i64 - v5) >> 4;
      if ( (__int64)((__int64)v25->m128i_i64 - v5) > 0 )
      {
        do
        {
          v27 = v25[-1].m128i_i32[0];
          --v25;
          --v24;
          v24->m128i_i32[0] = v27;
          v24->m128i_i64[1] = v25->m128i_i64[1];
          --v26;
        }
        while ( v26 );
        LODWORD(v22) = *(_DWORD *)(a1 + 120);
      }
      v28 = v22 + 1;
      v29 = v16;
      *(_DWORD *)(a1 + 120) = v28;
      if ( v5 <= (unsigned __int64)&v36 && (unsigned __int64)&v36 < *(_QWORD *)(a1 + 112) + 16 * (unsigned __int64)v28 )
      {
        v7 = v37;
        v29 = v38;
      }
      *(_DWORD *)v5 = v7;
      *(_QWORD *)(v5 + 8) = v29;
    }
    v30 = *v13;
    v31 = *(_QWORD *)v16;
    *(_QWORD *)(v16 + 8) = v13;
    v30 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)v16 = v30 | v31 & 7;
    *(_QWORD *)(v30 + 8) = v17;
    *v13 = *v13 & 7 | v17;
    *(_QWORD *)(v16 + 24) = a1;
  }
  return v13;
}
