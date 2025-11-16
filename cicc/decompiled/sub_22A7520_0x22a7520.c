// Function: sub_22A7520
// Address: 0x22a7520
//
void __fastcall sub_22A7520(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  char v5; // r9
  __int64 v6; // rax
  __int64 v7; // rdi
  __int64 v8; // rsi
  __int64 v9; // rdx
  __m128i *v10; // rax
  __int64 v11; // rsi
  __m128i v12; // xmm0
  __m128i v13; // xmm1
  __m128i v14; // xmm2
  __m128i v15; // xmm7
  __m128i v16; // xmm6
  __m128i v17; // xmm7
  __m128i v18; // xmm0
  __int64 v19; // r12
  __m128i v20; // xmm1
  __m128i v21; // xmm2
  unsigned __int8 v22; // dl
  __int64 v23; // rax
  __m128i v24; // xmm7
  __m128i v25; // xmm6
  unsigned __int32 v26; // eax
  unsigned __int32 v27; // ecx
  __int64 v28; // rax
  __m128i v29; // xmm5
  __m128i v30; // xmm3
  __m128i v31; // xmm4
  unsigned __int32 v32; // ecx
  __m128i v33; // [rsp-78h] [rbp-78h] BYREF
  __m128i v34; // [rsp-68h] [rbp-68h] BYREF
  __m128i v35; // [rsp-58h] [rbp-58h] BYREF
  __int64 v36; // [rsp-48h] [rbp-48h]

  if ( a1 == a2 )
    return;
  v2 = a1 + 56;
  if ( a2 == a1 + 56 )
    return;
  do
  {
    v5 = sub_22A71D0(v2, a1);
    v6 = v2;
    v2 += 56;
    if ( v5 )
    {
      v7 = *(_QWORD *)(v2 - 8);
      v8 = v6 - a1;
      v33 = _mm_loadu_si128((const __m128i *)(v2 - 56));
      v9 = 0x6DB6DB6DB6DB6DB7LL * ((v6 - a1) >> 3);
      v10 = (__m128i *)v2;
      v34 = _mm_loadu_si128((const __m128i *)(v2 - 40));
      v35 = _mm_loadu_si128((const __m128i *)(v2 - 24));
      if ( v8 > 0 )
      {
        do
        {
          v11 = v10[-4].m128i_i64[0];
          v12 = _mm_loadu_si128(v10 - 6);
          v10 = (__m128i *)((char *)v10 - 56);
          v13 = _mm_loadu_si128((__m128i *)((char *)v10 - 24));
          v14 = _mm_loadu_si128((__m128i *)((char *)v10 - 56));
          v10[3].m128i_i64[0] = v11;
          v10[2] = v13;
          *v10 = v14;
          v10[1] = v12;
          --v9;
        }
        while ( v9 );
      }
      v15 = _mm_loadu_si128(&v34);
      v16 = _mm_loadu_si128(&v35);
      *(_QWORD *)(a1 + 48) = v7;
      *(__m128i *)(a1 + 16) = v15;
      v17 = _mm_loadu_si128(&v33);
      *(__m128i *)(a1 + 32) = v16;
      *(__m128i *)a1 = v17;
      continue;
    }
    v18 = _mm_loadu_si128((const __m128i *)(v2 - 56));
    v19 = v2 - 112;
    v20 = _mm_loadu_si128((const __m128i *)(v2 - 40));
    v21 = _mm_loadu_si128((const __m128i *)(v2 - 24));
    v36 = *(_QWORD *)(v2 - 8);
    v33 = v18;
    v22 = v18.m128i_u8[10];
    v34 = v20;
    v35 = v21;
    while ( 1 )
    {
      if ( *(_BYTE *)(v19 + 10) <= v22 )
      {
        if ( *(_BYTE *)(v19 + 10) != v22 )
          goto LABEL_12;
        v26 = *(_DWORD *)(v19 + 16);
        if ( v34.m128i_i32[0] >= v26 )
        {
          if ( v34.m128i_i32[0] != v26 )
            break;
          v27 = *(_DWORD *)(v19 + 20);
          if ( v34.m128i_i32[1] >= v27 )
          {
            if ( v34.m128i_i32[1] != v27 )
              break;
            v32 = *(_DWORD *)(v19 + 24);
            if ( v34.m128i_i32[2] >= v32 && (v34.m128i_i32[2] != v32 || v34.m128i_i32[3] >= *(_DWORD *)(v19 + 28)) )
              break;
          }
        }
      }
LABEL_22:
      v28 = *(_QWORD *)(v19 + 48);
      v29 = _mm_loadu_si128((const __m128i *)v19);
      v19 -= 56;
      v30 = _mm_loadu_si128((const __m128i *)(v19 + 72));
      v31 = _mm_loadu_si128((const __m128i *)(v19 + 88));
      *(_QWORD *)(v19 + 160) = v28;
      *(__m128i *)(v19 + 128) = v30;
      *(__m128i *)(v19 + 144) = v31;
      *(__m128i *)(v19 + 112) = v29;
    }
    if ( v34.m128i_i32[0] > v26
      || *(_DWORD *)(v19 + 20) < v34.m128i_i32[1]
      || *(_DWORD *)(v19 + 20) == v34.m128i_i32[1]
      && (*(_DWORD *)(v19 + 24) < v34.m128i_i32[2]
       || *(_DWORD *)(v19 + 24) == v34.m128i_i32[2] && *(_DWORD *)(v19 + 28) < v34.m128i_i32[3]) )
    {
      goto LABEL_12;
    }
    if ( (unsigned __int8)sub_22A6F20((__int64)&v33, (__int64 *)v19) )
    {
      v22 = v33.m128i_u8[10];
      goto LABEL_22;
    }
    sub_22A6F20(v19, v33.m128i_i64);
LABEL_12:
    v23 = v36;
    v24 = _mm_loadu_si128(&v35);
    *(__m128i *)(v19 + 72) = _mm_loadu_si128(&v34);
    v25 = _mm_loadu_si128(&v33);
    *(_QWORD *)(v19 + 104) = v23;
    *(__m128i *)(v19 + 88) = v24;
    *(__m128i *)(v19 + 56) = v25;
  }
  while ( a2 != v2 );
}
