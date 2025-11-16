// Function: sub_2046F10
// Address: 0x2046f10
//
__int64 __fastcall sub_2046F10(__int64 a1, __m128i *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, __m128i a7)
{
  __int64 result; // rax
  unsigned __int32 v8; // r15d
  __m128i *v9; // rbx
  unsigned __int32 v10; // r13d
  unsigned int v11; // eax
  unsigned __int32 v12; // r12d
  unsigned int v13; // eax
  __m128i v14; // xmm1
  __int64 v15; // rax
  __m128i *v16; // r15
  unsigned __int32 v17; // r12d
  __m128i *v18; // r13
  bool v19; // cf
  bool v20; // zf
  unsigned int v21; // eax
  __m128i *v22; // rbx
  bool v23; // al
  __int64 v24; // rax
  __m128i v25; // xmm1
  __int64 v26; // rax
  unsigned int v27; // eax
  __int64 v28; // r12
  __int64 i; // r13
  __int8 *v30; // rbx
  __int128 v31; // xmm4
  __int64 v32; // r13
  __int64 v33; // xmm5_8
  __int64 v34; // rdx
  __m128i v35; // xmm7
  int v36; // edx
  __m128i v37; // xmm5
  __int64 v38; // [rsp+10h] [rbp-80h]
  __m128i *v39; // [rsp+18h] [rbp-78h]
  __m128i *v40; // [rsp+20h] [rbp-70h]

  result = (__int64)a2->m128i_i64 - a1;
  v38 = a3;
  v39 = a2;
  if ( (__int64)a2->m128i_i64 - a1 <= 640 )
    return result;
  if ( !a3 )
  {
    v40 = a2;
    goto LABEL_34;
  }
  while ( 2 )
  {
    v8 = *(_DWORD *)(a1 + 72);
    --v38;
    v9 = (__m128i *)(a1 + 40 * ((__int64)(0xCCCCCCCCCCCCCCCDLL * (((__int64)v39->m128i_i64 - a1) >> 3)) / 2));
    v10 = v9[2].m128i_u32[0];
    if ( v10 == v8 )
      v11 = (unsigned int)sub_16AEA10(*(_QWORD *)(a1 + 48) + 24LL, v9->m128i_i64[1] + 24) >> 31;
    else
      LOBYTE(v11) = v10 < v8;
    v12 = v39[-1].m128i_u32[2];
    if ( !(_BYTE)v11 )
    {
      if ( v8 == v12 )
      {
        if ( (int)sub_16AEA10(*(_QWORD *)(a1 + 48) + 24LL, v39[-2].m128i_i64[0] + 24) >= 0 )
          goto LABEL_28;
      }
      else if ( v8 <= v12 )
      {
LABEL_28:
        if ( v10 == v12 )
          v27 = (unsigned int)sub_16AEA10(v9->m128i_i64[1] + 24, v39[-2].m128i_i64[0] + 24) >> 31;
        else
          LOBYTE(v27) = v10 > v12;
        v14 = _mm_loadu_si128((const __m128i *)a1);
        a7 = _mm_loadu_si128((const __m128i *)(a1 + 16));
        v20 = (_BYTE)v27 == 0;
        v15 = *(_QWORD *)(a1 + 32);
        if ( !v20 )
          goto LABEL_11;
        *(__m128i *)a1 = _mm_loadu_si128(v9);
        *(__m128i *)(a1 + 16) = _mm_loadu_si128(v9 + 1);
        goto LABEL_45;
      }
      v15 = *(_QWORD *)(a1 + 32);
      v14 = _mm_loadu_si128((const __m128i *)a1);
      a7 = _mm_loadu_si128((const __m128i *)(a1 + 16));
      v35 = _mm_loadu_si128((const __m128i *)(a1 + 56));
      *(__m128i *)a1 = _mm_loadu_si128((const __m128i *)(a1 + 40));
      *(__m128i *)(a1 + 16) = v35;
LABEL_42:
      v36 = *(_DWORD *)(a1 + 72);
      *(__m128i *)(a1 + 40) = v14;
      *(_DWORD *)(a1 + 72) = v15;
      *(_DWORD *)(a1 + 32) = v36;
      *(__m128i *)(a1 + 56) = a7;
      goto LABEL_12;
    }
    if ( v10 == v12 )
    {
      if ( (int)sub_16AEA10(v9->m128i_i64[1] + 24, v39[-2].m128i_i64[0] + 24) >= 0 )
        goto LABEL_8;
    }
    else if ( v10 <= v12 )
    {
LABEL_8:
      if ( v8 == v12 )
        v13 = (unsigned int)sub_16AEA10(*(_QWORD *)(a1 + 48) + 24LL, v39[-2].m128i_i64[0] + 24) >> 31;
      else
        LOBYTE(v13) = v8 > v12;
      v14 = _mm_loadu_si128((const __m128i *)a1);
      a7 = _mm_loadu_si128((const __m128i *)(a1 + 16));
      v20 = (_BYTE)v13 == 0;
      v15 = *(_QWORD *)(a1 + 32);
      if ( !v20 )
      {
LABEL_11:
        *(__m128i *)a1 = _mm_loadu_si128((__m128i *)((char *)v39 - 40));
        *(__m128i *)(a1 + 16) = _mm_loadu_si128((__m128i *)((char *)v39 - 24));
        *(_DWORD *)(a1 + 32) = v39[-1].m128i_i32[2];
        v39[-1].m128i_i32[2] = v15;
        *(__m128i *)((char *)v39 - 40) = v14;
        *(__m128i *)((char *)v39 - 24) = a7;
        goto LABEL_12;
      }
      v37 = _mm_loadu_si128((const __m128i *)(a1 + 56));
      *(__m128i *)a1 = _mm_loadu_si128((const __m128i *)(a1 + 40));
      *(__m128i *)(a1 + 16) = v37;
      goto LABEL_42;
    }
    v15 = *(_QWORD *)(a1 + 32);
    v14 = _mm_loadu_si128((const __m128i *)a1);
    a7 = _mm_loadu_si128((const __m128i *)(a1 + 16));
    *(__m128i *)a1 = _mm_loadu_si128(v9);
    *(__m128i *)(a1 + 16) = _mm_loadu_si128(v9 + 1);
LABEL_45:
    *(_DWORD *)(a1 + 32) = v9[2].m128i_i32[0];
    v9[2].m128i_i32[0] = v15;
    *v9 = v14;
    v9[1] = a7;
LABEL_12:
    v16 = (__m128i *)(a1 + 40);
    v17 = *(_DWORD *)(a1 + 32);
    v18 = v39;
    v40 = (__m128i *)(a1 + 40);
    v19 = *(_DWORD *)(a1 + 72) < v17;
    v20 = *(_DWORD *)(a1 + 72) == v17;
    if ( *(_DWORD *)(a1 + 72) != v17 )
    {
LABEL_13:
      LOBYTE(v21) = !v19 && !v20;
      goto LABEL_14;
    }
    while ( 1 )
    {
      v21 = (unsigned int)sub_16AEA10(v16->m128i_i64[1] + 24, *(_QWORD *)(a1 + 8) + 24LL) >> 31;
LABEL_14:
      if ( (_BYTE)v21 )
        goto LABEL_21;
      v22 = (__m128i *)((char *)v18 - 40);
      do
      {
        while ( 1 )
        {
          v18 = v22;
          if ( v22[2].m128i_i32[0] == v17 )
            break;
          v23 = v22[2].m128i_i32[0] < v17;
          v22 = (__m128i *)((char *)v22 - 40);
          if ( !v23 )
            goto LABEL_19;
        }
        v24 = v22->m128i_i64[1];
        v22 = (__m128i *)((char *)v22 - 40);
      }
      while ( (int)sub_16AEA10(*(_QWORD *)(a1 + 8) + 24LL, v24 + 24) < 0 );
LABEL_19:
      if ( v16 >= v18 )
        break;
      v25 = _mm_loadu_si128(v16);
      v26 = v16[2].m128i_i64[0];
      a7 = _mm_loadu_si128(v16 + 1);
      *v16 = _mm_loadu_si128(v18);
      v16[1] = _mm_loadu_si128(v18 + 1);
      v16[2].m128i_i32[0] = v18[2].m128i_i32[0];
      v18[2].m128i_i32[0] = v26;
      *v18 = v25;
      v18[1] = a7;
      v17 = *(_DWORD *)(a1 + 32);
LABEL_21:
      v16 = (__m128i *)((char *)v16 + 40);
      v40 = v16;
      v19 = v16[2].m128i_i32[0] < v17;
      v20 = v16[2].m128i_i32[0] == v17;
      if ( v16[2].m128i_i32[0] != v17 )
        goto LABEL_13;
    }
    sub_2046F10(v16, v39, v38);
    result = (__int64)v16->m128i_i64 - a1;
    if ( (__int64)v16->m128i_i64 - a1 > 640 )
    {
      if ( v38 )
      {
        v39 = v16;
        continue;
      }
LABEL_34:
      v28 = 0xCCCCCCCCCCCCCCCDLL * (result >> 3);
      for ( i = (v28 - 2) >> 1; ; --i )
      {
        sub_2045EF0(
          a1,
          i,
          v28,
          a4,
          a5,
          a6,
          a7,
          *(_OWORD *)&_mm_loadu_si128((const __m128i *)(a1 + 40 * i)),
          _mm_loadu_si128((const __m128i *)(a1 + 40 * i + 16)).m128i_i64[0]);
        if ( !i )
          break;
      }
      v30 = &v40[-3].m128i_i8[8];
      do
      {
        v31 = (__int128)_mm_loadu_si128((const __m128i *)v30);
        v32 = (__int64)&v30[-a1];
        v33 = _mm_loadu_si128((const __m128i *)v30 + 1).m128i_u64[0];
        *(__m128i *)v30 = _mm_loadu_si128((const __m128i *)a1);
        v34 = (__int64)&v30[-a1] >> 3;
        v30 -= 40;
        *(__m128i *)(v30 + 56) = _mm_loadu_si128((const __m128i *)(a1 + 16));
        *((_DWORD *)v30 + 18) = *(_DWORD *)(a1 + 32);
        result = sub_2045EF0(a1, 0, 0xCCCCCCCCCCCCCCCDLL * v34, a4, a5, a6, a7, v31, v33);
      }
      while ( v32 > 40 );
    }
    return result;
  }
}
