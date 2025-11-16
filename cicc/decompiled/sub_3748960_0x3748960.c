// Function: sub_3748960
// Address: 0x3748960
//
__int64 __fastcall sub_3748960(__int64 *a1, __int64 a2)
{
  __int64 v2; // rax
  int v3; // edx
  __int32 v4; // eax
  __m128i v5; // xmm1
  __int32 v6; // eax
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // rax
  unsigned __int64 v10; // rdx
  __m128i *v11; // rcx
  __m128i *v12; // rdx
  __m128i *v13; // rax
  __m128i v14; // xmm3
  __int64 v15; // rsi
  __int64 v16; // rax
  __int64 *v17; // r12
  __int64 v18; // r15
  _QWORD *v19; // r13
  __int64 v20; // r15
  __int64 v21; // rdx
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rdx
  const __m128i *v25; // rbx
  const __m128i *v26; // r12
  const __m128i *v27; // rdx
  __m128i *v29; // r13
  __int64 v30; // [rsp+0h] [rbp-1C0h]
  __m128i v31; // [rsp+10h] [rbp-1B0h] BYREF
  __m128i v32; // [rsp+20h] [rbp-1A0h] BYREF
  __int64 v33; // [rsp+30h] [rbp-190h]
  __m128i *v34; // [rsp+40h] [rbp-180h] BYREF
  __int64 v35; // [rsp+48h] [rbp-178h]
  __m128i v36[2]; // [rsp+50h] [rbp-170h] BYREF
  __int64 v37; // [rsp+70h] [rbp-150h]

  v2 = a1[13];
  v3 = *(_DWORD *)(v2 + 544);
  if ( (unsigned int)(v3 - 3) > 1 || *(_DWORD *)(v2 + 560) == 13 || v3 == 39 )
  {
    v34 = v36;
    v35 = 0x800000000LL;
    v4 = sub_3746830(a1, *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
    v31.m128i_i64[0] = 0;
    v31.m128i_i32[2] = v4;
    v32 = 0u;
    v33 = 0;
    v36[0] = _mm_loadu_si128(&v31);
    v5 = _mm_loadu_si128(&v32);
    LODWORD(v35) = 1;
    v36[1] = v5;
    v37 = 0;
    v6 = sub_3746830(a1, *(_QWORD *)(a2 + 32 * (1LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))));
    v31.m128i_i64[0] = 0;
    v31.m128i_i32[2] = v6;
    v9 = (unsigned int)v35;
    v32 = 0u;
    v10 = (unsigned int)v35 + 1LL;
    v33 = 0;
    if ( v10 > HIDWORD(v35) )
    {
      v29 = v34;
      if ( v34 > &v31 || &v31 >= (__m128i *)((char *)v34 + 40 * (unsigned int)v35) )
      {
        sub_C8D5F0((__int64)&v34, v36, v10, 0x28u, v7, v8);
        v11 = v34;
        v9 = (unsigned int)v35;
        v12 = &v31;
      }
      else
      {
        sub_C8D5F0((__int64)&v34, v36, v10, 0x28u, v7, v8);
        v11 = v34;
        v9 = (unsigned int)v35;
        v12 = (__m128i *)((char *)v34 + (char *)&v31 - (char *)v29);
      }
    }
    else
    {
      v11 = v34;
      v12 = &v31;
    }
    v13 = (__m128i *)((char *)v11 + 40 * v9);
    *v13 = _mm_loadu_si128(v12);
    v14 = _mm_loadu_si128(v12 + 1);
    LODWORD(v35) = v35 + 1;
    v13[1] = v14;
    v13[2].m128i_i64[0] = v12[2].m128i_i64[0];
    v15 = a1[10];
    v16 = a1[5];
    v17 = *(__int64 **)(v16 + 752);
    v18 = *(_QWORD *)(a1[15] + 8) - 1600LL;
    v19 = *(_QWORD **)(*(_QWORD *)(v16 + 744) + 32LL);
    v30 = *(_QWORD *)(v16 + 744);
    v31.m128i_i64[0] = v15;
    if ( v15 )
      sub_B96E90((__int64)&v31, v15, 1);
    v20 = (__int64)sub_2E7B380(v19, v18, (unsigned __int8 **)&v31, 0);
    if ( v31.m128i_i64[0] )
      sub_B91220((__int64)&v31, v31.m128i_i64[0]);
    sub_2E31040((__int64 *)(v30 + 40), v20);
    v21 = *v17;
    v22 = *(_QWORD *)v20;
    *(_QWORD *)(v20 + 8) = v17;
    v21 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)v20 = v21 | v22 & 7;
    *(_QWORD *)(v21 + 8) = v20;
    *v17 = v20 | *v17 & 7;
    v23 = a1[11];
    if ( v23 )
      sub_2E882B0(v20, (__int64)v19, v23);
    v24 = a1[12];
    if ( v24 )
      sub_2E88680(v20, (__int64)v19, v24);
    v25 = v34;
    v26 = (__m128i *)((char *)v34 + 40 * (unsigned int)v35);
    if ( v34 != v26 )
    {
      do
      {
        v27 = v25;
        v25 = (const __m128i *)((char *)v25 + 40);
        sub_2E8EAD0(v20, (__int64)v19, v27);
      }
      while ( v26 != v25 );
      v26 = v34;
    }
    if ( v26 != v36 )
      _libc_free((unsigned __int64)v26);
  }
  return 1;
}
