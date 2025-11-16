// Function: sub_3748D50
// Address: 0x3748d50
//
__int64 __fastcall sub_3748D50(__int64 *a1, __int64 a2)
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
  __int32 v15; // eax
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 v18; // rax
  unsigned __int64 v19; // rdx
  __m128i *v20; // rcx
  __m128i *v21; // rdx
  __m128i *v22; // rax
  __m128i v23; // xmm5
  __int64 v24; // rsi
  __int64 v25; // rax
  __int64 *v26; // rbx
  __int64 v27; // r15
  _QWORD *v28; // r13
  __int64 v29; // r15
  __int64 v30; // rdx
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 v33; // rdx
  const __m128i *v34; // rbx
  const __m128i *v35; // r12
  const __m128i *v36; // rdx
  __m128i *v38; // r15
  __m128i *v39; // rbx
  __int64 v40; // [rsp+0h] [rbp-1C0h]
  __m128i v41; // [rsp+10h] [rbp-1B0h] BYREF
  __m128i v42; // [rsp+20h] [rbp-1A0h] BYREF
  __int64 v43; // [rsp+30h] [rbp-190h]
  __m128i *v44; // [rsp+40h] [rbp-180h] BYREF
  __int64 v45; // [rsp+48h] [rbp-178h]
  __m128i v46[2]; // [rsp+50h] [rbp-170h] BYREF
  __int64 v47; // [rsp+70h] [rbp-150h]

  v2 = a1[13];
  v3 = *(_DWORD *)(v2 + 544);
  if ( (unsigned int)(v3 - 3) > 1 || *(_DWORD *)(v2 + 560) == 13 || v3 == 39 )
  {
    v44 = v46;
    v45 = 0x800000000LL;
    v4 = sub_3746830(a1, *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
    v41.m128i_i64[0] = 0;
    v41.m128i_i32[2] = v4;
    v42 = 0u;
    v43 = 0;
    v46[0] = _mm_loadu_si128(&v41);
    v5 = _mm_loadu_si128(&v42);
    LODWORD(v45) = 1;
    v46[1] = v5;
    v47 = 0;
    v6 = sub_3746830(a1, *(_QWORD *)(a2 + 32 * (1LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))));
    v41.m128i_i64[0] = 0;
    v41.m128i_i32[2] = v6;
    v9 = (unsigned int)v45;
    v42 = 0u;
    v10 = (unsigned int)v45 + 1LL;
    v43 = 0;
    if ( v10 > HIDWORD(v45) )
    {
      v38 = v44;
      if ( v44 > &v41 || &v41 >= (__m128i *)((char *)v44 + 40 * (unsigned int)v45) )
      {
        sub_C8D5F0((__int64)&v44, v46, v10, 0x28u, v7, v8);
        v11 = v44;
        v9 = (unsigned int)v45;
        v12 = &v41;
      }
      else
      {
        sub_C8D5F0((__int64)&v44, v46, v10, 0x28u, v7, v8);
        v11 = v44;
        v9 = (unsigned int)v45;
        v12 = (__m128i *)((char *)v44 + (char *)&v41 - (char *)v38);
      }
    }
    else
    {
      v11 = v44;
      v12 = &v41;
    }
    v13 = (__m128i *)((char *)v11 + 40 * v9);
    *v13 = _mm_loadu_si128(v12);
    v14 = _mm_loadu_si128(v12 + 1);
    LODWORD(v45) = v45 + 1;
    v13[1] = v14;
    v13[2].m128i_i64[0] = v12[2].m128i_i64[0];
    v15 = sub_3746830(a1, *(_QWORD *)(a2 + 32 * (2LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))));
    v41.m128i_i64[0] = 0;
    v41.m128i_i32[2] = v15;
    v18 = (unsigned int)v45;
    v42 = 0u;
    v19 = (unsigned int)v45 + 1LL;
    v43 = 0;
    if ( v19 > HIDWORD(v45) )
    {
      v39 = v44;
      if ( v44 > &v41 || &v41 >= (__m128i *)((char *)v44 + 40 * (unsigned int)v45) )
      {
        sub_C8D5F0((__int64)&v44, v46, v19, 0x28u, v16, v17);
        v20 = v44;
        v18 = (unsigned int)v45;
        v21 = &v41;
      }
      else
      {
        sub_C8D5F0((__int64)&v44, v46, v19, 0x28u, v16, v17);
        v20 = v44;
        v18 = (unsigned int)v45;
        v21 = (__m128i *)((char *)v44 + (char *)&v41 - (char *)v39);
      }
    }
    else
    {
      v20 = v44;
      v21 = &v41;
    }
    v22 = (__m128i *)((char *)v20 + 40 * v18);
    *v22 = _mm_loadu_si128(v21);
    v23 = _mm_loadu_si128(v21 + 1);
    LODWORD(v45) = v45 + 1;
    v22[1] = v23;
    v22[2].m128i_i64[0] = v21[2].m128i_i64[0];
    v24 = a1[10];
    v25 = a1[5];
    v26 = *(__int64 **)(v25 + 752);
    v27 = *(_QWORD *)(a1[15] + 8) - 1640LL;
    v28 = *(_QWORD **)(*(_QWORD *)(v25 + 744) + 32LL);
    v40 = *(_QWORD *)(v25 + 744);
    v41.m128i_i64[0] = v24;
    if ( v24 )
      sub_B96E90((__int64)&v41, v24, 1);
    v29 = (__int64)sub_2E7B380(v28, v27, (unsigned __int8 **)&v41, 0);
    if ( v41.m128i_i64[0] )
      sub_B91220((__int64)&v41, v41.m128i_i64[0]);
    sub_2E31040((__int64 *)(v40 + 40), v29);
    v30 = *v26;
    v31 = *(_QWORD *)v29;
    *(_QWORD *)(v29 + 8) = v26;
    v30 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)v29 = v30 | v31 & 7;
    *(_QWORD *)(v30 + 8) = v29;
    *v26 = v29 | *v26 & 7;
    v32 = a1[11];
    if ( v32 )
      sub_2E882B0(v29, (__int64)v28, v32);
    v33 = a1[12];
    if ( v33 )
      sub_2E88680(v29, (__int64)v28, v33);
    v34 = v44;
    v35 = (__m128i *)((char *)v44 + 40 * (unsigned int)v45);
    if ( v44 != v35 )
    {
      do
      {
        v36 = v34;
        v34 = (const __m128i *)((char *)v34 + 40);
        sub_2E8EAD0(v29, (__int64)v28, v36);
      }
      while ( v35 != v34 );
      v35 = v44;
    }
    if ( v35 != v46 )
      _libc_free((unsigned __int64)v35);
  }
  return 1;
}
