// Function: sub_3423300
// Address: 0x3423300
//
void __fastcall sub_3423300(__int64 a1, __int64 a2)
{
  __int64 v4; // rbx
  __int64 v5; // rsi
  __int64 v6; // rax
  unsigned __int64 v7; // rsi
  unsigned __int64 v8; // r9
  int v9; // edx
  __int64 v10; // rcx
  bool v11; // zf
  __int64 v12; // r8
  __m128i v13; // xmm0
  __m128i v14; // xmm0
  __int64 v15; // rax
  __m128i v16; // xmm0
  __int64 v17; // rax
  __m128i v18; // xmm0
  __int64 v19; // rbx
  __int64 v20; // rax
  __int64 *v21; // r15
  __m128i v22; // xmm0
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rax
  bool v26; // cc
  __int64 v27; // r8
  __int64 v28; // r14
  __int64 *v29; // rbx
  unsigned __int64 **v30; // rdi
  __int64 v31; // rax
  unsigned __int64 v32; // rdx
  __int64 v33; // rax
  __int64 v34; // rax
  __m128i v35; // xmm0
  __m128i v36; // [rsp+0h] [rbp-2F0h] BYREF
  __m128i v37; // [rsp+10h] [rbp-2E0h] BYREF
  __m128i v38; // [rsp+20h] [rbp-2D0h] BYREF
  __m128i v39; // [rsp+30h] [rbp-2C0h] BYREF
  __m128i v40; // [rsp+40h] [rbp-2B0h] BYREF
  __m128i v41; // [rsp+50h] [rbp-2A0h] BYREF
  unsigned __int64 *v42; // [rsp+68h] [rbp-288h]
  __int64 v43; // [rsp+70h] [rbp-280h] BYREF
  int v44; // [rsp+78h] [rbp-278h]
  __int64 v45; // [rsp+80h] [rbp-270h] BYREF
  int v46; // [rsp+88h] [rbp-268h]
  __m128i v47; // [rsp+90h] [rbp-260h] BYREF
  __int64 v48; // [rsp+A0h] [rbp-250h]
  unsigned __int64 *v49; // [rsp+B0h] [rbp-240h] BYREF
  __int64 v50; // [rsp+B8h] [rbp-238h]
  _BYTE v51[560]; // [rsp+C0h] [rbp-230h] BYREF

  v4 = *(_QWORD *)(a2 + 40);
  v5 = *(_QWORD *)(a2 + 80);
  v42 = (unsigned __int64 *)v51;
  v49 = (unsigned __int64 *)v51;
  v50 = 0x2000000000LL;
  v43 = v5;
  if ( v5 )
  {
    sub_B96E90((__int64)&v43, v5, 1);
    v6 = (unsigned int)v50;
    v7 = HIDWORD(v50);
    v8 = (unsigned int)v50 + 1LL;
  }
  else
  {
    v7 = 32;
    v8 = 1;
    v6 = 0;
  }
  v9 = *(_DWORD *)(a2 + 72);
  v10 = *(_QWORD *)(v4 + 40);
  v48 = 0;
  v47 = 0;
  v44 = v9;
  v11 = *(_WORD *)(*(_QWORD *)(v10 + 48) + 16LL * *(unsigned int *)(v4 + 48)) == 262;
  v40 = _mm_loadu_si128((const __m128i *)v4);
  if ( v11 )
  {
    v41.m128i_i8[0] = 1;
    v12 = v4 + 80;
    v47 = _mm_loadu_si128((const __m128i *)(v4 + 40));
  }
  else
  {
    v41.m128i_i8[0] = 0;
    v12 = v4 + 40;
  }
  v13 = _mm_loadu_si128((const __m128i *)(v12 + 40));
  v39 = _mm_loadu_si128((const __m128i *)v12);
  if ( v8 > v7 )
  {
    v38.m128i_i64[0] = v12;
    v37 = v13;
    sub_C8D5F0((__int64)&v49, v42, v8, 0x10u, v12, v8);
    v6 = (unsigned int)v50;
    v13 = _mm_load_si128(&v37);
    v12 = v38.m128i_i64[0];
  }
  *(__m128i *)&v49[2 * v6] = v13;
  v14 = _mm_loadu_si128((const __m128i *)(v12 + 80));
  LODWORD(v50) = v50 + 1;
  v15 = (unsigned int)v50;
  if ( (unsigned __int64)(unsigned int)v50 + 1 > HIDWORD(v50) )
  {
    v38.m128i_i64[0] = v12;
    v37 = v14;
    sub_C8D5F0((__int64)&v49, v42, (unsigned int)v50 + 1LL, 0x10u, v12, v8);
    v15 = (unsigned int)v50;
    v14 = _mm_load_si128(&v37);
    v12 = v38.m128i_i64[0];
  }
  *(__m128i *)&v49[2 * v15] = v14;
  v16 = _mm_loadu_si128((const __m128i *)(v12 + 120));
  LODWORD(v50) = v50 + 1;
  v17 = (unsigned int)v50;
  if ( (unsigned __int64)(unsigned int)v50 + 1 > HIDWORD(v50) )
  {
    v38.m128i_i64[0] = v12;
    v37 = v16;
    sub_C8D5F0((__int64)&v49, v42, (unsigned int)v50 + 1LL, 0x10u, v12, v8);
    v17 = (unsigned int)v50;
    v16 = _mm_load_si128(&v37);
    v12 = v38.m128i_i64[0];
  }
  *(__m128i *)&v49[2 * v17] = v16;
  v18 = _mm_loadu_si128((const __m128i *)(v12 + 160));
  v19 = *(_QWORD *)(v12 + 160);
  LODWORD(v50) = v50 + 1;
  v20 = (unsigned int)v50;
  if ( (unsigned __int64)(unsigned int)v50 + 1 > HIDWORD(v50) )
  {
    v38.m128i_i64[0] = v12;
    v37 = v18;
    sub_C8D5F0((__int64)&v49, v42, (unsigned int)v50 + 1LL, 0x10u, v12, v8);
    v20 = (unsigned int)v50;
    v18 = _mm_load_si128(&v37);
    v12 = v38.m128i_i64[0];
  }
  v21 = (__int64 *)(v12 + 240);
  *(__m128i *)&v49[2 * v20] = v18;
  v22 = _mm_loadu_si128((const __m128i *)(v12 + 200));
  LODWORD(v50) = v50 + 1;
  v23 = (unsigned int)v50;
  if ( (unsigned __int64)(unsigned int)v50 + 1 > HIDWORD(v50) )
  {
    v38 = v22;
    sub_C8D5F0((__int64)&v49, v42, (unsigned int)v50 + 1LL, 0x10u, v12, v8);
    v23 = (unsigned int)v50;
    v22 = _mm_load_si128(&v38);
  }
  *(__m128i *)&v49[2 * v23] = v22;
  v24 = *(_QWORD *)(v19 + 96);
  v25 = (unsigned int)(v50 + 1);
  v26 = *(_DWORD *)(v24 + 32) <= 0x40u;
  LODWORD(v50) = v50 + 1;
  if ( v26 )
    v27 = *(_QWORD *)(v24 + 24);
  else
    v27 = **(_QWORD **)(v24 + 24);
  if ( v27 )
  {
    v28 = v27;
    v29 = v21;
    v30 = &v49;
    do
    {
      v29 += 5;
      v22 = _mm_loadu_si128((const __m128i *)(v29 - 5));
      if ( v25 + 1 > (unsigned __int64)HIDWORD(v50) )
      {
        v37.m128i_i64[0] = v27;
        v38.m128i_i64[0] = (__int64)v30;
        v36 = v22;
        sub_C8D5F0((__int64)v30, v42, v25 + 1, 0x10u, v27, v8);
        v25 = (unsigned int)v50;
        v22 = _mm_load_si128(&v36);
        v27 = v37.m128i_i64[0];
        v30 = (unsigned __int64 **)v38.m128i_i64[0];
      }
      *(__m128i *)&v49[2 * v25] = v22;
      v25 = (unsigned int)(v50 + 1);
      LODWORD(v50) = v50 + 1;
      --v28;
    }
    while ( v28 );
    v21 += 5 * v27;
  }
  for ( ; v21 != (__int64 *)(*(_QWORD *)(a2 + 40) + 40LL * *(unsigned int *)(a2 + 64)); v21 += 5 )
  {
    v45 = v43;
    if ( v43 )
      sub_B96E90((__int64)&v45, v43, 1);
    v46 = v44;
    sub_3422DF0(a1, (__int64)&v49, *v21, v21[1], (__int64)&v45, v8, v22);
    if ( v45 )
      sub_B91220((__int64)&v45, v45);
  }
  v31 = (unsigned int)v50;
  v32 = (unsigned int)v50 + 1LL;
  if ( v32 > HIDWORD(v50) )
  {
    sub_C8D5F0((__int64)&v49, v42, v32, 0x10u, v27, v8);
    v31 = (unsigned int)v50;
  }
  *(__m128i *)&v49[2 * v31] = _mm_load_si128(&v39);
  LODWORD(v50) = v50 + 1;
  v33 = (unsigned int)v50;
  if ( (unsigned __int64)(unsigned int)v50 + 1 > HIDWORD(v50) )
  {
    sub_C8D5F0((__int64)&v49, v42, (unsigned int)v50 + 1LL, 0x10u, v27, v8);
    v33 = (unsigned int)v50;
  }
  *(__m128i *)&v49[2 * v33] = _mm_load_si128(&v40);
  v34 = (unsigned int)(v50 + 1);
  LODWORD(v50) = v50 + 1;
  if ( v41.m128i_i8[0] )
  {
    v35 = _mm_load_si128(&v47);
    if ( v34 + 1 > (unsigned __int64)HIDWORD(v50) )
    {
      v41 = v35;
      sub_C8D5F0((__int64)&v49, v42, v34 + 1, 0x10u, v27, v8);
      v34 = (unsigned int)v50;
      v35 = _mm_load_si128(&v41);
    }
    *(__m128i *)&v49[2 * v34] = v35;
    LODWORD(v34) = v50 + 1;
    LODWORD(v50) = v50 + 1;
  }
  sub_3415C10(
    *(const __m128i **)(a1 + 64),
    a2,
    28,
    *(_QWORD *)(a2 + 48),
    *(_DWORD *)(a2 + 68),
    v8,
    v49,
    (unsigned int)v34);
  if ( v43 )
    sub_B91220((__int64)&v43, v43);
  if ( v49 != v42 )
    _libc_free((unsigned __int64)v49);
}
