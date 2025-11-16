// Function: sub_36E8110
// Address: 0x36e8110
//
void __fastcall sub_36E8110(__int64 a1, int a2, __int64 a3, __m128i a4)
{
  unsigned int v4; // eax
  __int64 v8; // rsi
  __int64 v9; // rdi
  unsigned __int32 v10; // eax
  const __m128i *v11; // rdx
  unsigned __int32 v12; // ecx
  __int64 v13; // rsi
  int v14; // eax
  __int64 v15; // rax
  _QWORD *v16; // r13
  __m128i v17; // xmm1
  __int64 v18; // rsi
  __int64 v19; // rdi
  unsigned __int8 *v20; // r8
  __int64 v21; // rax
  unsigned int v22; // edx
  __int64 v23; // r9
  unsigned __int64 v24; // rdx
  unsigned __int64 *v25; // rax
  __int64 v26; // rax
  __m128i v27; // xmm0
  __int64 v28; // rsi
  __int64 v29; // rdi
  __int64 v30; // r9
  unsigned __int8 *v31; // r13
  __int64 v32; // rax
  unsigned int v33; // edx
  __int64 v34; // r8
  unsigned __int64 v35; // rdx
  unsigned __int64 *v36; // rax
  __int64 v37; // rax
  __int64 v38; // r13
  unsigned __int64 **v39; // rdi
  __int64 v40; // r15
  __m128i v41; // xmm0
  __m128i v42; // xmm0
  _QWORD *v43; // r9
  unsigned __int64 v44; // rcx
  __int64 v45; // r12
  __int64 v46; // rcx
  __int64 v47; // r8
  __int64 v48; // r9
  __m128i v49; // [rsp+0h] [rbp-290h] BYREF
  __m128i v50; // [rsp+10h] [rbp-280h] BYREF
  unsigned __int8 *v51; // [rsp+20h] [rbp-270h]
  unsigned __int64 *v52; // [rsp+28h] [rbp-268h]
  __int64 v53; // [rsp+30h] [rbp-260h] BYREF
  int v54; // [rsp+38h] [rbp-258h]
  __int64 v55; // [rsp+40h] [rbp-250h] BYREF
  int v56; // [rsp+48h] [rbp-248h]
  unsigned __int64 *v57; // [rsp+50h] [rbp-240h] BYREF
  __int64 v58; // [rsp+58h] [rbp-238h]
  _OWORD v59[35]; // [rsp+60h] [rbp-230h] BYREF

  v4 = *(_DWORD *)(*(_QWORD *)(a1 + 1136) + 344LL);
  if ( v4 <= 0x47 )
    goto LABEL_39;
  if ( (unsigned int)(a2 - 1596) <= 1 || (unsigned int)(a2 - 387) <= 1 )
  {
    if ( v4 != 72 )
    {
      v50.m128i_i32[0] = 2;
      goto LABEL_6;
    }
LABEL_39:
    sub_C64ED0("imma stc not supported on this architecture", 1u);
  }
  v50.m128i_i32[0] = 8;
LABEL_6:
  v8 = *(_QWORD *)(a3 + 80);
  v53 = v8;
  if ( v8 )
    sub_B96E90((__int64)&v53, v8, 1);
  v9 = *(_QWORD *)(a3 + 112);
  v54 = *(_DWORD *)(a3 + 72);
  v10 = sub_36D7800(v9);
  v11 = *(const __m128i **)(a3 + 40);
  v12 = v10;
  v13 = v11[10].m128i_i64[0];
  v14 = *(_DWORD *)(v13 + 24);
  if ( v14 != 11 && v14 != 35 )
    sub_C64ED0("rowcol not constant", 1u);
  v15 = *(_QWORD *)(v13 + 96);
  v16 = *(_QWORD **)(v15 + 24);
  if ( *(_DWORD *)(v15 + 32) > 0x40u )
    v16 = (_QWORD *)*v16;
  v17 = _mm_loadu_si128(v11 + 5);
  v18 = *(_QWORD *)(a3 + 80);
  v52 = (unsigned __int64 *)v59;
  v57 = (unsigned __int64 *)v59;
  v58 = 0x2000000001LL;
  v55 = v18;
  v59[0] = v17;
  if ( v18 )
  {
    v49.m128i_i32[0] = v12;
    sub_B96E90((__int64)&v55, v18, 1);
    v12 = v49.m128i_i32[0];
  }
  v19 = *(_QWORD *)(a1 + 64);
  v56 = *(_DWORD *)(a3 + 72);
  v20 = sub_3400BD0(v19, v12, (__int64)&v55, 7, 0, 1u, a4, 0);
  v21 = (unsigned int)v58;
  v23 = v22;
  v24 = (unsigned int)v58 + 1LL;
  if ( v24 > HIDWORD(v58) )
  {
    v51 = v20;
    v49.m128i_i64[0] = v23;
    sub_C8D5F0((__int64)&v57, v52, v24, 0x10u, (__int64)v20, v23);
    v21 = (unsigned int)v58;
    v20 = v51;
    v23 = v49.m128i_i64[0];
  }
  v25 = &v57[2 * v21];
  *v25 = (unsigned __int64)v20;
  v25[1] = v23;
  v26 = (unsigned int)(v58 + 1);
  LODWORD(v58) = v58 + 1;
  if ( v55 )
  {
    sub_B91220((__int64)&v55, v55);
    v26 = (unsigned int)v58;
  }
  v27 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(a3 + 40) + 120LL));
  if ( v26 + 1 > (unsigned __int64)HIDWORD(v58) )
  {
    v49 = v27;
    sub_C8D5F0((__int64)&v57, v52, v26 + 1, 0x10u, (__int64)v20, v23);
    v26 = (unsigned int)v58;
    v27 = _mm_load_si128(&v49);
  }
  *(__m128i *)&v57[2 * v26] = v27;
  v28 = *(_QWORD *)(a3 + 80);
  LODWORD(v58) = v58 + 1;
  v55 = v28;
  if ( v28 )
    sub_B96E90((__int64)&v55, v28, 1);
  v29 = *(_QWORD *)(a1 + 64);
  v56 = *(_DWORD *)(a3 + 72);
  v31 = sub_3400BD0(v29, (unsigned int)v16, (__int64)&v55, 7, 0, 1u, v27, 0);
  v32 = (unsigned int)v58;
  v34 = v33;
  v35 = (unsigned int)v58 + 1LL;
  if ( v35 > HIDWORD(v58) )
  {
    v49.m128i_i64[0] = v34;
    sub_C8D5F0((__int64)&v57, v52, v35, 0x10u, v34, v30);
    v32 = (unsigned int)v58;
    v34 = v49.m128i_i64[0];
  }
  v36 = &v57[2 * v32];
  *v36 = (unsigned __int64)v31;
  v36[1] = v34;
  v37 = (unsigned int)(v58 + 1);
  LODWORD(v58) = v58 + 1;
  if ( v55 )
  {
    sub_B91220((__int64)&v55, v55);
    v37 = (unsigned int)v58;
  }
  v38 = 200;
  v39 = &v57;
  v40 = 40LL * (unsigned int)(v50.m128i_i32[0] - 1) + 240;
  do
  {
    v41 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(a3 + 40) + v38));
    if ( v37 + 1 > (unsigned __int64)HIDWORD(v58) )
    {
      v50.m128i_i64[0] = (__int64)v39;
      v49 = v41;
      sub_C8D5F0((__int64)v39, v52, v37 + 1, 0x10u, v34, v30);
      v37 = (unsigned int)v58;
      v41 = _mm_load_si128(&v49);
      v39 = (unsigned __int64 **)v50.m128i_i64[0];
    }
    v38 += 40;
    *(__m128i *)&v57[2 * v37] = v41;
    v37 = (unsigned int)(v58 + 1);
    LODWORD(v58) = v58 + 1;
  }
  while ( v40 != v38 );
  v42 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a3 + 40));
  if ( v37 + 1 > (unsigned __int64)HIDWORD(v58) )
  {
    v50 = v42;
    sub_C8D5F0((__int64)&v57, v52, v37 + 1, 0x10u, v34, v30);
    v37 = (unsigned int)v58;
    v42 = _mm_load_si128(&v50);
  }
  *(__m128i *)&v57[2 * v37] = v42;
  v43 = *(_QWORD **)(a1 + 64);
  v44 = *(_QWORD *)(a3 + 48);
  LODWORD(v58) = v58 + 1;
  v45 = sub_33E66D0(v43, a2, (__int64)&v53, v44, *(unsigned int *)(a3 + 68), (__int64)v43, v57, (unsigned int)v58);
  sub_34158F0(*(_QWORD *)(a1 + 64), a3, v45, v46, v47, v48);
  sub_3421DB0(v45);
  sub_33ECEA0(*(const __m128i **)(a1 + 64), a3);
  if ( v57 != v52 )
    _libc_free((unsigned __int64)v57);
  if ( v53 )
    sub_B91220((__int64)&v53, v53);
}
