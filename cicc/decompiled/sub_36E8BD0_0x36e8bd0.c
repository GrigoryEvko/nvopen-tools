// Function: sub_36E8BD0
// Address: 0x36e8bd0
//
void __fastcall sub_36E8BD0(__int64 a1, unsigned int a2, int a3, __int64 a4, __m128i a5)
{
  __int64 v5; // rax
  __int64 v7; // rsi
  int v10; // eax
  __int64 v11; // rcx
  int v12; // edx
  __int64 v13; // rdx
  __int64 v14; // rsi
  __int64 v15; // rdi
  unsigned __int8 *v16; // r8
  __int64 v17; // rax
  unsigned int v18; // edx
  __int64 v19; // r9
  unsigned __int64 v20; // rdx
  unsigned __int64 *v21; // rax
  __int64 v22; // rsi
  __int64 v23; // rdi
  unsigned __int8 *v24; // r8
  __int64 v25; // rax
  unsigned int v26; // edx
  __int64 v27; // r9
  unsigned __int64 v28; // rdx
  unsigned __int64 *v29; // rax
  __int64 v30; // rsi
  __int64 v31; // rdi
  __int64 v32; // r9
  unsigned __int8 *v33; // r12
  __int64 v34; // rax
  unsigned int v35; // edx
  __int64 v36; // r8
  unsigned __int64 v37; // rdx
  unsigned __int64 *v38; // rax
  __int64 v39; // rax
  __m128i v40; // xmm0
  __int64 v41; // rax
  __m128i v42; // xmm0
  __int64 v43; // rax
  __m128i v44; // xmm0
  __int64 v45; // rax
  __m128i v46; // xmm0
  const __m128i *v47; // rdx
  __int64 v48; // rax
  __m128i v49; // xmm0
  _QWORD *v50; // r9
  unsigned __int64 v51; // rcx
  __int64 v52; // r12
  __int64 v53; // rcx
  __int64 v54; // r8
  __int64 v55; // r9
  unsigned __int8 *v56; // [rsp+8h] [rbp-288h]
  __m128i v57; // [rsp+10h] [rbp-280h] BYREF
  unsigned __int8 *v58; // [rsp+20h] [rbp-270h]
  int v59; // [rsp+2Ch] [rbp-264h]
  __int64 v60; // [rsp+30h] [rbp-260h] BYREF
  int v61; // [rsp+38h] [rbp-258h]
  __int64 v62; // [rsp+40h] [rbp-250h] BYREF
  int v63; // [rsp+48h] [rbp-248h]
  unsigned __int64 *v64; // [rsp+50h] [rbp-240h] BYREF
  __int64 v65; // [rsp+58h] [rbp-238h]
  _BYTE v66[560]; // [rsp+60h] [rbp-230h] BYREF

  v5 = *(_QWORD *)(a1 + 1136);
  v59 = a3;
  if ( *(_DWORD *)(v5 + 344) <= 0x48u )
    sub_C64ED0("bmmamma is not supported on this architecture", 1u);
  v7 = *(_QWORD *)(a4 + 80);
  v60 = v7;
  if ( v7 )
    sub_B96E90((__int64)&v60, v7, 1);
  v10 = *(_DWORD *)(a4 + 72);
  v11 = *(_QWORD *)(*(_QWORD *)(a4 + 40) + 80LL);
  v61 = v10;
  v12 = *(_DWORD *)(v11 + 24);
  if ( v12 != 35 && v12 != 11 )
    sub_C64ED0("rowcol not constant", 1u);
  v13 = *(_QWORD *)(v11 + 96);
  if ( *(_DWORD *)(v13 + 32) <= 0x40u )
    v57.m128i_i64[0] = *(_QWORD *)(v13 + 24);
  else
    v57.m128i_i64[0] = **(_QWORD **)(v13 + 24);
  v14 = *(_QWORD *)(a4 + 80);
  v64 = (unsigned __int64 *)v66;
  v65 = 0x2000000000LL;
  v62 = v14;
  if ( v14 )
  {
    sub_B96E90((__int64)&v62, v14, 1);
    v10 = *(_DWORD *)(a4 + 72);
  }
  v15 = *(_QWORD *)(a1 + 64);
  v63 = v10;
  v16 = sub_3400BD0(v15, 2, (__int64)&v62, 7, 0, 1u, a5, 0);
  v17 = (unsigned int)v65;
  v19 = v18;
  v20 = (unsigned int)v65 + 1LL;
  if ( v20 > HIDWORD(v65) )
  {
    v56 = v16;
    v58 = (unsigned __int8 *)v19;
    sub_C8D5F0((__int64)&v64, v66, v20, 0x10u, (__int64)v16, v19);
    v17 = (unsigned int)v65;
    v16 = v56;
    v19 = (__int64)v58;
  }
  v21 = &v64[2 * v17];
  *v21 = (unsigned __int64)v16;
  v21[1] = v19;
  LODWORD(v65) = v65 + 1;
  if ( v62 )
    sub_B91220((__int64)&v62, v62);
  v22 = *(_QWORD *)(a4 + 80);
  v62 = v22;
  if ( v22 )
    sub_B96E90((__int64)&v62, v22, 1);
  v23 = *(_QWORD *)(a1 + 64);
  v63 = *(_DWORD *)(a4 + 72);
  v24 = sub_3400BD0(v23, v57.m128i_u32[0], (__int64)&v62, 7, 0, 1u, a5, 0);
  v25 = (unsigned int)v65;
  v27 = v26;
  v28 = (unsigned int)v65 + 1LL;
  if ( v28 > HIDWORD(v65) )
  {
    v58 = v24;
    v57.m128i_i64[0] = v27;
    sub_C8D5F0((__int64)&v64, v66, v28, 0x10u, (__int64)v24, v27);
    v25 = (unsigned int)v65;
    v24 = v58;
    v27 = v57.m128i_i64[0];
  }
  v29 = &v64[2 * v25];
  *v29 = (unsigned __int64)v24;
  v29[1] = v27;
  LODWORD(v65) = v65 + 1;
  if ( v62 )
    sub_B91220((__int64)&v62, v62);
  v30 = *(_QWORD *)(a4 + 80);
  v62 = v30;
  if ( v30 )
    sub_B96E90((__int64)&v62, v30, 1);
  v31 = *(_QWORD *)(a1 + 64);
  v63 = *(_DWORD *)(a4 + 72);
  v33 = sub_3400BD0(v31, a2, (__int64)&v62, 7, 0, 1u, a5, 0);
  v34 = (unsigned int)v65;
  v36 = v35;
  v37 = (unsigned int)v65 + 1LL;
  if ( v37 > HIDWORD(v65) )
  {
    v57.m128i_i64[0] = v36;
    sub_C8D5F0((__int64)&v64, v66, v37, 0x10u, v36, v32);
    v34 = (unsigned int)v65;
    v36 = v57.m128i_i64[0];
  }
  v38 = &v64[2 * v34];
  *v38 = (unsigned __int64)v33;
  v38[1] = v36;
  v39 = (unsigned int)(v65 + 1);
  LODWORD(v65) = v65 + 1;
  if ( v62 )
  {
    sub_B91220((__int64)&v62, v62);
    v39 = (unsigned int)v65;
  }
  v40 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(a4 + 40) + 120LL));
  if ( v39 + 1 > (unsigned __int64)HIDWORD(v65) )
  {
    v57 = v40;
    sub_C8D5F0((__int64)&v64, v66, v39 + 1, 0x10u, v36, v32);
    v39 = (unsigned int)v65;
    v40 = _mm_load_si128(&v57);
  }
  *(__m128i *)&v64[2 * v39] = v40;
  v42 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(a4 + 40) + 160LL));
  LODWORD(v65) = v65 + 1;
  v41 = (unsigned int)v65;
  if ( (unsigned __int64)(unsigned int)v65 + 1 > HIDWORD(v65) )
  {
    v57 = v42;
    sub_C8D5F0((__int64)&v64, v66, (unsigned int)v65 + 1LL, 0x10u, v36, v32);
    v41 = (unsigned int)v65;
    v42 = _mm_load_si128(&v57);
  }
  *(__m128i *)&v64[2 * v41] = v42;
  v44 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(a4 + 40) + 200LL));
  LODWORD(v65) = v65 + 1;
  v43 = (unsigned int)v65;
  if ( (unsigned __int64)(unsigned int)v65 + 1 > HIDWORD(v65) )
  {
    v57 = v44;
    sub_C8D5F0((__int64)&v64, v66, (unsigned int)v65 + 1LL, 0x10u, v36, v32);
    v43 = (unsigned int)v65;
    v44 = _mm_load_si128(&v57);
  }
  *(__m128i *)&v64[2 * v43] = v44;
  v46 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(a4 + 40) + 240LL));
  LODWORD(v65) = v65 + 1;
  v45 = (unsigned int)v65;
  if ( (unsigned __int64)(unsigned int)v65 + 1 > HIDWORD(v65) )
  {
    v57 = v46;
    sub_C8D5F0((__int64)&v64, v66, (unsigned int)v65 + 1LL, 0x10u, v36, v32);
    v45 = (unsigned int)v65;
    v46 = _mm_load_si128(&v57);
  }
  *(__m128i *)&v64[2 * v45] = v46;
  v47 = *(const __m128i **)(a4 + 40);
  LODWORD(v65) = v65 + 1;
  v48 = (unsigned int)v65;
  v49 = _mm_loadu_si128(v47);
  if ( (unsigned __int64)(unsigned int)v65 + 1 > HIDWORD(v65) )
  {
    v57 = v49;
    sub_C8D5F0((__int64)&v64, v66, (unsigned int)v65 + 1LL, 0x10u, v36, v32);
    v48 = (unsigned int)v65;
    v49 = _mm_load_si128(&v57);
  }
  *(__m128i *)&v64[2 * v48] = v49;
  v50 = *(_QWORD **)(a1 + 64);
  v51 = *(_QWORD *)(a4 + 48);
  LODWORD(v65) = v65 + 1;
  v52 = sub_33E66D0(v50, v59, (__int64)&v60, v51, *(unsigned int *)(a4 + 68), (__int64)v50, v64, (unsigned int)v65);
  sub_34158F0(*(_QWORD *)(a1 + 64), a4, v52, v53, v54, v55);
  sub_3421DB0(v52);
  sub_33ECEA0(*(const __m128i **)(a1 + 64), a4);
  if ( v64 != (unsigned __int64 *)v66 )
    _libc_free((unsigned __int64)v64);
  if ( v60 )
    sub_B91220((__int64)&v60, v60);
}
