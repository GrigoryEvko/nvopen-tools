// Function: sub_36EF6F0
// Address: 0x36ef6f0
//
void __fastcall sub_36EF6F0(__int64 a1, __int64 a2, unsigned int a3, int a4, __m128i a5)
{
  __int64 v5; // r8
  __int64 v8; // r12
  const __m128i *v9; // rax
  int v10; // edx
  __int64 v11; // r12
  __int64 v12; // rcx
  _QWORD *v13; // rdx
  __int64 v14; // rsi
  __int64 v15; // rcx
  int v16; // edx
  __int64 v17; // rcx
  const __m128i *v18; // rax
  __int64 v19; // r9
  int v20; // esi
  unsigned __int64 v21; // r11
  __m128i *v22; // rdx
  __int64 v23; // rdi
  unsigned __int8 *v24; // r8
  __int64 v25; // rax
  unsigned int v26; // edx
  __int64 v27; // r9
  unsigned __int64 v28; // rdx
  unsigned __int64 *v29; // rax
  const __m128i *v30; // rdx
  __int64 v31; // rax
  __m128i v32; // xmm0
  __int64 v33; // rax
  __int64 v34; // rax
  _DWORD *v35; // rax
  int v36; // eax
  __int64 v37; // r9
  __int64 v38; // r12
  __int64 v39; // rcx
  __int64 v40; // r8
  __int64 v41; // r9
  int v42; // [rsp+Ch] [rbp-144h]
  __int8 *v43; // [rsp+10h] [rbp-140h]
  const __m128i *v44; // [rsp+18h] [rbp-138h]
  unsigned __int8 *v45; // [rsp+18h] [rbp-138h]
  __m128i v46; // [rsp+20h] [rbp-130h] BYREF
  unsigned __int64 **v47; // [rsp+30h] [rbp-120h]
  int v48; // [rsp+38h] [rbp-118h]
  char v49; // [rsp+3Fh] [rbp-111h]
  __int64 v50; // [rsp+40h] [rbp-110h] BYREF
  int v51; // [rsp+48h] [rbp-108h]
  unsigned __int64 *v52; // [rsp+50h] [rbp-100h] BYREF
  __int64 v53; // [rsp+58h] [rbp-F8h]
  _BYTE v54[240]; // [rsp+60h] [rbp-F0h] BYREF

  v5 = a3;
  v8 = *(unsigned int *)(a2 + 64);
  v9 = *(const __m128i **)(a2 + 40);
  v48 = a4;
  v10 = v8;
  v11 = v8 - 6;
  v12 = *(_QWORD *)(v9->m128i_i64[5 * (unsigned int)(v10 - 1)] + 96);
  v13 = *(_QWORD **)(v12 + 24);
  if ( *(_DWORD *)(v12 + 32) > 0x40u )
    v13 = (_QWORD *)*v13;
  v14 = *(_QWORD *)(a2 + 80);
  v49 = v13 == (_QWORD *)1;
  v50 = v14;
  v15 = (v13 == (_QWORD *)1) + v11 + 2;
  if ( v14 )
  {
    v46.m128i_i32[0] = v5;
    v47 = (unsigned __int64 **)((v13 == (_QWORD *)1) + v11 + 2);
    sub_B96E90((__int64)&v50, v14, 1);
    v9 = *(const __m128i **)(a2 + 40);
    v5 = v46.m128i_u32[0];
    v15 = (__int64)v47;
  }
  v16 = *(_DWORD *)(a2 + 72);
  v17 = 40 * v15;
  v18 = v9 + 5;
  v53 = 0xC00000000LL;
  v51 = v16;
  v19 = (__int64)&v18->m128i_i64[(unsigned __int64)v17 / 8];
  v20 = 0;
  v47 = &v52;
  v21 = 0xCCCCCCCCCCCCCCCDLL * (v17 >> 3);
  v52 = (unsigned __int64 *)v54;
  v22 = (__m128i *)v54;
  if ( (unsigned __int64)v17 > 0x1E0 )
  {
    v42 = v5;
    v43 = &v18->m128i_i8[v17];
    v44 = v18;
    v46.m128i_i64[0] = 0xCCCCCCCCCCCCCCCDLL * (v17 >> 3);
    sub_C8D5F0((__int64)&v52, v54, v46.m128i_u64[0], 0x10u, v5, v19);
    v20 = v53;
    LODWORD(v5) = v42;
    v19 = (__int64)v43;
    v18 = v44;
    LODWORD(v21) = v46.m128i_i32[0];
    v22 = (__m128i *)&v52[2 * (unsigned int)v53];
  }
  if ( v18 != (const __m128i *)v19 )
  {
    do
    {
      if ( v22 )
        *v22 = _mm_loadu_si128(v18);
      v18 = (const __m128i *)((char *)v18 + 40);
      ++v22;
    }
    while ( (const __m128i *)v19 != v18 );
    v20 = v53;
  }
  v23 = *(_QWORD *)(a1 + 64);
  LODWORD(v53) = v20 + v21;
  v24 = sub_3400BD0(v23, (unsigned int)v5, (__int64)&v50, 7, 0, 1u, a5, 0);
  v25 = (unsigned int)v53;
  v27 = v26;
  v28 = (unsigned int)v53 + 1LL;
  if ( v28 > HIDWORD(v53) )
  {
    v45 = v24;
    v46.m128i_i64[0] = v27;
    sub_C8D5F0((__int64)v47, v54, v28, 0x10u, (__int64)v24, v27);
    v25 = (unsigned int)v53;
    v24 = v45;
    v27 = v46.m128i_i64[0];
  }
  v29 = &v52[2 * v25];
  *v29 = (unsigned __int64)v24;
  v29[1] = v27;
  v30 = *(const __m128i **)(a2 + 40);
  LODWORD(v53) = v53 + 1;
  v31 = (unsigned int)v53;
  v32 = _mm_loadu_si128(v30);
  if ( (unsigned __int64)(unsigned int)v53 + 1 > HIDWORD(v53) )
  {
    v46 = v32;
    sub_C8D5F0((__int64)v47, v54, (unsigned int)v53 + 1LL, 0x10u, (__int64)v24, v27);
    v31 = (unsigned int)v53;
    v32 = _mm_load_si128(&v46);
  }
  *(__m128i *)&v52[2 * v31] = v32;
  v33 = *(_QWORD *)(a1 + 64);
  LODWORD(v53) = v53 + 1;
  v34 = sub_2E79000(*(__int64 **)(v33 + 40));
  v35 = sub_AE2980(v34, 3u);
  v36 = sub_36D70A0(v11, v35[1] == 32, v49, v48, 1);
  v38 = sub_33E66D0(
          *(_QWORD **)(a1 + 64),
          v36,
          (__int64)&v50,
          *(_QWORD *)(a2 + 48),
          *(unsigned int *)(a2 + 68),
          v37,
          v52,
          (unsigned int)v53);
  sub_34158F0(*(_QWORD *)(a1 + 64), a2, v38, v39, v40, v41);
  sub_3421DB0(v38);
  sub_33ECEA0(*(const __m128i **)(a1 + 64), a2);
  if ( v52 != (unsigned __int64 *)v54 )
    _libc_free((unsigned __int64)v52);
  if ( v50 )
    sub_B91220((__int64)&v50, v50);
}
