// Function: sub_2023B70
// Address: 0x2023b70
//
void __fastcall sub_2023B70(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __m128 a5, __m128 a6)
{
  __int64 v8; // rsi
  __int64 v9; // rsi
  char *v10; // rax
  char v11; // dl
  __int64 v12; // rax
  __m128i v13; // xmm2
  __int64 v14; // rdx
  const __m128i *v15; // rax
  unsigned __int64 v16; // r10
  const __m128i *v17; // r9
  unsigned __int64 v18; // rdx
  __m128 *v19; // rcx
  int v20; // edi
  _BYTE *v21; // rsi
  unsigned int v22; // ecx
  __int64 *v23; // rdi
  int v24; // edx
  __int64 v25; // rdx
  __int64 v26; // rax
  const __m128i *v27; // r8
  __int64 v28; // rdx
  const __m128i *v29; // rax
  unsigned __int64 v30; // r9
  __m128 *v31; // rdx
  __int32 v32; // ecx
  unsigned __int8 *v33; // rsi
  __int64 *v34; // rdi
  __int64 *v35; // rax
  unsigned __int8 *v36; // rdi
  int v37; // edx
  __int128 v38; // [rsp-10h] [rbp-1E0h]
  __int128 v39; // [rsp-10h] [rbp-1E0h]
  unsigned __int64 v40; // [rsp+0h] [rbp-1D0h]
  const __m128i *v41; // [rsp+8h] [rbp-1C8h]
  const __m128i *v42; // [rsp+8h] [rbp-1C8h]
  const __m128i *v43; // [rsp+10h] [rbp-1C0h]
  const __m128i *v44; // [rsp+10h] [rbp-1C0h]
  unsigned __int64 v45; // [rsp+18h] [rbp-1B8h]
  int v46; // [rsp+18h] [rbp-1B8h]
  int v47; // [rsp+18h] [rbp-1B8h]
  unsigned __int8 v50; // [rsp+37h] [rbp-199h]
  const void **v51; // [rsp+38h] [rbp-198h]
  __m128i v52; // [rsp+60h] [rbp-170h] BYREF
  __int64 v53; // [rsp+70h] [rbp-160h] BYREF
  int v54; // [rsp+78h] [rbp-158h]
  _BYTE *v55; // [rsp+80h] [rbp-150h] BYREF
  __int64 v56; // [rsp+88h] [rbp-148h]
  _BYTE v57[128]; // [rsp+90h] [rbp-140h] BYREF
  __m128i v58; // [rsp+110h] [rbp-C0h] BYREF
  char v59[8]; // [rsp+120h] [rbp-B0h] BYREF
  const void **v60; // [rsp+128h] [rbp-A8h]

  v8 = *(_QWORD *)(a2 + 72);
  v53 = v8;
  if ( v8 )
    sub_1623A60((__int64)&v53, v8, 2);
  v9 = *(_QWORD *)(a1 + 8);
  v54 = *(_DWORD *)(a2 + 64);
  v10 = *(char **)(a2 + 40);
  v11 = *v10;
  v12 = *((_QWORD *)v10 + 1);
  LOBYTE(v55) = v11;
  v56 = v12;
  sub_1D19A30((__int64)&v58, v9, &v55);
  v13 = _mm_loadu_si128(&v58);
  v50 = v59[0];
  v52 = v13;
  v51 = v60;
  if ( v58.m128i_i8[0] )
    v14 = word_4305480[(unsigned __int8)(v58.m128i_i8[0] - 14)];
  else
    v14 = (unsigned int)sub_1F58D30((__int64)&v52);
  v15 = *(const __m128i **)(a2 + 32);
  v16 = 40 * v14;
  v55 = v57;
  v56 = 0x800000000LL;
  v17 = (const __m128i *)((char *)v15 + 40 * v14);
  v18 = 0xCCCCCCCCCCCCCCCDLL * ((40 * v14) >> 3);
  if ( v16 > 0x140 )
  {
    v40 = v16;
    v41 = v15;
    v43 = v17;
    v46 = v18;
    sub_16CD150((__int64)&v55, v57, v18, 16, (int)&v55, (int)v17);
    v20 = v56;
    v21 = v55;
    LODWORD(v18) = v46;
    v17 = v43;
    v15 = v41;
    v16 = v40;
    v19 = (__m128 *)&v55[16 * (unsigned int)v56];
  }
  else
  {
    v19 = (__m128 *)v57;
    v20 = 0;
    v21 = v57;
  }
  if ( v17 != v15 )
  {
    do
    {
      if ( v19 )
      {
        a5 = (__m128)_mm_loadu_si128(v15);
        *v19 = a5;
      }
      v15 = (const __m128i *)((char *)v15 + 40);
      ++v19;
    }
    while ( v17 != v15 );
    v21 = v55;
    v20 = v56;
  }
  v22 = v18 + v20;
  v23 = *(__int64 **)(a1 + 8);
  LODWORD(v56) = v22;
  *((_QWORD *)&v38 + 1) = v22;
  *(_QWORD *)&v38 = v21;
  v45 = v16;
  *(_QWORD *)a3 = sub_1D359D0(
                    v23,
                    104,
                    (__int64)&v53,
                    v52.m128i_i64[0],
                    (const void **)v52.m128i_i64[1],
                    0,
                    *(double *)a5.m128_u64,
                    *(double *)a6.m128_u64,
                    v13,
                    v38);
  v58.m128i_i64[0] = (__int64)v59;
  *(_DWORD *)(a3 + 8) = v24;
  v25 = *(unsigned int *)(a2 + 56);
  v26 = *(_QWORD *)(a2 + 32);
  v58.m128i_i64[1] = 0x800000000LL;
  v25 *= 40;
  v27 = (const __m128i *)(v26 + v25);
  v28 = v25 - v45;
  v29 = (const __m128i *)(v45 + v26);
  v30 = 0xCCCCCCCCCCCCCCCDLL * (v28 >> 3);
  if ( (unsigned __int64)v28 > 0x140 )
  {
    v42 = v29;
    v44 = v27;
    v47 = -858993459 * (v28 >> 3);
    sub_16CD150((__int64)&v58, v59, 0xCCCCCCCCCCCCCCCDLL * (v28 >> 3), 16, (int)v27, v30);
    v32 = v58.m128i_i32[2];
    v33 = (unsigned __int8 *)v58.m128i_i64[0];
    LODWORD(v30) = v47;
    v27 = v44;
    v29 = v42;
    v31 = (__m128 *)(v58.m128i_i64[0] + 16LL * v58.m128i_u32[2]);
  }
  else
  {
    v31 = (__m128 *)v59;
    v32 = 0;
    v33 = (unsigned __int8 *)v59;
  }
  if ( v29 != v27 )
  {
    do
    {
      if ( v31 )
      {
        a6 = (__m128)_mm_loadu_si128(v29);
        *v31 = a6;
      }
      v29 = (const __m128i *)((char *)v29 + 40);
      ++v31;
    }
    while ( v27 != v29 );
    v33 = (unsigned __int8 *)v58.m128i_i64[0];
    v32 = v58.m128i_i32[2];
  }
  v34 = *(__int64 **)(a1 + 8);
  v58.m128i_i32[2] = v32 + v30;
  *((_QWORD *)&v39 + 1) = (unsigned int)(v32 + v30);
  *(_QWORD *)&v39 = v33;
  v35 = sub_1D359D0(v34, 104, (__int64)&v53, v50, v51, 0, *(double *)a5.m128_u64, *(double *)a6.m128_u64, v13, v39);
  v36 = (unsigned __int8 *)v58.m128i_i64[0];
  *(_QWORD *)a4 = v35;
  *(_DWORD *)(a4 + 8) = v37;
  if ( v36 != (unsigned __int8 *)v59 )
    _libc_free((unsigned __int64)v36);
  if ( v55 != v57 )
    _libc_free((unsigned __int64)v55);
  if ( v53 )
    sub_161E7C0((__int64)&v53, v53);
}
