// Function: sub_3424A90
// Address: 0x3424a90
//
void __fastcall sub_3424A90(__int64 *a1, unsigned __int64 *a2, __int64 a3)
{
  unsigned int v3; // r15d
  const __m128i *v5; // rbx
  __int64 v6; // rax
  __m128i v7; // xmm0
  __int64 v8; // rbx
  __int64 v9; // r12
  __int64 v10; // rax
  __m128i v11; // xmm0
  __int64 v12; // rdx
  __int64 v13; // rax
  const __m128i *v14; // rbx
  __int64 v15; // rax
  __m128i v16; // xmm0
  __int64 v17; // r12
  __int64 v18; // rbx
  __int64 v19; // rax
  __m128i v20; // xmm0
  __int64 v21; // rdx
  __int64 v22; // rax
  const __m128i *v23; // rbx
  __int64 v24; // rax
  __m128i v25; // xmm0
  __int64 v26; // r12
  __int64 v27; // rbx
  __int64 v28; // rax
  __m128i v29; // xmm0
  __int64 v30; // rdx
  __int64 v31; // rax
  const __m128i *v32; // rbx
  __int64 v33; // rax
  __m128i v34; // xmm0
  __int64 v35; // r12
  __int64 v36; // rbx
  __int64 v37; // rax
  __m128i v38; // xmm0
  __int64 v39; // rdx
  __int64 v40; // rax
  unsigned __int64 v41; // rbx
  unsigned __int64 v42; // rsi
  __int64 v43; // rdx
  __int64 v44; // rcx
  unsigned int v45; // r12d
  __int64 v46; // rbx
  const __m128i *v47; // rdx
  __int64 v48; // rcx
  _QWORD *v49; // rax
  _QWORD *v50; // rbx
  int v51; // eax
  __int64 v52; // rdx
  _QWORD *v53; // rax
  unsigned int v54; // edx
  int v55; // edx
  __int64 v56; // rcx
  __int64 v57; // rax
  __int64 v58; // rax
  __int64 (*v59)(); // rax
  int v60; // ebx
  unsigned __int8 *v61; // rax
  unsigned int v62; // edx
  unsigned int v63; // ebx
  __int64 v64; // rax
  __int64 v65; // r8
  __int64 v66; // rdx
  __int64 v67; // rax
  const __m128i *v68; // rbx
  __m128i *v69; // rsi
  _QWORD *v70; // rbx
  unsigned __int64 v71; // r12
  __int64 v72; // rax
  __m128i v73; // xmm0
  __int64 v74; // rbx
  __int64 v75; // r12
  __int64 v76; // rax
  __m128i v77; // xmm0
  __int64 v78; // rdx
  __int64 v79; // rax
  int v81; // [rsp+24h] [rbp-ECh]
  unsigned __int8 *v82; // [rsp+28h] [rbp-E8h]
  __m128i v83; // [rsp+30h] [rbp-E0h] BYREF
  __m128i v84; // [rsp+40h] [rbp-D0h]
  unsigned __int8 *v85; // [rsp+50h] [rbp-C0h]
  __int64 v86; // [rsp+58h] [rbp-B8h]
  __m128i v87; // [rsp+60h] [rbp-B0h]
  __m128i v88; // [rsp+70h] [rbp-A0h]
  __m128i v89; // [rsp+80h] [rbp-90h]
  __m128i v90; // [rsp+90h] [rbp-80h]
  _QWORD v91[2]; // [rsp+A0h] [rbp-70h] BYREF
  __int64 v92; // [rsp+B0h] [rbp-60h]
  const __m128i *v93; // [rsp+C0h] [rbp-50h] BYREF
  const __m128i *v94; // [rsp+C8h] [rbp-48h]
  __int64 v95; // [rsp+D0h] [rbp-40h]

  v5 = (const __m128i *)*a2;
  v91[1] = v91;
  v91[0] = v91;
  v92 = 0;
  v6 = sub_22077B0(0x98u);
  v7 = _mm_loadu_si128(v5);
  v8 = v5->m128i_i64[0];
  v9 = v6;
  v83 = v7;
  v10 = sub_33ECD10(1u);
  v11 = _mm_load_si128(&v83);
  *(_QWORD *)(v9 + 64) = v10;
  v12 = v9 + 112;
  *(_QWORD *)(v9 + 80) = 0x100000000LL;
  *(_QWORD *)(v9 + 104) = 0xFFFFFFFFLL;
  *(_QWORD *)(v9 + 144) = 0;
  *(_QWORD *)(v9 + 16) = 0;
  *(_QWORD *)(v9 + 24) = 0;
  *(_QWORD *)(v9 + 32) = 0;
  *(_QWORD *)(v9 + 40) = 328;
  *(_WORD *)(v9 + 50) = -1;
  *(_DWORD *)(v9 + 52) = -1;
  *(_QWORD *)(v9 + 56) = 0;
  *(_QWORD *)(v9 + 72) = 0;
  *(_DWORD *)(v9 + 88) = 0;
  *(_QWORD *)(v9 + 96) = 0;
  *(_WORD *)(v9 + 48) = 0;
  *(_QWORD *)(v9 + 136) = 0;
  *(_QWORD *)(v9 + 128) = v9 + 16;
  v90 = v11;
  *(_QWORD *)(v9 + 112) = v11.m128i_i64[0];
  *(_DWORD *)(v9 + 120) = v90.m128i_i32[2];
  v13 = *(_QWORD *)(v8 + 56);
  *(_QWORD *)(v9 + 144) = v13;
  if ( v13 )
    *(_QWORD *)(v13 + 24) = v9 + 144;
  *(_QWORD *)(v9 + 136) = v8 + 56;
  *(_QWORD *)(v8 + 56) = v12;
  *(_QWORD *)(v9 + 56) = v12;
  *(_DWORD *)(v9 + 80) = 1;
  sub_2208C80((_QWORD *)v9, (__int64)v91);
  v14 = (const __m128i *)*a2;
  ++v92;
  v15 = sub_22077B0(0x98u);
  v16 = _mm_loadu_si128(v14 + 1);
  v17 = v15;
  v18 = v14[1].m128i_i64[0];
  v83 = v16;
  v19 = sub_33ECD10(1u);
  v20 = _mm_load_si128(&v83);
  *(_QWORD *)(v17 + 64) = v19;
  v21 = v17 + 112;
  *(_QWORD *)(v17 + 80) = 0x100000000LL;
  *(_QWORD *)(v17 + 104) = 0xFFFFFFFFLL;
  *(_QWORD *)(v17 + 144) = 0;
  *(_QWORD *)(v17 + 16) = 0;
  *(_QWORD *)(v17 + 24) = 0;
  *(_QWORD *)(v17 + 32) = 0;
  *(_QWORD *)(v17 + 40) = 328;
  *(_WORD *)(v17 + 50) = -1;
  *(_DWORD *)(v17 + 52) = -1;
  *(_QWORD *)(v17 + 56) = 0;
  *(_QWORD *)(v17 + 72) = 0;
  *(_DWORD *)(v17 + 88) = 0;
  *(_QWORD *)(v17 + 96) = 0;
  *(_WORD *)(v17 + 48) = 0;
  *(_QWORD *)(v17 + 136) = 0;
  *(_QWORD *)(v17 + 128) = v17 + 16;
  v89 = v20;
  *(_QWORD *)(v17 + 112) = v20.m128i_i64[0];
  *(_DWORD *)(v17 + 120) = v89.m128i_i32[2];
  v22 = *(_QWORD *)(v18 + 56);
  *(_QWORD *)(v17 + 144) = v22;
  if ( v22 )
    *(_QWORD *)(v22 + 24) = v17 + 144;
  *(_QWORD *)(v17 + 136) = v18 + 56;
  *(_QWORD *)(v18 + 56) = v21;
  *(_QWORD *)(v17 + 56) = v21;
  *(_DWORD *)(v17 + 80) = 1;
  sub_2208C80((_QWORD *)v17, (__int64)v91);
  v23 = (const __m128i *)*a2;
  ++v92;
  v24 = sub_22077B0(0x98u);
  v25 = _mm_loadu_si128(v23 + 2);
  v26 = v24;
  v27 = v23[2].m128i_i64[0];
  v83 = v25;
  v28 = sub_33ECD10(1u);
  v29 = _mm_load_si128(&v83);
  *(_QWORD *)(v26 + 64) = v28;
  v30 = v26 + 112;
  *(_QWORD *)(v26 + 80) = 0x100000000LL;
  *(_QWORD *)(v26 + 104) = 0xFFFFFFFFLL;
  *(_WORD *)(v26 + 50) = -1;
  *(_QWORD *)(v26 + 144) = 0;
  *(_QWORD *)(v26 + 16) = 0;
  *(_QWORD *)(v26 + 24) = 0;
  *(_QWORD *)(v26 + 32) = 0;
  *(_QWORD *)(v26 + 40) = 328;
  *(_DWORD *)(v26 + 52) = -1;
  *(_QWORD *)(v26 + 56) = 0;
  *(_QWORD *)(v26 + 72) = 0;
  *(_DWORD *)(v26 + 88) = 0;
  *(_QWORD *)(v26 + 96) = 0;
  *(_WORD *)(v26 + 48) = 0;
  *(_QWORD *)(v26 + 136) = 0;
  *(_QWORD *)(v26 + 128) = v26 + 16;
  v88 = v29;
  *(_QWORD *)(v26 + 112) = v29.m128i_i64[0];
  *(_DWORD *)(v26 + 120) = v88.m128i_i32[2];
  v31 = *(_QWORD *)(v27 + 56);
  *(_QWORD *)(v26 + 144) = v31;
  if ( v31 )
    *(_QWORD *)(v31 + 24) = v26 + 144;
  *(_QWORD *)(v26 + 136) = v27 + 56;
  *(_QWORD *)(v27 + 56) = v30;
  *(_QWORD *)(v26 + 56) = v30;
  *(_DWORD *)(v26 + 80) = 1;
  sub_2208C80((_QWORD *)v26, (__int64)v91);
  v32 = (const __m128i *)*a2;
  ++v92;
  v33 = sub_22077B0(0x98u);
  v34 = _mm_loadu_si128(v32 + 3);
  v35 = v33;
  v36 = v32[3].m128i_i64[0];
  v83 = v34;
  v37 = sub_33ECD10(1u);
  v38 = _mm_load_si128(&v83);
  *(_QWORD *)(v35 + 64) = v37;
  *(_QWORD *)(v35 + 80) = 0x100000000LL;
  *(_QWORD *)(v35 + 104) = 0xFFFFFFFFLL;
  *(_WORD *)(v35 + 48) = 0;
  v39 = v35 + 112;
  *(_QWORD *)(v35 + 144) = 0;
  *(_QWORD *)(v35 + 16) = 0;
  *(_QWORD *)(v35 + 24) = 0;
  *(_QWORD *)(v35 + 32) = 0;
  *(_QWORD *)(v35 + 40) = 328;
  *(_WORD *)(v35 + 50) = -1;
  *(_DWORD *)(v35 + 52) = -1;
  *(_QWORD *)(v35 + 56) = 0;
  *(_QWORD *)(v35 + 72) = 0;
  *(_DWORD *)(v35 + 88) = 0;
  *(_QWORD *)(v35 + 96) = 0;
  *(_QWORD *)(v35 + 136) = 0;
  *(_QWORD *)(v35 + 128) = v35 + 16;
  v87 = v38;
  *(_QWORD *)(v35 + 112) = v38.m128i_i64[0];
  *(_DWORD *)(v35 + 120) = v87.m128i_i32[2];
  v40 = *(_QWORD *)(v36 + 56);
  *(_QWORD *)(v35 + 144) = v40;
  if ( v40 )
    *(_QWORD *)(v40 + 24) = v35 + 144;
  *(_QWORD *)(v36 + 56) = v39;
  *(_QWORD *)(v35 + 136) = v36 + 56;
  *(_QWORD *)(v35 + 56) = v39;
  *(_DWORD *)(v35 + 80) = 1;
  sub_2208C80((_QWORD *)v35, (__int64)v91);
  v41 = a2[1];
  v42 = *a2;
  ++v92;
  v43 = (__int64)(v41 - v42) >> 4;
  v44 = (unsigned int)(v43 - 1);
  if ( *(_WORD *)(*(_QWORD *)(*(_QWORD *)(v42 + 16 * v44) + 48LL) + 16LL * *(unsigned int *)(v42 + 16 * v44 + 8)) != 262 )
    LODWORD(v44) = (__int64)(v41 - v42) >> 4;
  v81 = v44;
  if ( (_DWORD)v44 != 4 )
  {
    v45 = 4;
    do
    {
      v47 = (const __m128i *)(v42 + 16LL * v45);
      v48 = *(_QWORD *)(v47->m128i_i64[0] + 96);
      v49 = *(_QWORD **)(v48 + 24);
      if ( *(_DWORD *)(v48 + 32) > 0x40u )
        v49 = (_QWORD *)*v49;
      LODWORD(v50) = (_DWORD)v49;
      v51 = (unsigned __int8)v49 & 7;
      if ( v51 == 7 || v51 == 6 )
      {
        if ( (int)v50 < 0 )
        {
          v52 = *(_QWORD *)(*(_QWORD *)(v42 + 64) + 96LL);
          v53 = *(_QWORD **)(v52 + 24);
          if ( *(_DWORD *)(v52 + 32) > 0x40u )
            v53 = (_QWORD *)*v53;
          v54 = (unsigned int)v50;
          LODWORD(v50) = (_DWORD)v53;
          v55 = HIWORD(v54) & 0x7FFF;
          if ( v55 )
          {
            LODWORD(v56) = 4;
            do
            {
              v56 = (unsigned int)v56 + ((unsigned __int16)v50 >> 3) + 1;
              v57 = *(_QWORD *)(*(_QWORD *)(v42 + 16 * v56) + 96LL);
              v50 = *(_QWORD **)(v57 + 24);
              if ( *(_DWORD *)(v57 + 32) > 0x40u )
                v50 = (_QWORD *)*v50;
              --v55;
            }
            while ( v55 );
          }
        }
        v58 = *a1;
        v93 = 0;
        v94 = 0;
        v59 = *(__int64 (**)())(v58 + 56);
        v95 = 0;
        if ( v59 == sub_341E350
          || (v83.m128i_i32[0] = ((unsigned int)v50 >> 16) & 0x7FFF,
              ((unsigned __int8 (__fastcall *)(__int64 *, unsigned __int64, _QWORD, const __m128i **))v59)(
                a1,
                16LL * (v45 + 1) + v42,
                v83.m128i_u32[0],
                &v93)) )
        {
          sub_C64ED0("Could not match memory address.  Inline asm failure!", 1u);
        }
        v60 = (unsigned __int8)v50 & 7;
        if ( v60 != 6 )
          v60 = 7;
        LOWORD(v3) = 7;
        v61 = sub_3400BD0(
                a1[8],
                (v83.m128i_i32[0] << 16) | (v60 | (8 * (unsigned int)(v94 - v93))) & 0x8000FFFF,
                a3,
                v3,
                0,
                1u,
                v38,
                0);
        v63 = v62;
        v82 = v61;
        v83.m128i_i64[0] = sub_22077B0(0x98u);
        v64 = sub_33ECD10(1u);
        v65 = v83.m128i_i64[0];
        *(_QWORD *)(v83.m128i_i64[0] + 64) = v64;
        v66 = v65 + 112;
        *(_QWORD *)(v65 + 80) = 0x100000000LL;
        *(_QWORD *)(v65 + 104) = 0xFFFFFFFFLL;
        *(_WORD *)(v65 + 50) = -1;
        *(_QWORD *)(v65 + 144) = 0;
        *(_QWORD *)(v65 + 16) = 0;
        *(_QWORD *)(v65 + 24) = 0;
        *(_QWORD *)(v65 + 32) = 0;
        *(_QWORD *)(v65 + 40) = 328;
        *(_DWORD *)(v65 + 52) = -1;
        *(_QWORD *)(v65 + 56) = 0;
        *(_QWORD *)(v65 + 72) = 0;
        *(_DWORD *)(v65 + 88) = 0;
        *(_QWORD *)(v65 + 96) = 0;
        *(_WORD *)(v65 + 48) = 0;
        *(_QWORD *)(v65 + 136) = 0;
        *(_QWORD *)(v65 + 128) = v65 + 16;
        v85 = v82;
        v86 = v63;
        v67 = *((_QWORD *)v82 + 7);
        *(_QWORD *)(v65 + 112) = v82;
        *(_DWORD *)(v65 + 120) = v63;
        *(_QWORD *)(v65 + 144) = v67;
        if ( v67 )
          *(_QWORD *)(v67 + 24) = v65 + 144;
        *(_QWORD *)(v65 + 136) = v82 + 56;
        v45 += 2;
        *((_QWORD *)v82 + 7) = v66;
        *(_QWORD *)(v65 + 56) = v66;
        *(_DWORD *)(v65 + 80) = 1;
        sub_2208C80((_QWORD *)v65, (__int64)v91);
        ++v92;
        sub_3424060((__int64)v91, (__int64)v91, v93, v94);
        if ( v93 )
          j_j___libc_free_0((unsigned __int64)v93);
      }
      else
      {
        v46 = (unsigned int)((unsigned __int16)v50 >> 3) + 1;
        v45 += v46;
        sub_3424060((__int64)v91, (__int64)v91, v47, &v47[v46]);
      }
      v42 = *a2;
    }
    while ( v81 != v45 );
    v41 = a2[1];
    v43 = (__int64)(v41 - v42) >> 4;
  }
  if ( v81 != v43 )
  {
    v72 = sub_22077B0(0x98u);
    v73 = _mm_loadu_si128((const __m128i *)(v41 - 16));
    v74 = *(_QWORD *)(v41 - 16);
    v75 = v72;
    v83 = v73;
    v76 = sub_33ECD10(1u);
    v77 = _mm_load_si128(&v83);
    *(_QWORD *)(v75 + 64) = v76;
    *(_QWORD *)(v75 + 80) = 0x100000000LL;
    *(_QWORD *)(v75 + 104) = 0xFFFFFFFFLL;
    *(_WORD *)(v75 + 50) = -1;
    v78 = v75 + 112;
    *(_WORD *)(v75 + 48) = 0;
    *(_QWORD *)(v75 + 144) = 0;
    *(_QWORD *)(v75 + 16) = 0;
    *(_QWORD *)(v75 + 24) = 0;
    *(_QWORD *)(v75 + 32) = 0;
    *(_QWORD *)(v75 + 40) = 328;
    *(_DWORD *)(v75 + 52) = -1;
    *(_QWORD *)(v75 + 56) = 0;
    *(_QWORD *)(v75 + 72) = 0;
    *(_DWORD *)(v75 + 88) = 0;
    *(_QWORD *)(v75 + 96) = 0;
    *(_QWORD *)(v75 + 136) = 0;
    *(_QWORD *)(v75 + 128) = v75 + 16;
    v84 = v77;
    *(_QWORD *)(v75 + 112) = v77.m128i_i64[0];
    *(_DWORD *)(v75 + 120) = v84.m128i_i32[2];
    v79 = *(_QWORD *)(v74 + 56);
    *(_QWORD *)(v75 + 144) = v79;
    if ( v79 )
      *(_QWORD *)(v79 + 24) = v75 + 144;
    *(_QWORD *)(v74 + 56) = v78;
    *(_QWORD *)(v75 + 136) = v74 + 56;
    *(_DWORD *)(v75 + 80) = 1;
    *(_QWORD *)(v75 + 56) = v78;
    sub_2208C80((_QWORD *)v75, (__int64)v91);
    ++v92;
    v42 = *a2;
    v41 = a2[1];
  }
  if ( v41 != v42 )
    a2[1] = v42;
  v68 = (const __m128i *)v91[0];
  if ( (_QWORD *)v91[0] != v91 )
  {
    do
    {
      while ( 1 )
      {
        v69 = (__m128i *)a2[1];
        if ( v69 != (__m128i *)a2[2] )
          break;
        sub_33764F0(a2, v69, v68 + 7);
        v68 = (const __m128i *)v68->m128i_i64[0];
        if ( v68 == (const __m128i *)v91 )
          goto LABEL_47;
      }
      if ( v69 )
      {
        *v69 = _mm_loadu_si128(v68 + 7);
        v69 = (__m128i *)a2[1];
      }
      a2[1] = (unsigned __int64)&v69[1];
      v68 = (const __m128i *)v68->m128i_i64[0];
    }
    while ( v68 != (const __m128i *)v91 );
LABEL_47:
    v70 = (_QWORD *)v91[0];
    while ( v70 != v91 )
    {
      v71 = (unsigned __int64)v70;
      v70 = (_QWORD *)*v70;
      sub_33CF710(v71 + 16);
      j_j___libc_free_0(v71);
    }
  }
}
