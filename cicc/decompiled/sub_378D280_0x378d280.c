// Function: sub_378D280
// Address: 0x378d280
//
__m128i *__fastcall sub_378D280(unsigned __int64 *a1, __int64 a2, __m128i a3)
{
  __int64 v4; // rsi
  unsigned __int64 v5; // r10
  const __m128i *v6; // r13
  unsigned __int64 v7; // rax
  __int64 v8; // rdi
  int v9; // edx
  __int64 v10; // rcx
  __int64 v11; // r11
  __int32 v12; // esi
  int v13; // r14d
  __int64 v14; // r15
  int v15; // r9d
  _QWORD *v16; // rsi
  __int64 v17; // rcx
  __int64 v18; // rdx
  __int64 v19; // rdx
  __m128i v20; // xmm4
  __int64 v21; // rcx
  _QWORD *v22; // rsi
  __m128i v23; // xmm0
  __int64 v24; // rdi
  __int64 v25; // rdx
  __int64 v26; // rdx
  __int64 v27; // rdx
  __m128i v28; // xmm6
  __m128i v29; // xmm7
  __int128 *v30; // rdi
  __m128i *v31; // rax
  __int32 v32; // edx
  __m128i v33; // rax
  _QWORD *v34; // r15
  __m128i *v35; // rax
  __int32 v36; // edx
  __m128i *v37; // r14
  unsigned __int64 *v39; // [rsp-20h] [rbp-200h]
  unsigned __int64 v40; // [rsp+0h] [rbp-1E0h]
  __int64 v41; // [rsp+8h] [rbp-1D8h]
  unsigned __int64 v42; // [rsp+10h] [rbp-1D0h]
  __int64 v43; // [rsp+18h] [rbp-1C8h]
  __m128i *v44; // [rsp+20h] [rbp-1C0h]
  __m128i *v45; // [rsp+28h] [rbp-1B8h]
  _OWORD *v46; // [rsp+30h] [rbp-1B0h]
  __int64 v47; // [rsp+38h] [rbp-1A8h]
  __int128 *v48; // [rsp+40h] [rbp-1A0h]
  __int64 v49; // [rsp+48h] [rbp-198h]
  __int64 v50; // [rsp+50h] [rbp-190h]
  int v51; // [rsp+5Ch] [rbp-184h]
  __int64 v52; // [rsp+60h] [rbp-180h]
  const __m128i *v53; // [rsp+68h] [rbp-178h]
  __int64 v54; // [rsp+70h] [rbp-170h]
  int v55; // [rsp+78h] [rbp-168h]
  __int32 v56; // [rsp+7Ch] [rbp-164h]
  __int64 v57; // [rsp+80h] [rbp-160h]
  int v58; // [rsp+88h] [rbp-158h]
  int v59; // [rsp+8Ch] [rbp-154h]
  __m128i *v60; // [rsp+90h] [rbp-150h]
  __int64 v61; // [rsp+98h] [rbp-148h]
  __int64 v62; // [rsp+A0h] [rbp-140h] BYREF
  int v63; // [rsp+A8h] [rbp-138h]
  __m128i v64; // [rsp+B0h] [rbp-130h] BYREF
  __m128i v65; // [rsp+C0h] [rbp-120h] BYREF
  __m128i v66; // [rsp+D0h] [rbp-110h] BYREF
  __int64 v67; // [rsp+E0h] [rbp-100h]
  __int32 v68; // [rsp+E8h] [rbp-F8h]
  __int64 v69; // [rsp+F0h] [rbp-F0h]
  __int64 v70; // [rsp+F8h] [rbp-E8h]
  __int64 v71; // [rsp+100h] [rbp-E0h]
  int v72; // [rsp+108h] [rbp-D8h]
  __int64 v73; // [rsp+110h] [rbp-D0h]
  unsigned __int64 v74; // [rsp+118h] [rbp-C8h]
  __int64 v75; // [rsp+120h] [rbp-C0h]
  int v76; // [rsp+128h] [rbp-B8h]
  __int64 v77; // [rsp+130h] [rbp-B0h]
  int v78; // [rsp+138h] [rbp-A8h]
  __m128i v79; // [rsp+140h] [rbp-A0h] BYREF
  __m128i v80; // [rsp+150h] [rbp-90h] BYREF
  __m128i *v81; // [rsp+160h] [rbp-80h]
  __m128i *v82; // [rsp+168h] [rbp-78h]
  __int64 v83; // [rsp+170h] [rbp-70h]
  int v84; // [rsp+178h] [rbp-68h]
  __int64 v85; // [rsp+180h] [rbp-60h]
  __int64 v86; // [rsp+188h] [rbp-58h]
  __int64 v87; // [rsp+190h] [rbp-50h]
  int v88; // [rsp+198h] [rbp-48h]
  __int64 v89; // [rsp+1A0h] [rbp-40h]
  int v90; // [rsp+1A8h] [rbp-38h]

  v4 = *(_QWORD *)(a2 + 80);
  v60 = (__m128i *)a1;
  v62 = v4;
  if ( v4 )
    sub_B96E90((__int64)&v62, v4, 1);
  v5 = *(_QWORD *)(a2 + 104);
  v6 = *(const __m128i **)(a2 + 112);
  v63 = *(_DWORD *)(a2 + 72);
  v7 = *(_QWORD *)(a2 + 40);
  v40 = v5;
  v8 = *(_QWORD *)(v7 + 40);
  v9 = *(_DWORD *)(v7 + 128);
  v53 = v6;
  v10 = *(_QWORD *)(v7 + 120);
  v11 = *(_QWORD *)(v7 + 200);
  v12 = *(_DWORD *)(v7 + 48);
  v50 = v8;
  LOWORD(v8) = *(_WORD *)(a2 + 32);
  v58 = v9;
  v54 = v10;
  v13 = *(_DWORD *)(v7 + 208);
  v14 = *(_QWORD *)(v7 + 240);
  v15 = *(_DWORD *)(v7 + 248);
  v57 = v11;
  v56 = v12;
  v51 = v15;
  v16 = (_QWORD *)v60->m128i_i64[1];
  v59 = v13;
  v55 = ((unsigned __int16)v8 >> 7) & 7;
  LOWORD(v13) = *(_WORD *)(a2 + 96);
  v52 = v14;
  v64.m128i_i16[0] = 0;
  v64.m128i_i64[1] = 0;
  v65.m128i_i16[0] = 0;
  v65.m128i_i64[1] = 0;
  v17 = *(_QWORD *)(v7 + 160);
  v18 = *(unsigned int *)(v7 + 168);
  v46 = (_OWORD *)v7;
  v49 = (__int64)v16;
  v19 = *(_QWORD *)(v17 + 48) + 16 * v18;
  LOWORD(v17) = *(_WORD *)v19;
  v66.m128i_i64[1] = *(_QWORD *)(v19 + 8);
  v66.m128i_i16[0] = v17;
  sub_33D0340((__int64)&v79, (__int64)v16, v66.m128i_i64);
  v20 = _mm_loadu_si128(&v80);
  v64 = _mm_loadu_si128(&v79);
  v65 = v20;
  v44 = &v65;
  v45 = &v64;
  sub_3408290((__int64)&v79, v16, v46 + 10, (__int64)&v62, (unsigned int *)&v64, (unsigned int *)&v65, a3);
  v21 = *(_QWORD *)(a2 + 40);
  v65.m128i_i16[0] = 0;
  v22 = (_QWORD *)v60->m128i_i64[1];
  v64.m128i_i64[1] = 0;
  v64.m128i_i16[0] = 0;
  v23 = _mm_cvtsi32_si128(v80.m128i_u32[2]);
  v41 = v79.m128i_i64[0];
  v48 = (__int128 *)v21;
  v65.m128i_i64[1] = 0;
  v24 = *(_QWORD *)(v21 + 80);
  v25 = *(unsigned int *)(v21 + 88);
  v49 = v80.m128i_i64[0];
  v43 = v23.m128i_i64[0];
  v26 = *(_QWORD *)(v24 + 48) + 16 * v25;
  v42 = _mm_cvtsi32_si128(v79.m128i_u32[2]).m128i_u64[0];
  LOWORD(v24) = *(_WORD *)v26;
  v27 = *(_QWORD *)(v26 + 8);
  v46 = v22;
  v66.m128i_i16[0] = v24;
  v66.m128i_i64[1] = v27;
  sub_33D0340((__int64)&v79, (__int64)v22, v66.m128i_i64);
  v28 = _mm_loadu_si128(&v80);
  v64 = _mm_loadu_si128(&v79);
  v65 = v28;
  sub_3408290((__int64)&v79, v22, v48 + 5, (__int64)&v62, (unsigned int *)&v64, (unsigned int *)&v65, v23);
  v29 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a2 + 40));
  v44 = (__m128i *)v80.m128i_i64[0];
  v67 = v50;
  v69 = v79.m128i_i64[0];
  v70 = v79.m128i_u32[2];
  v66 = v29;
  v74 = v42;
  v75 = v57;
  v45 = (__m128i *)_mm_cvtsi32_si128(v80.m128i_u32[2]).m128i_u64[0];
  v76 = v59;
  v77 = v14;
  v78 = v51;
  v68 = v56;
  v71 = v54;
  v72 = v58;
  v73 = v41;
  v30 = (__int128 *)v60->m128i_i64[1];
  v46 = &v66;
  v47 = 7;
  v48 = v30;
  v31 = sub_33ED250((__int64)v30, 1, 0);
  v39 = (unsigned __int64 *)v46;
  v46 = (_OWORD *)v40;
  v33.m128i_i64[0] = (__int64)sub_33E74D0(v48, (unsigned __int64)v31, v32, v13, v40, (__int64)&v62, v39, v47, v53, v55);
  v79 = v33;
  v34 = (_QWORD *)v60->m128i_i64[1];
  v80.m128i_i64[0] = v50;
  v81 = v44;
  v80.m128i_i32[2] = v56;
  v83 = v54;
  v84 = v58;
  v87 = v57;
  v88 = v59;
  v89 = v52;
  v90 = v51;
  v61 = 7;
  v82 = v45;
  v86 = v43;
  v85 = v49;
  v60 = &v79;
  v35 = sub_33ED250((__int64)v34, 1, 0);
  v37 = sub_33E74D0(
          v34,
          (unsigned __int64)v35,
          v36,
          v13,
          (unsigned __int64)v46,
          (__int64)&v62,
          (unsigned __int64 *)v60,
          v61,
          v53,
          v55);
  if ( v62 )
    sub_B91220((__int64)&v62, v62);
  return v37;
}
