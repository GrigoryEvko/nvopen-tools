// Function: sub_3781AE0
// Address: 0x3781ae0
//
void __fastcall sub_3781AE0(__int64 *a1, __int64 a2, __int64 a3, const __m128i *a4, __m128i a5)
{
  __int64 v7; // rsi
  int v8; // eax
  __int64 v9; // rsi
  __int16 *v10; // rax
  __int16 v11; // dx
  __int64 v12; // rax
  __int64 v13; // rsi
  unsigned __int16 *v14; // rax
  __int64 *v15; // r9
  __int64 v16; // rsi
  __int64 v17; // r11
  __int128 *v18; // r10
  int v19; // eax
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // rdx
  __m128i v23; // xmm4
  __int64 v24; // rdi
  __m128i *v25; // rcx
  int v26; // r15d
  unsigned __int32 v27; // r11d
  _QWORD *v28; // rdi
  __int64 v29; // rbx
  unsigned __int8 *v30; // rax
  __int64 v31; // rcx
  __int64 v32; // rdx
  __int64 v33; // r8
  __int64 v34; // rsi
  const __m128i *v35; // rbx
  __int64 v36; // rdx
  __int64 v37; // rsi
  _QWORD *v38; // rsi
  __int64 v39; // r9
  __int64 v40; // r8
  __int64 v41; // rdx
  __int64 v42; // rcx
  __int64 v43; // rdx
  _QWORD *v44; // rdi
  __m128i v45; // xmm5
  __int64 v46; // rbx
  unsigned __int8 *v47; // rax
  __int64 v48; // rdi
  __int64 v49; // rdx
  unsigned __int64 v50; // xmm0_8
  __int64 v51; // rsi
  __int64 v52; // rcx
  __int64 v53; // r8
  _QWORD *v54; // rdi
  __m128i *v55; // r12
  __m128i v56; // xmm7
  unsigned __int8 *v57; // rax
  __int64 v58; // rdx
  __int64 v59; // r13
  unsigned __int8 *v60; // rax
  __int64 v61; // rdx
  __int64 v62; // rcx
  __int64 v63; // r8
  __int64 v64; // rax
  const __m128i *v65; // rbx
  __int64 v66; // rdx
  unsigned __int32 v67; // [rsp+Ch] [rbp-184h]
  __m128i v68; // [rsp+10h] [rbp-180h]
  __int64 v69; // [rsp+20h] [rbp-170h]
  unsigned __int64 v70; // [rsp+28h] [rbp-168h]
  __int128 *v71; // [rsp+30h] [rbp-160h]
  __int64 v72; // [rsp+38h] [rbp-158h]
  __int64 v73; // [rsp+40h] [rbp-150h]
  __int64 v74; // [rsp+48h] [rbp-148h]
  __m128i v75; // [rsp+50h] [rbp-140h]
  __int64 v76; // [rsp+60h] [rbp-130h]
  __m128i *v77; // [rsp+68h] [rbp-128h]
  __m128i v78; // [rsp+70h] [rbp-120h]
  unsigned __int8 *v79; // [rsp+80h] [rbp-110h]
  __int64 v80; // [rsp+88h] [rbp-108h]
  unsigned __int8 *v81; // [rsp+90h] [rbp-100h]
  __int64 v82; // [rsp+98h] [rbp-F8h]
  unsigned __int8 *v83; // [rsp+A0h] [rbp-F0h]
  __int64 v84; // [rsp+A8h] [rbp-E8h]
  unsigned __int8 *v85; // [rsp+B0h] [rbp-E0h]
  __int64 v86; // [rsp+B8h] [rbp-D8h]
  unsigned __int8 *v87; // [rsp+C0h] [rbp-D0h]
  __int64 v88; // [rsp+C8h] [rbp-C8h]
  unsigned __int8 *v89; // [rsp+D0h] [rbp-C0h]
  __int64 v90; // [rsp+D8h] [rbp-B8h]
  __int64 v91; // [rsp+E0h] [rbp-B0h] BYREF
  int v92; // [rsp+E8h] [rbp-A8h]
  __int64 v93; // [rsp+F0h] [rbp-A0h] BYREF
  int v94; // [rsp+F8h] [rbp-98h]
  __m128i v95; // [rsp+100h] [rbp-90h] BYREF
  __m128i v96; // [rsp+110h] [rbp-80h] BYREF
  __int64 v97; // [rsp+120h] [rbp-70h] BYREF
  __int64 v98; // [rsp+128h] [rbp-68h]
  __m128i v99; // [rsp+130h] [rbp-60h] BYREF
  __m128i v100; // [rsp+140h] [rbp-50h] BYREF
  __int64 v101; // [rsp+150h] [rbp-40h]
  unsigned __int64 v102; // [rsp+158h] [rbp-38h]

  v7 = *(_QWORD *)(a2 + 80);
  v78.m128i_i64[0] = a3;
  v77 = (__m128i *)a4;
  v91 = v7;
  if ( v7 )
    sub_B96E90((__int64)&v91, v7, 1);
  v8 = *(_DWORD *)(a2 + 72);
  v9 = a1[1];
  v75.m128i_i64[0] = (__int64)&v97;
  v92 = v8;
  v10 = *(__int16 **)(a2 + 48);
  v11 = *v10;
  v12 = *((_QWORD *)v10 + 1);
  LOWORD(v97) = v11;
  v98 = v12;
  sub_33D0340((__int64)&v99, v9, &v97);
  v13 = *a1;
  v74 = v99.m128i_i64[1];
  v76 = v99.m128i_i64[0];
  v72 = v100.m128i_i64[1];
  v73 = v100.m128i_i64[0];
  v14 = (unsigned __int16 *)(*(_QWORD *)(**(_QWORD **)(a2 + 40) + 48LL)
                           + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 40) + 8LL));
  sub_2FE6CC0((__int64)&v99, v13, *(_QWORD *)(a1[1] + 64), *v14, *((_QWORD *)v14 + 1));
  v15 = &v97;
  if ( v99.m128i_i8[0] == 6 )
  {
    sub_375E8D0(
      (__int64)a1,
      **(_QWORD **)(a2 + 40),
      *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL),
      v78.m128i_i64[0],
      (__int64)v77);
  }
  else
  {
    v16 = *(_QWORD *)(a2 + 80);
    v17 = a1[1];
    v93 = v16;
    if ( v16 )
    {
      v71 = (__int128 *)v75.m128i_i64[0];
      v75.m128i_i64[0] = v17;
      sub_B96E90((__int64)&v93, v16, 1);
      v15 = (__int64 *)v71;
      v17 = v75.m128i_i64[0];
    }
    v18 = *(__int128 **)(a2 + 40);
    v19 = *(_DWORD *)(a2 + 72);
    v95.m128i_i64[1] = 0;
    v94 = v19;
    v96.m128i_i64[1] = 0;
    v20 = *((unsigned int *)v18 + 2);
    v96.m128i_i16[0] = 0;
    v95.m128i_i16[0] = 0;
    v21 = *(_QWORD *)v18;
    v71 = v18;
    v22 = *(_QWORD *)(v21 + 48) + 16 * v20;
    v75.m128i_i64[0] = v17;
    LOWORD(v21) = *(_WORD *)v22;
    v98 = *(_QWORD *)(v22 + 8);
    LOWORD(v97) = v21;
    sub_33D0340((__int64)&v99, v17, v15);
    v23 = _mm_loadu_si128(&v100);
    v95 = _mm_loadu_si128(&v99);
    v96 = v23;
    sub_3408290((__int64)&v99, v75.m128i_i64[0], v71, (__int64)&v93, (unsigned int *)&v95, (unsigned int *)&v96, a5);
    if ( v93 )
      sub_B91220((__int64)&v93, v93);
    v24 = v78.m128i_i64[0];
    v25 = v77;
    *(_QWORD *)v78.m128i_i64[0] = v99.m128i_i64[0];
    *(_DWORD *)(v24 + 8) = v99.m128i_i32[2];
    v25->m128i_i64[0] = v100.m128i_i64[0];
    v25->m128i_i32[2] = v100.m128i_i32[2];
  }
  v26 = *(_DWORD *)(a2 + 28);
  v27 = *(_DWORD *)(a2 + 24);
  if ( *(_DWORD *)(a2 + 64) > 2u )
  {
    v67 = *(_DWORD *)(a2 + 24);
    sub_3777990(&v99, a1, *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL), a5);
    v38 = (_QWORD *)a1[1];
    v39 = *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL);
    v40 = **(unsigned __int16 **)(a2 + 48);
    v41 = *(_QWORD *)(a2 + 40);
    v68.m128i_i64[0] = v99.m128i_i64[0];
    v69 = _mm_cvtsi32_si128(v100.m128i_u32[2]).m128i_u64[0];
    v42 = *(_QWORD *)(v41 + 88);
    v43 = *(_QWORD *)(v41 + 80);
    v71 = (__int128 *)v100.m128i_i64[0];
    v68.m128i_i64[1] = _mm_cvtsi32_si128(v99.m128i_u32[2]).m128i_u64[0];
    sub_3408380(&v99, v38, v43, v42, v40, v39, a5, (__int64)&v91);
    v44 = (_QWORD *)a1[1];
    v102 = v99.m128i_u32[2];
    v45 = _mm_loadu_si128((const __m128i *)v78.m128i_i64[0]);
    v46 = v100.m128i_i64[0];
    v101 = v99.m128i_i64[0];
    v75 = v45;
    v70 = _mm_cvtsi32_si128(v100.m128i_u32[2]).m128i_u64[0];
    v75.m128i_i32[0] = v67;
    v99 = v45;
    v100 = v68;
    v47 = sub_33FBA10(v44, v67, (__int64)&v91, v76, v74, v26, (__int64)&v99, 3);
    v48 = v78.m128i_i64[0];
    v82 = v49;
    v50 = v70;
    v51 = v75.m128i_u32[0];
    v101 = v46;
    v81 = v47;
    v52 = v73;
    *(_QWORD *)v78.m128i_i64[0] = v47;
    v53 = v72;
    v100.m128i_i64[1] = v69;
    *(_DWORD *)(v48 + 8) = v82;
    v54 = (_QWORD *)a1[1];
    v55 = v77;
    v102 = v50;
    v56 = _mm_loadu_si128(v77);
    v100.m128i_i64[0] = (__int64)v71;
    v78 = v56;
    v99 = v56;
    v57 = sub_33FBA10(v54, v51, (__int64)&v91, v52, v53, v26, (__int64)&v99, 3);
    v37 = v91;
    v80 = v58;
    v79 = v57;
    v55->m128i_i64[0] = (__int64)v57;
    v55->m128i_i32[2] = v80;
    if ( v37 )
LABEL_13:
      sub_B91220((__int64)&v91, v37);
  }
  else
  {
    v28 = (_QWORD *)a1[1];
    if ( v27 == 230 )
    {
      v59 = v78.m128i_i64[0];
      v60 = sub_3405C90(
              v28,
              0xE6u,
              (__int64)&v91,
              v76,
              v74,
              v26,
              a5,
              *(_OWORD *)v78.m128i_i64[0],
              *(_OWORD *)(*(_QWORD *)(a2 + 40) + 40LL));
      v90 = v61;
      v62 = v73;
      v89 = v60;
      v63 = v72;
      *(_QWORD *)v78.m128i_i64[0] = v60;
      *(_DWORD *)(v59 + 8) = v90;
      v64 = *(_QWORD *)(a2 + 40);
      v65 = v77;
      v87 = sub_3405C90((_QWORD *)a1[1], 0xE6u, (__int64)&v91, v62, v63, v26, a5, (__int128)*v77, *(_OWORD *)(v64 + 40));
      v88 = v66;
      v77->m128i_i64[0] = (__int64)v87;
      v65->m128i_i32[2] = v88;
    }
    else
    {
      v29 = v78.m128i_i64[0];
      v78.m128i_i32[0] = v27;
      v30 = sub_33FA050((__int64)v28, v27, (__int64)&v91, v76, v74, v26, *(unsigned __int8 **)v29, *(_QWORD *)(v29 + 8));
      v31 = v73;
      v86 = v32;
      v33 = v72;
      v85 = v30;
      v34 = v78.m128i_u32[0];
      *(_QWORD *)v29 = v30;
      *(_DWORD *)(v29 + 8) = v86;
      v35 = v77;
      v83 = sub_33FA050(
              a1[1],
              v34,
              (__int64)&v91,
              v31,
              v33,
              v26,
              (unsigned __int8 *)v77->m128i_i64[0],
              v77->m128i_i64[1]);
      v84 = v36;
      v77->m128i_i64[0] = (__int64)v83;
      v35->m128i_i32[2] = v84;
    }
    v37 = v91;
    if ( v91 )
      goto LABEL_13;
  }
}
