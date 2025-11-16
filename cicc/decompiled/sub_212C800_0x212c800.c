// Function: sub_212C800
// Address: 0x212c800
//
unsigned __int64 __fastcall sub_212C800(
        _QWORD *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        double a5,
        double a6,
        __m128i a7)
{
  __int64 v10; // rsi
  __int16 v11; // ax
  unsigned int v12; // r12d
  unsigned __int64 *v13; // rax
  __int64 v14; // rdx
  __int64 v15; // r15
  __int64 v16; // rax
  unsigned __int8 v17; // di
  __int64 v18; // rax
  __int64 v19; // rax
  char v20; // al
  __int64 *v21; // rdi
  const void **v22; // rdx
  __m128 v23; // xmm0
  __m128i v24; // xmm1
  __int64 v25; // rsi
  __int64 *v26; // rax
  __m128i v27; // kr00_16
  __int64 v28; // r14
  __int64 v29; // rcx
  __int64 v30; // rdx
  __int64 *v31; // r13
  __int64 v32; // r15
  __int64 v33; // rax
  __int64 v34; // rdx
  __int64 *v35; // rax
  __int64 v36; // rcx
  unsigned __int64 v37; // r12
  __int16 *v38; // r13
  __int64 v39; // r14
  __int64 v40; // r15
  __int64 v41; // rdx
  __int64 v42; // r8
  __int64 v43; // r9
  __int64 v44; // rax
  __int64 v45; // rdx
  __int64 *v46; // rax
  __int64 *v47; // r12
  unsigned int v48; // edx
  unsigned __int64 v49; // rdx
  __int64 v50; // rax
  __int64 v51; // r14
  __int16 *v52; // rcx
  __int8 v53; // si
  unsigned __int64 v54; // rax
  __int64 v55; // r15
  unsigned int v56; // r10d
  __int128 v57; // kr10_16
  unsigned int v58; // esi
  __int64 *v59; // rax
  __int64 *v60; // rdi
  __int64 *v61; // r13
  __m128i v62; // xmm3
  unsigned int v63; // edx
  __int64 v64; // r12
  __m128i v65; // xmm2
  __int64 *v66; // rax
  __int64 *v67; // rbx
  __int64 *v68; // r10
  __int64 v69; // rdx
  __int64 v70; // r11
  unsigned __int64 v71; // rcx
  __int64 v72; // r8
  __int64 v73; // rax
  __int16 *v74; // r15
  __int64 v75; // r9
  __int8 v76; // dl
  unsigned __int64 v77; // rax
  unsigned int v78; // esi
  __int64 *v79; // rax
  __int64 v80; // rsi
  unsigned int v81; // edx
  unsigned __int64 result; // rax
  bool v83; // al
  bool v84; // al
  __int128 v85; // [rsp-30h] [rbp-170h]
  __int128 v86; // [rsp-20h] [rbp-160h]
  __int128 v87; // [rsp-20h] [rbp-160h]
  __int128 v88; // [rsp-10h] [rbp-150h]
  __int128 v89; // [rsp-10h] [rbp-150h]
  __int64 *v90; // [rsp+8h] [rbp-138h]
  unsigned __int64 v91; // [rsp+10h] [rbp-130h]
  unsigned __int64 v92; // [rsp+10h] [rbp-130h]
  __int16 *v93; // [rsp+18h] [rbp-128h]
  unsigned int v94; // [rsp+20h] [rbp-120h]
  unsigned int v95; // [rsp+20h] [rbp-120h]
  __m128i v96; // [rsp+20h] [rbp-120h]
  unsigned int v98; // [rsp+3Ch] [rbp-104h]
  __int64 (__fastcall *v99)(__int64, __int64, __int64, _QWORD, const void **); // [rsp+48h] [rbp-F8h]
  __int64 *v100; // [rsp+48h] [rbp-F8h]
  unsigned int v101; // [rsp+48h] [rbp-F8h]
  unsigned __int64 v102; // [rsp+48h] [rbp-F8h]
  __int64 v103; // [rsp+50h] [rbp-F0h]
  const void **v104; // [rsp+50h] [rbp-F0h]
  unsigned __int64 v105; // [rsp+50h] [rbp-F0h]
  unsigned __int8 v106; // [rsp+60h] [rbp-E0h]
  __int64 *v107; // [rsp+60h] [rbp-E0h]
  __int64 v108; // [rsp+68h] [rbp-D8h]
  const void **v109; // [rsp+70h] [rbp-D0h]
  __int64 v110; // [rsp+A0h] [rbp-A0h] BYREF
  int v111; // [rsp+A8h] [rbp-98h]
  __int128 v112; // [rsp+B0h] [rbp-90h] BYREF
  __m128i v113; // [rsp+C0h] [rbp-80h] BYREF
  __m128i v114; // [rsp+D0h] [rbp-70h] BYREF
  __m128i v115; // [rsp+E0h] [rbp-60h] BYREF
  __m128 v116; // [rsp+F0h] [rbp-50h] BYREF
  __m128i v117; // [rsp+100h] [rbp-40h]

  v10 = *(_QWORD *)(a2 + 72);
  v110 = v10;
  if ( v10 )
    sub_1623A60((__int64)&v110, v10, 2);
  v111 = *(_DWORD *)(a2 + 64);
  v11 = *(_WORD *)(a2 + 24);
  if ( v11 == 116 )
  {
    v98 = 116;
    v12 = 12;
  }
  else if ( v11 > 116 )
  {
    v98 = 117;
    v12 = 10;
  }
  else
  {
    v98 = (v11 != 114) + 116;
    v12 = 2 * (v11 == 114) + 18;
  }
  v13 = *(unsigned __int64 **)(a2 + 32);
  v113.m128i_i32[2] = 0;
  DWORD2(v112) = 0;
  v114.m128i_i32[2] = 0;
  v115.m128i_i32[2] = 0;
  v14 = v13[1];
  *(_QWORD *)&v112 = 0;
  v113.m128i_i64[0] = 0;
  v114.m128i_i64[0] = 0;
  v115.m128i_i64[0] = 0;
  sub_20174B0((__int64)a1, *v13, v14, &v112, &v113);
  sub_20174B0(
    (__int64)a1,
    *(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL),
    *(_QWORD *)(*(_QWORD *)(a2 + 32) + 48LL),
    &v114,
    &v115);
  v15 = *a1;
  v16 = *(_QWORD *)(v112 + 40) + 16LL * DWORD2(v112);
  v17 = *(_BYTE *)v16;
  v109 = *(const void ***)(v16 + 8);
  v18 = a1[1];
  v106 = v17;
  v99 = *(__int64 (__fastcall **)(__int64, __int64, __int64, _QWORD, const void **))(*(_QWORD *)*a1 + 264LL);
  v103 = *(_QWORD *)(v18 + 48);
  v19 = sub_1E0A0C0(*(_QWORD *)(v18 + 32));
  v20 = v99(v15, v19, v103, v17, v109);
  v21 = (__int64 *)a1[1];
  LOBYTE(v15) = v20;
  v104 = v22;
  v23 = (__m128)_mm_loadu_si128(&v113);
  v24 = _mm_loadu_si128(&v115);
  v25 = *(unsigned __int16 *)(a2 + 24);
  *((_QWORD *)&v88 + 1) = 2;
  *(_QWORD *)&v88 = &v116;
  v116 = v23;
  v117 = v24;
  v26 = sub_1D359D0(v21, v25, (__int64)&v110, v106, v109, 0, *(double *)v23.m128_u64, *(double *)v24.m128i_i64, a7, v88);
  v27 = v113;
  v28 = v115.m128i_i64[0];
  *(_QWORD *)a4 = v26;
  LOBYTE(v99) = v15;
  v29 = (unsigned __int8)v15;
  *(_DWORD *)(a4 + 8) = v30;
  v31 = (__int64 *)a1[1];
  v32 = v115.m128i_i64[1];
  v91 = v29;
  v33 = sub_1D28D50(v31, v12, v30, v29, v27.m128i_i64[0], v27.m128i_i64[1]);
  *((_QWORD *)&v85 + 1) = v32;
  *(_QWORD *)&v85 = v28;
  v35 = sub_1D3A900(
          v31,
          0x89u,
          (__int64)&v110,
          v91,
          v104,
          0,
          v23,
          *(double *)v24.m128i_i64,
          a7,
          v27.m128i_u64[0],
          (__int16 *)v27.m128i_i64[1],
          v85,
          v33,
          v34);
  v36 = (unsigned __int8)v99;
  v37 = v113.m128i_i64[0];
  v90 = v35;
  v38 = (__int16 *)v113.m128i_i64[1];
  v39 = v115.m128i_i64[0];
  v100 = (__int64 *)a1[1];
  v40 = v115.m128i_i64[1];
  v92 = v36;
  v94 = v41;
  v44 = sub_1D28D50(v100, 0x11u, v41, v36, v42, v43);
  *((_QWORD *)&v86 + 1) = v40;
  *(_QWORD *)&v86 = v39;
  v46 = sub_1D3A900(
          v100,
          0x89u,
          (__int64)&v110,
          v92,
          v104,
          0,
          v23,
          *(double *)v24.m128i_i64,
          a7,
          v37,
          v38,
          v86,
          v44,
          v45);
  v47 = (__int64 *)a1[1];
  v105 = (unsigned __int64)v46;
  v101 = v48;
  v49 = (unsigned __int64)v90;
  v50 = v90[5] + 16LL * v94;
  v51 = v114.m128i_i64[0];
  v52 = (__int16 *)v94;
  v53 = *(_BYTE *)v50;
  v54 = *(_QWORD *)(v50 + 8);
  v55 = v114.m128i_i64[1];
  v116.m128_i8[0] = v53;
  v56 = v106;
  v116.m128_u64[1] = v54;
  v57 = v112;
  if ( v53 )
  {
    v58 = ((unsigned __int8)(v53 - 14) < 0x60u) + 134;
  }
  else
  {
    v93 = (__int16 *)v94;
    v96 = (__m128i)v112;
    v84 = sub_1F58D20((__int64)&v116);
    v56 = v106;
    v49 = (unsigned __int64)v90;
    v52 = v93;
    v57 = (__int128)v96;
    v58 = 134 - (!v84 - 1);
  }
  v95 = v56;
  v59 = sub_1D3A900(v47, v58, (__int64)&v110, v106, v109, 0, v23, *(double *)v24.m128i_i64, a7, v49, v52, v57, v51, v55);
  v60 = (__int64 *)a1[1];
  v61 = v59;
  v62 = _mm_loadu_si128(&v114);
  v64 = v63;
  v65 = _mm_loadu_si128((const __m128i *)&v112);
  *((_QWORD *)&v89 + 1) = 2;
  *(_QWORD *)&v89 = &v116;
  v116 = (__m128)v65;
  v117 = v62;
  v66 = sub_1D359D0(v60, v98, (__int64)&v110, v95, v109, 0, *(double *)v23.m128_u64, *(double *)v24.m128i_i64, v65, v89);
  v67 = (__int64 *)a1[1];
  v68 = v66;
  v70 = v69;
  v71 = v106;
  v72 = (__int64)v61;
  v73 = *(_QWORD *)(v105 + 40) + 16LL * v101;
  v74 = (__int16 *)v101;
  v75 = v64;
  v76 = *(_BYTE *)v73;
  v77 = *(_QWORD *)(v73 + 8);
  v116.m128_i8[0] = v76;
  v116.m128_u64[1] = v77;
  if ( v76 )
  {
    v78 = ((unsigned __int8)(v76 - 14) < 0x60u) + 134;
  }
  else
  {
    v102 = v106;
    v107 = v68;
    v108 = v70;
    v83 = sub_1F58D20((__int64)&v116);
    v71 = v102;
    v72 = (__int64)v61;
    v75 = v64;
    v68 = v107;
    v70 = v108;
    v78 = 134 - (!v83 - 1);
  }
  *((_QWORD *)&v87 + 1) = v70;
  *(_QWORD *)&v87 = v68;
  v79 = sub_1D3A900(
          v67,
          v78,
          (__int64)&v110,
          v71,
          v109,
          0,
          v23,
          *(double *)v24.m128i_i64,
          v65,
          v105,
          v74,
          v87,
          v72,
          v75);
  v80 = v110;
  *(_QWORD *)a3 = v79;
  result = v81;
  *(_DWORD *)(a3 + 8) = v81;
  if ( v80 )
    return sub_161E7C0((__int64)&v110, v80);
  return result;
}
