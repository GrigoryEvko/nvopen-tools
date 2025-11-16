// Function: sub_3787FB0
// Address: 0x3787fb0
//
unsigned __int8 *__fastcall sub_3787FB0(__int64 a1, __int64 a2, int a3)
{
  __int64 v4; // rsi
  int v5; // eax
  __int64 v6; // rsi
  __int64 v7; // rax
  __m128i v8; // xmm0
  unsigned __int16 *v9; // rax
  _QWORD *v10; // rsi
  __int64 v11; // rax
  __int16 v12; // dx
  __int64 v13; // rax
  __m128i v14; // xmm5
  __int64 v15; // rsi
  __int64 v16; // rax
  __int16 v17; // dx
  __int64 v18; // rax
  __int16 v19; // dx
  __int64 v20; // rax
  __int64 v21; // rax
  __m128i v22; // xmm1
  __m128i v23; // xmm2
  __m128i v24; // xmm3
  unsigned __int16 *v25; // rax
  _QWORD *v26; // rsi
  __int64 v27; // rax
  __int16 v28; // dx
  __int64 v29; // rax
  __m128i v30; // xmm7
  unsigned __int16 *v31; // rax
  __int64 v32; // r14
  __int64 v33; // r15
  __int64 v34; // rcx
  __int128 v35; // rax
  unsigned __int8 *v36; // r14
  __int64 v38; // rdx
  unsigned __int16 *v39; // rax
  __int128 v40; // rax
  __int128 v41; // rax
  unsigned __int8 *v42; // rax
  __int64 v43; // r8
  __int64 v44; // r14
  __int64 v45; // rdx
  __int64 v46; // r15
  char v47; // cl
  __m128i v48; // rax
  unsigned __int64 v49; // rax
  bool v50; // al
  __int64 v51; // r9
  _QWORD *v52; // r10
  const __m128i *v53; // rax
  __int64 v54; // rcx
  __int128 v55; // rax
  __int64 v56; // r9
  __int128 v57; // [rsp-50h] [rbp-1C0h]
  __int128 v58; // [rsp-20h] [rbp-190h]
  _QWORD *v59; // [rsp+0h] [rbp-170h]
  __int64 v60; // [rsp+0h] [rbp-170h]
  unsigned __int8 v61; // [rsp+0h] [rbp-170h]
  unsigned int v62; // [rsp+8h] [rbp-168h]
  __int64 v63; // [rsp+8h] [rbp-168h]
  char v64; // [rsp+8h] [rbp-168h]
  _QWORD *v65; // [rsp+8h] [rbp-168h]
  __int128 v66; // [rsp+10h] [rbp-160h]
  __int128 v68; // [rsp+30h] [rbp-140h]
  __int64 v69; // [rsp+40h] [rbp-130h]
  char v70; // [rsp+40h] [rbp-130h]
  int v71; // [rsp+40h] [rbp-130h]
  char v72; // [rsp+5Fh] [rbp-111h] BYREF
  __int64 v73; // [rsp+60h] [rbp-110h] BYREF
  int v74; // [rsp+68h] [rbp-108h]
  __m128i v75; // [rsp+70h] [rbp-100h] BYREF
  __int64 v76; // [rsp+80h] [rbp-F0h] BYREF
  __int64 v77; // [rsp+88h] [rbp-E8h]
  __int64 v78; // [rsp+90h] [rbp-E0h] BYREF
  __int64 v79; // [rsp+98h] [rbp-D8h]
  __m128i v80; // [rsp+A0h] [rbp-D0h] BYREF
  __m128i v81; // [rsp+B0h] [rbp-C0h] BYREF
  __int128 v82; // [rsp+C0h] [rbp-B0h] BYREF
  __int128 v83; // [rsp+D0h] [rbp-A0h] BYREF
  __m128i v84; // [rsp+E0h] [rbp-90h] BYREF
  __m128i v85; // [rsp+F0h] [rbp-80h] BYREF
  __int64 v86; // [rsp+100h] [rbp-70h] BYREF
  __int64 v87; // [rsp+108h] [rbp-68h]
  __int64 v88; // [rsp+110h] [rbp-60h]
  __m128i v89; // [rsp+120h] [rbp-50h] BYREF
  __m128i v90; // [rsp+130h] [rbp-40h] BYREF

  v4 = *(_QWORD *)(a2 + 80);
  v73 = v4;
  if ( v4 )
    sub_B96E90((__int64)&v73, v4, 1);
  v5 = *(_DWORD *)(a2 + 72);
  v6 = *(_QWORD *)a1;
  LODWORD(v77) = 0;
  v74 = v5;
  v7 = *(_QWORD *)(a2 + 40);
  LODWORD(v79) = 0;
  v76 = 0;
  v8 = _mm_loadu_si128((const __m128i *)(v7 + 40));
  v78 = 0;
  v75 = v8;
  v9 = (unsigned __int16 *)(*(_QWORD *)(v8.m128i_i64[0] + 48) + 16LL * v8.m128i_u32[2]);
  sub_2FE6CC0((__int64)&v89, v6, *(_QWORD *)(*(_QWORD *)(a1 + 8) + 64LL), *v9, *((_QWORD *)v9 + 1));
  if ( v89.m128i_i8[0] == 6 )
  {
    sub_375E8D0(a1, v75.m128i_u64[0], v75.m128i_i64[1], (__int64)&v76, (__int64)&v78);
  }
  else
  {
    v10 = *(_QWORD **)(a1 + 8);
    v85.m128i_i16[0] = 0;
    v84.m128i_i16[0] = 0;
    v84.m128i_i64[1] = 0;
    v85.m128i_i64[1] = 0;
    v11 = *(_QWORD *)(v75.m128i_i64[0] + 48) + 16LL * v75.m128i_u32[2];
    v12 = *(_WORD *)v11;
    v13 = *(_QWORD *)(v11 + 8);
    LOWORD(v86) = v12;
    v87 = v13;
    sub_33D0340((__int64)&v89, (__int64)v10, &v86);
    v14 = _mm_loadu_si128(&v90);
    v84 = _mm_loadu_si128(&v89);
    v85 = v14;
    sub_3408290(
      (__int64)&v89,
      v10,
      (__int128 *)v75.m128i_i8,
      (__int64)&v73,
      (unsigned int *)&v84,
      (unsigned int *)&v85,
      v8);
    v76 = v89.m128i_i64[0];
    LODWORD(v77) = v89.m128i_i32[2];
    v78 = v90.m128i_i64[0];
    LODWORD(v79) = v90.m128i_i32[2];
  }
  v15 = *(_QWORD *)(a1 + 8);
  v80.m128i_i16[0] = 0;
  v80.m128i_i64[1] = 0;
  v16 = *(_QWORD *)(v76 + 48) + 16LL * (unsigned int)v77;
  v72 = 0;
  v17 = *(_WORD *)v16;
  v18 = *(_QWORD *)(v16 + 8);
  LOWORD(v86) = v17;
  v19 = *(_WORD *)(a2 + 96);
  v87 = v18;
  v20 = *(_QWORD *)(a2 + 104);
  v85.m128i_i16[0] = v19;
  v85.m128i_i64[1] = v20;
  sub_33D04E0((__int64)&v89, v15, (unsigned __int16 *)&v85, (unsigned __int16 *)&v86, &v72);
  v21 = *(_QWORD *)(a2 + 40);
  v22 = _mm_loadu_si128(&v89);
  *(_QWORD *)&v82 = 0;
  v23 = _mm_loadu_si128(&v90);
  DWORD2(v82) = 0;
  v24 = _mm_loadu_si128((const __m128i *)(v21 + 200));
  v80 = v22;
  *(_QWORD *)&v83 = 0;
  v81 = v24;
  DWORD2(v83) = 0;
  if ( a3 == 1 && *(_DWORD *)(v24.m128i_i64[0] + 24) == 208 )
  {
    sub_377EF80((__int64 *)a1, v24.m128i_i64[0], (__int64)&v82, (__int64)&v83, v8);
  }
  else
  {
    v25 = (unsigned __int16 *)(*(_QWORD *)(v24.m128i_i64[0] + 48) + 16LL * v81.m128i_u32[2]);
    sub_2FE6CC0((__int64)&v89, *(_QWORD *)a1, *(_QWORD *)(*(_QWORD *)(a1 + 8) + 64LL), *v25, *((_QWORD *)v25 + 1));
    if ( v89.m128i_i8[0] == 6 )
    {
      sub_375E8D0(a1, v81.m128i_u64[0], v81.m128i_i64[1], (__int64)&v82, (__int64)&v83);
    }
    else
    {
      v85.m128i_i64[1] = 0;
      v26 = *(_QWORD **)(a1 + 8);
      v84.m128i_i16[0] = 0;
      v85.m128i_i16[0] = 0;
      v84.m128i_i64[1] = 0;
      v27 = *(_QWORD *)(v81.m128i_i64[0] + 48) + 16LL * v81.m128i_u32[2];
      v28 = *(_WORD *)v27;
      v29 = *(_QWORD *)(v27 + 8);
      LOWORD(v86) = v28;
      v87 = v29;
      sub_33D0340((__int64)&v89, (__int64)v26, &v86);
      v30 = _mm_loadu_si128(&v90);
      v84 = _mm_loadu_si128(&v89);
      v85 = v30;
      sub_3408290(
        (__int64)&v89,
        v26,
        (__int128 *)v81.m128i_i8,
        (__int64)&v73,
        (unsigned int *)&v84,
        (unsigned int *)&v85,
        v8);
      *(_QWORD *)&v82 = v89.m128i_i64[0];
      DWORD2(v82) = v89.m128i_i32[2];
      *(_QWORD *)&v83 = v90.m128i_i64[0];
      DWORD2(v83) = v90.m128i_i32[2];
    }
  }
  v31 = (unsigned __int16 *)(*(_QWORD *)(v75.m128i_i64[0] + 48) + 16LL * v75.m128i_u32[2]);
  sub_3408380(
    &v89,
    *(_QWORD **)(a1 + 8),
    *(_QWORD *)(*(_QWORD *)(a2 + 40) + 240LL),
    *(_QWORD *)(*(_QWORD *)(a2 + 40) + 248LL),
    *v31,
    *((_QWORD *)v31 + 1),
    v8,
    (__int64)&v73);
  v32 = v89.m128i_i64[0];
  v33 = v89.m128i_u32[2];
  *(_QWORD *)&v68 = v90.m128i_i64[0];
  v34 = *(_QWORD *)(a2 + 40);
  *((_QWORD *)&v68 + 1) = v90.m128i_u32[2];
  *((_QWORD *)&v57 + 1) = v89.m128i_u32[2];
  *(_QWORD *)&v57 = v89.m128i_i64[0];
  *(_QWORD *)&v35 = sub_33F5F90(
                      *(__int64 **)(a1 + 8),
                      *(_QWORD *)v34,
                      *(_QWORD *)(v34 + 8),
                      (__int64)&v73,
                      v76,
                      v77,
                      *(_QWORD *)(v34 + 80),
                      *(_QWORD *)(v34 + 88),
                      *(_OWORD *)(v34 + 120),
                      *(_OWORD *)(v34 + 160),
                      v82,
                      v57,
                      v80.m128i_i64[0],
                      v80.m128i_i64[1],
                      *(const __m128i **)(a2 + 112),
                      (*(_WORD *)(a2 + 32) >> 7) & 7,
                      (*(_BYTE *)(a2 + 33) & 4) != 0,
                      (*(_BYTE *)(a2 + 33) & 8) != 0);
  v66 = v35;
  if ( v72 )
  {
    v36 = (unsigned __int8 *)v35;
    goto LABEL_11;
  }
  v38 = *(_QWORD *)(a2 + 40);
  v59 = *(_QWORD **)(a1 + 8);
  v39 = (unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)(v38 + 80) + 48LL) + 16LL * *(unsigned int *)(v38 + 88));
  v62 = *v39;
  v69 = *((_QWORD *)v39 + 1);
  *(_QWORD *)&v40 = sub_33FB160(
                      (__int64)v59,
                      *(_QWORD *)(v38 + 160),
                      *(_QWORD *)(v38 + 168),
                      (__int64)&v73,
                      *v39,
                      v69,
                      v8);
  *((_QWORD *)&v58 + 1) = v33;
  *(_QWORD *)&v58 = v32;
  *(_QWORD *)&v41 = sub_3406EB0(v59, 0x3Au, (__int64)&v73, v62, v69, v69, v58, v40);
  v42 = sub_3406EB0(
          *(_QWORD **)(a1 + 8),
          0x38u,
          (__int64)&v73,
          v62,
          v69,
          v69,
          *(_OWORD *)(*(_QWORD *)(a2 + 40) + 80LL),
          v41);
  v43 = *(_QWORD *)(a2 + 112);
  v44 = (__int64)v42;
  v46 = v45;
  v47 = *(_BYTE *)(v43 + 34);
  if ( v80.m128i_i16[0] )
  {
    if ( (unsigned __int16)(v80.m128i_i16[0] - 176) <= 0x34u )
      goto LABEL_17;
  }
  else
  {
    v60 = *(_QWORD *)(a2 + 112);
    v64 = *(_BYTE *)(v43 + 34);
    v50 = sub_3007100((__int64)&v80);
    v47 = v64;
    v43 = v60;
    if ( v50 )
    {
LABEL_17:
      v63 = v43;
      v70 = v47;
      v48.m128i_i64[0] = sub_2D5B750((unsigned __int16 *)&v80);
      v43 = v63;
      v89 = v48;
      v47 = -1;
      v49 = -(((unsigned __int64)v48.m128i_i64[0] >> 3) | (1LL << v70))
          & (((unsigned __int64)v48.m128i_i64[0] >> 3) | (1LL << v70));
      if ( v49 )
      {
        _BitScanReverse64(&v49, v49);
        v47 = 63 - (v49 ^ 0x3F);
      }
    }
  }
  v61 = v47;
  v51 = *(_QWORD *)(v43 + 72);
  v52 = *(_QWORD **)(*(_QWORD *)(a1 + 8) + 40LL);
  v89 = _mm_loadu_si128((const __m128i *)(v43 + 40));
  v71 = v51;
  v90 = _mm_loadu_si128((const __m128i *)(v43 + 56));
  v65 = v52;
  LODWORD(v88) = sub_2EAC1E0(v43);
  v87 = 0;
  BYTE4(v88) = 0;
  v86 = 0;
  v53 = (const __m128i *)sub_2E7BD70(v65, 2u, -1, v61, (int)&v89, v71, 0, v88, 1u, 0, 0);
  v54 = *(_QWORD *)(a2 + 40);
  *(_QWORD *)&v55 = sub_33F5F90(
                      *(__int64 **)(a1 + 8),
                      *(_QWORD *)v54,
                      *(_QWORD *)(v54 + 8),
                      (__int64)&v73,
                      v78,
                      v79,
                      v44,
                      v46,
                      *(_OWORD *)(v54 + 120),
                      *(_OWORD *)(v54 + 160),
                      v83,
                      v68,
                      v23.m128i_i64[0],
                      v23.m128i_i64[1],
                      v53,
                      (*(_WORD *)(a2 + 32) >> 7) & 7,
                      (*(_BYTE *)(a2 + 33) & 4) != 0,
                      (*(_BYTE *)(a2 + 33) & 8) != 0);
  v36 = sub_3406EB0(*(_QWORD **)(a1 + 8), 2u, (__int64)&v73, 1, 0, v56, v66, v55);
LABEL_11:
  if ( v73 )
    sub_B91220((__int64)&v73, v73);
  return v36;
}
