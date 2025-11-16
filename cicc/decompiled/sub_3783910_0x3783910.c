// Function: sub_3783910
// Address: 0x3783910
//
void __fastcall sub_3783910(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int16 *v6; // rax
  __int16 v7; // dx
  __int64 v8; // rax
  __int64 v9; // rsi
  __int64 v10; // rcx
  __m128i v11; // xmm0
  __int64 v12; // rdi
  __int64 v13; // rsi
  unsigned int v14; // r12d
  __int64 v15; // rdx
  __int64 v16; // r8
  unsigned __int16 v17; // ax
  __int8 v18; // cl
  int v19; // esi
  __int64 v20; // rdx
  __int64 v21; // r15
  __int8 v22; // dl
  unsigned int v23; // r13d
  __int64 *v24; // r14
  unsigned __int16 v25; // ax
  __int64 v26; // r8
  __int64 v27; // r9
  __int64 v28; // r13
  __int64 v29; // rsi
  char v30; // al
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 v33; // rdx
  __int128 v34; // rax
  unsigned int v35; // edi
  unsigned __int16 v36; // r15
  __int64 v37; // r14
  _QWORD *v38; // rdi
  _QWORD *v39; // rdi
  unsigned __int64 v40; // rax
  unsigned __int16 v41; // r12
  __int64 v42; // rdx
  __int64 v43; // rdx
  __int128 v44; // rax
  __int128 v45; // rax
  __int64 v46; // r9
  __int128 v47; // rax
  unsigned __int64 v48; // r10
  __int128 v49; // rax
  __int64 v50; // r9
  __int128 v51; // rax
  __int64 v52; // r9
  unsigned __int8 *v53; // rax
  __int64 v54; // rdx
  __int64 v55; // rdx
  __int64 v56; // r9
  __int128 v57; // rax
  __int64 *v58; // r13
  __int128 v59; // rax
  __m128i *v60; // rax
  __int64 v61; // rdx
  __int128 v62; // rax
  _QWORD *v63; // r13
  __m128i v64; // xmm4
  __int64 v65; // rsi
  __int64 v66; // rdx
  __int64 v67; // rcx
  __int64 v68; // r8
  _QWORD *v69; // [rsp+0h] [rbp-1D0h]
  __int64 v70; // [rsp+0h] [rbp-1D0h]
  __int128 v72; // [rsp+10h] [rbp-1C0h]
  __int128 v73; // [rsp+30h] [rbp-1A0h]
  __int128 v74; // [rsp+30h] [rbp-1A0h]
  __int128 v75; // [rsp+30h] [rbp-1A0h]
  unsigned int v76; // [rsp+40h] [rbp-190h]
  __int128 v77; // [rsp+40h] [rbp-190h]
  __int64 v78; // [rsp+50h] [rbp-180h]
  __int128 v79; // [rsp+50h] [rbp-180h]
  const __m128i *v81; // [rsp+68h] [rbp-168h]
  unsigned __int64 v82; // [rsp+70h] [rbp-160h]
  _QWORD *v83; // [rsp+70h] [rbp-160h]
  __int64 v84; // [rsp+70h] [rbp-160h]
  __int64 v85; // [rsp+78h] [rbp-158h]
  __int128 v86; // [rsp+80h] [rbp-150h]
  __m128i v87; // [rsp+90h] [rbp-140h]
  const __m128i *v88; // [rsp+A0h] [rbp-130h]
  unsigned __int64 v89; // [rsp+B8h] [rbp-118h]
  __int128 v90; // [rsp+C0h] [rbp-110h] BYREF
  __int64 v91; // [rsp+D0h] [rbp-100h] BYREF
  int v92; // [rsp+D8h] [rbp-F8h]
  __int64 v93; // [rsp+E0h] [rbp-F0h] BYREF
  __int64 v94; // [rsp+E8h] [rbp-E8h]
  __int128 v95; // [rsp+F0h] [rbp-E0h] BYREF
  unsigned __int64 v96; // [rsp+100h] [rbp-D0h]
  __int64 v97; // [rsp+108h] [rbp-C8h]
  __m128i v98; // [rsp+110h] [rbp-C0h] BYREF
  __m128i v99; // [rsp+120h] [rbp-B0h] BYREF
  __int64 v100[2]; // [rsp+130h] [rbp-A0h] BYREF
  __int64 v101; // [rsp+140h] [rbp-90h]
  __int64 v102; // [rsp+148h] [rbp-88h]
  __int64 v103; // [rsp+150h] [rbp-80h]
  __int64 v104; // [rsp+158h] [rbp-78h]
  __int128 v105; // [rsp+160h] [rbp-70h] BYREF
  __int64 v106; // [rsp+170h] [rbp-60h]
  __m128i v107; // [rsp+180h] [rbp-50h] BYREF
  __m128i v108; // [rsp+190h] [rbp-40h] BYREF

  v6 = *(__int16 **)(a2 + 48);
  v7 = *v6;
  *((_QWORD *)&v90 + 1) = *((_QWORD *)v6 + 1);
  v8 = *(_QWORD *)(a2 + 40);
  LOWORD(v90) = v7;
  v9 = *(_QWORD *)(a2 + 80);
  v10 = *(_QWORD *)(v8 + 40);
  v11 = _mm_loadu_si128((const __m128i *)v8);
  v91 = v9;
  v78 = v10;
  v76 = *(_DWORD *)(v8 + 48);
  v72 = (__int128)_mm_loadu_si128((const __m128i *)(v8 + 40));
  v87 = _mm_loadu_si128((const __m128i *)(v8 + 80));
  if ( v9 )
    sub_B96E90((__int64)&v91, v9, 1);
  v12 = *(_QWORD *)(a1 + 8);
  v13 = (unsigned int)v90;
  v92 = *(_DWORD *)(a2 + 72);
  v14 = sub_33CD850(v12, (unsigned int)v90, *((unsigned __int64 *)&v90 + 1), 0);
  if ( (_WORD)v90 )
  {
    v22 = (unsigned __int16)(v90 - 176) <= 0x34u;
    v21 = 0;
    v31 = (unsigned __int16)v90 - 1;
    v19 = word_4456340[v31];
    v17 = word_4456580[v31];
    v18 = v22;
  }
  else
  {
    v89 = sub_3007240((__int64)&v90);
    v17 = sub_3009970((__int64)&v90, v13, v15, HIDWORD(v89), v16);
    v18 = BYTE4(v89);
    v19 = v89;
    v21 = v20;
    v22 = BYTE4(v89);
  }
  v23 = v17;
  v24 = *(__int64 **)(*(_QWORD *)(a1 + 8) + 64LL);
  v107.m128i_i32[0] = v19;
  v107.m128i_i8[4] = v18;
  if ( v22 )
    v25 = sub_2D43AD0(v17, v19);
  else
    v25 = sub_2D43050(v17, v19);
  if ( v25 )
  {
    LOWORD(v93) = v25;
    v28 = *(_QWORD *)(a1 + 8);
    v94 = 0;
  }
  else
  {
    v25 = sub_3009450(v24, v23, v21, v107.m128i_i64[0], v26, v27);
    v28 = *(_QWORD *)(a1 + 8);
    LOWORD(v93) = v25;
    v94 = v32;
    if ( !v25 )
    {
      v101 = sub_3007260((__int64)&v93);
      v29 = v101;
      v102 = v33;
      v30 = v33;
      goto LABEL_15;
    }
  }
  if ( v25 == 1 || (unsigned __int16)(v25 - 504) <= 7u )
LABEL_32:
    BUG();
  v29 = *(_QWORD *)&byte_444C4A0[16 * v25 - 16];
  v30 = byte_444C4A0[16 * v25 - 8];
LABEL_15:
  LOBYTE(v97) = v30;
  v96 = (unsigned __int64)(v29 + 7) >> 3;
  *(_QWORD *)&v34 = sub_33EDE90(v28, v96, v97, v14);
  v35 = DWORD2(v34);
  v86 = v34;
  *((_QWORD *)&v34 + 1) = v34;
  *(_QWORD *)&v34 = *(_QWORD *)(v34 + 48) + 16LL * v35;
  v36 = *(_WORD *)v34;
  v37 = *(_QWORD *)(v34 + 8);
  sub_2EAC300((__int64)&v105, *(_QWORD *)(*(_QWORD *)(a1 + 8) + 40LL), *(_DWORD *)(*((_QWORD *)&v34 + 1) + 96LL), 0);
  v38 = *(_QWORD **)(*(_QWORD *)(a1 + 8) + 40LL);
  v107 = 0u;
  v108 = 0u;
  v88 = (const __m128i *)sub_2E7BD70(v38, 2u, -1, v14, (int)&v107, 0, v105, v106, 1u, 0, 0);
  v39 = *(_QWORD **)(*(_QWORD *)(a1 + 8) + 40LL);
  v107 = 0u;
  v108 = 0u;
  v40 = sub_2E7BD70(v39, 1u, -1, v14, (int)&v107, 0, v105, v106, 1u, 0, 0);
  v41 = v90;
  v81 = (const __m128i *)v40;
  if ( (_WORD)v90 )
  {
    if ( (unsigned __int16)(v90 - 17) <= 0xD3u )
    {
      v107.m128i_i64[1] = 0;
      v41 = word_4456580[(unsigned __int16)v90 - 1];
      v107.m128i_i16[0] = v41;
      if ( !v41 )
        goto LABEL_19;
      goto LABEL_26;
    }
    goto LABEL_17;
  }
  if ( !sub_30070B0((__int64)&v90) )
  {
LABEL_17:
    v42 = *((_QWORD *)&v90 + 1);
    goto LABEL_18;
  }
  v41 = sub_3009970((__int64)&v90, 1, v66, v67, v68);
LABEL_18:
  v107.m128i_i16[0] = v41;
  v107.m128i_i64[1] = v42;
  if ( !v41 )
  {
LABEL_19:
    v103 = sub_3007260((__int64)&v107);
    v104 = v43;
    v82 = v103;
    goto LABEL_20;
  }
LABEL_26:
  if ( v41 == 1 || (unsigned __int16)(v41 - 504) <= 7u )
    goto LABEL_32;
  v82 = *(_QWORD *)&byte_444C4A0[16 * v41 - 16];
LABEL_20:
  v69 = *(_QWORD **)(a1 + 8);
  *(_QWORD *)&v44 = sub_3400BD0((__int64)v69, 1, (__int64)&v91, v36, v37, 0, v11, 0);
  v73 = v44;
  *(_QWORD *)&v45 = sub_33FB310(*(_QWORD *)(a1 + 8), v87.m128i_i64[0], v87.m128i_u32[2], (__int64)&v91, v36, v37, v11);
  *(_QWORD *)&v47 = sub_3406EB0(v69, 0x39u, (__int64)&v91, v36, v37, v46, v45, v73);
  v48 = v82 >> 3;
  v83 = *(_QWORD **)(a1 + 8);
  v74 = v47;
  v70 = (unsigned int)v48;
  *(_QWORD *)&v49 = sub_3400BD0((__int64)v83, (unsigned int)v48, (__int64)&v91, v36, v37, 0, v11, 0);
  *(_QWORD *)&v51 = sub_3406EB0(v83, 0x3Au, (__int64)&v91, v36, v37, v50, v74, v49);
  v53 = sub_3406EB0(*(_QWORD **)(a1 + 8), 0x38u, (__int64)&v91, v36, v37, v52, v86, v51);
  v85 = v54;
  v84 = (__int64)v53;
  *(_QWORD *)&v75 = sub_3400BD0(*(_QWORD *)(a1 + 8), -v70, (__int64)&v91, v36, v37, 0, v11, 0);
  *((_QWORD *)&v75 + 1) = v55;
  *(_QWORD *)&v57 = sub_3401740(
                      *(_QWORD *)(a1 + 8),
                      1,
                      (__int64)&v91,
                      *(unsigned __int16 *)(*(_QWORD *)(v78 + 48) + 16LL * v76),
                      *(_QWORD *)(*(_QWORD *)(v78 + 48) + 16LL * v76 + 8),
                      v56,
                      v90);
  v58 = *(__int64 **)(a1 + 8);
  v79 = v57;
  v107.m128i_i64[0] = 0;
  v107.m128i_i32[2] = 0;
  *(_QWORD *)&v59 = sub_33F17F0(v58, 51, (__int64)&v107, v36, v37);
  if ( v107.m128i_i64[0] )
  {
    v77 = v59;
    sub_B91220((__int64)&v107, v107.m128i_i64[0]);
    v59 = v77;
  }
  v60 = sub_33F5F90(
          v58,
          *(_QWORD *)(a1 + 8) + 288LL,
          0,
          (__int64)&v91,
          v11.m128i_i64[0],
          v11.m128i_i64[1],
          v84,
          v85,
          v59,
          v75,
          v79,
          *(_OWORD *)&v87,
          v93,
          v94,
          v88,
          0,
          0,
          0);
  *(_QWORD *)&v62 = sub_33F1C00(
                      *(__int64 **)(a1 + 8),
                      (unsigned int)v90,
                      *((__int64 *)&v90 + 1),
                      (__int64)&v91,
                      (__int64)v60,
                      v61,
                      v86,
                      *((__int64 *)&v86 + 1),
                      v72,
                      *(_OWORD *)&v87,
                      v81,
                      0);
  v63 = *(_QWORD **)(a1 + 8);
  v98.m128i_i64[1] = 0;
  v95 = v62;
  v98.m128i_i16[0] = 0;
  v99.m128i_i64[1] = 0;
  *(_QWORD *)&v62 = *(_QWORD *)(v62 + 48) + 16LL * DWORD2(v62);
  v99.m128i_i16[0] = 0;
  WORD4(v62) = *(_WORD *)v62;
  *(_QWORD *)&v62 = *(_QWORD *)(v62 + 8);
  LOWORD(v100[0]) = WORD4(v62);
  v100[1] = v62;
  sub_33D0340((__int64)&v107, (__int64)v63, v100);
  v64 = _mm_loadu_si128(&v108);
  v98 = _mm_loadu_si128(&v107);
  v99 = v64;
  sub_3408290((__int64)&v107, v63, &v95, (__int64)&v91, (unsigned int *)&v98, (unsigned int *)&v99, v11);
  v65 = v91;
  *(_QWORD *)a3 = v107.m128i_i64[0];
  *(_DWORD *)(a3 + 8) = v107.m128i_i32[2];
  *(_QWORD *)a4 = v108.m128i_i64[0];
  *(_DWORD *)(a4 + 8) = v108.m128i_i32[2];
  if ( v65 )
    sub_B91220((__int64)&v91, v65);
}
