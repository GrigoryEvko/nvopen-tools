// Function: sub_1FA1A20
// Address: 0x1fa1a20
//
__int64 __fastcall sub_1FA1A20(__int64 *a1, __int64 a2, __m128i a3, __m128i a4, __m128i a5)
{
  __int64 *v5; // rax
  __int64 v6; // r15
  __int64 v7; // r10
  unsigned int v8; // r14d
  __int64 v9; // rax
  bool v10; // zf
  __int8 v11; // cl
  __int64 v12; // rax
  __int64 result; // rax
  __int64 v15; // rbx
  __int64 v16; // rax
  int v17; // eax
  __int64 v18; // rdi
  __int64 (*v19)(); // rax
  int v20; // r9d
  __m128i v21; // xmm3
  __int64 v22; // rsi
  __int64 v23; // r13
  __int64 v24; // r8
  __int64 v25; // r9
  int v26; // eax
  __int64 v27; // rax
  __m128i v28; // xmm4
  unsigned int v29; // r14d
  __int64 v30; // rbx
  unsigned __int64 v31; // r13
  unsigned int v32; // r15d
  int v33; // eax
  __int64 v34; // rsi
  __int128 *v35; // r8
  unsigned __int64 v36; // r9
  unsigned __int16 v37; // cx
  _QWORD *v38; // r11
  unsigned __int64 v39; // r13
  __int64 v40; // rax
  __int64 v41; // r10
  __int64 v42; // rax
  __int128 v43; // rax
  __int64 v44; // r10
  int v45; // r8d
  int v46; // r9d
  unsigned int v47; // edx
  __int64 v48; // rax
  __int64 *v49; // rax
  __int64 v50; // rax
  __int64 *v51; // rax
  unsigned int v52; // eax
  __int64 v53; // rax
  unsigned __int64 v54; // rsi
  __int64 v55; // rdi
  char v56; // r8
  __int64 v57; // rax
  unsigned int v58; // eax
  __int64 v59; // rsi
  __int64 v60; // rdx
  __m128i v61; // rax
  unsigned __int8 *v62; // rax
  __int64 v63; // rdx
  __int64 v64; // rax
  __int128 v65; // [rsp-10h] [rbp-280h]
  __int128 v66; // [rsp+0h] [rbp-270h]
  __int64 v67; // [rsp+8h] [rbp-268h]
  int v68; // [rsp+10h] [rbp-260h]
  int v69; // [rsp+10h] [rbp-260h]
  __int64 v70; // [rsp+18h] [rbp-258h]
  __int128 *v71; // [rsp+18h] [rbp-258h]
  int v72; // [rsp+20h] [rbp-250h]
  unsigned int v73; // [rsp+20h] [rbp-250h]
  __int64 v74; // [rsp+28h] [rbp-248h]
  __int64 v75; // [rsp+28h] [rbp-248h]
  __int64 v76; // [rsp+30h] [rbp-240h]
  unsigned int v77; // [rsp+58h] [rbp-218h]
  unsigned int v78; // [rsp+5Ch] [rbp-214h]
  unsigned int v79; // [rsp+68h] [rbp-208h]
  unsigned int v80; // [rsp+6Ch] [rbp-204h]
  unsigned int v81; // [rsp+70h] [rbp-200h]
  unsigned __int16 v82; // [rsp+70h] [rbp-200h]
  __int64 v83; // [rsp+70h] [rbp-200h]
  unsigned int v84; // [rsp+88h] [rbp-1E8h]
  unsigned int v85; // [rsp+90h] [rbp-1E0h]
  __int64 *v86; // [rsp+90h] [rbp-1E0h]
  __int64 *v87; // [rsp+90h] [rbp-1E0h]
  unsigned int v88; // [rsp+98h] [rbp-1D8h]
  const void **v89; // [rsp+98h] [rbp-1D8h]
  __int64 v90; // [rsp+A0h] [rbp-1D0h]
  __int64 v91; // [rsp+A0h] [rbp-1D0h]
  __int64 v92; // [rsp+A0h] [rbp-1D0h]
  __int64 v93; // [rsp+A8h] [rbp-1C8h]
  __int8 v94; // [rsp+B0h] [rbp-1C0h]
  unsigned int v95; // [rsp+B0h] [rbp-1C0h]
  _QWORD *v96; // [rsp+B0h] [rbp-1C0h]
  __int64 v97; // [rsp+B0h] [rbp-1C0h]
  __int64 v98; // [rsp+B0h] [rbp-1C0h]
  unsigned __int64 v99; // [rsp+B8h] [rbp-1B8h]
  __int64 v100; // [rsp+C0h] [rbp-1B0h]
  __int64 *v101; // [rsp+C0h] [rbp-1B0h]
  __int64 *v102; // [rsp+C0h] [rbp-1B0h]
  __m128i v103; // [rsp+D0h] [rbp-1A0h] BYREF
  __m128i v104; // [rsp+E0h] [rbp-190h] BYREF
  __m128 v105; // [rsp+F0h] [rbp-180h] BYREF
  __m128i v106; // [rsp+100h] [rbp-170h] BYREF
  __int64 v107; // [rsp+110h] [rbp-160h] BYREF
  int v108; // [rsp+118h] [rbp-158h]
  __int64 v109; // [rsp+120h] [rbp-150h] BYREF
  int v110; // [rsp+128h] [rbp-148h]
  __m128i v111; // [rsp+130h] [rbp-140h] BYREF
  __int64 v112; // [rsp+140h] [rbp-130h]
  __int128 v113; // [rsp+150h] [rbp-120h]
  __int64 v114; // [rsp+160h] [rbp-110h]
  unsigned __int64 v115[2]; // [rsp+170h] [rbp-100h] BYREF
  _BYTE v116[32]; // [rsp+180h] [rbp-F0h] BYREF
  _BYTE *v117; // [rsp+1A0h] [rbp-D0h] BYREF
  __int64 v118; // [rsp+1A8h] [rbp-C8h]
  _BYTE v119[64]; // [rsp+1B0h] [rbp-C0h] BYREF
  __m128i v120; // [rsp+1F0h] [rbp-80h] BYREF
  _BYTE v121[112]; // [rsp+200h] [rbp-70h] BYREF

  v5 = *(__int64 **)(a2 + 32);
  v6 = *v5;
  v7 = v5[1];
  v8 = *((_DWORD *)v5 + 2);
  v9 = *(_QWORD *)(a2 + 40);
  v10 = *(_WORD *)(v6 + 24) == 185;
  v11 = *(_BYTE *)v9;
  v12 = *(_QWORD *)(v9 + 8);
  v104.m128i_i8[0] = v11;
  v104.m128i_i64[1] = v12;
  if ( !v10 )
    return 0;
  v100 = v7;
  v103.m128i_i8[0] = v11;
  if ( (*(_BYTE *)(v6 + 27) & 0xC) != 0 )
    return 0;
  if ( (*(_WORD *)(v6 + 26) & 0x380) != 0 )
    return 0;
  v15 = a2;
  v90 = 16LL * v8;
  v16 = *(_QWORD *)(v6 + 40) + v90;
  v94 = *(_BYTE *)v16;
  v99 = *(_QWORD *)(v16 + 8);
  if ( !sub_1D18C00(v6, 1, v8) || (*(_BYTE *)(v6 + 26) & 8) != 0 )
    return 0;
  if ( !v103.m128i_i8[0] )
  {
    if ( sub_1F58D20((__int64)&v104) )
    {
      v103.m128i_i64[0] = v100;
      v17 = sub_1F58D30((__int64)&v104);
      goto LABEL_11;
    }
    return 0;
  }
  if ( (unsigned __int8)(v103.m128i_i8[0] - 14) > 0x5Fu )
    return 0;
  v17 = word_42FA680[(unsigned __int8)(v103.m128i_i8[0] - 14)];
LABEL_11:
  v103.m128i_i32[0] = v17 & (v17 - 1);
  if ( v103.m128i_i32[0] )
    return 0;
  v18 = a1[1];
  v19 = *(__int64 (**)())(*(_QWORD *)v18 + 896LL);
  if ( v19 == sub_1F3CBC0 || !((unsigned __int8 (__fastcall *)(__int64, __int64, _QWORD))v19)(v18, a2, 0) )
    return 0;
  v67 = a1[1];
  v20 = *(unsigned __int16 *)(a2 + 24);
  v115[0] = (unsigned __int64)v116;
  v115[1] = 0x400000000LL;
  if ( !(unsigned __int8)sub_1F6D830(v104.m128i_u32[0], v104.m128i_i64[1], a2, v6, v8, v20, (__int64)v115, v67) )
  {
LABEL_45:
    result = 0;
    goto LABEL_46;
  }
  v10 = *(_WORD *)(a2 + 24) == 142;
  v21 = _mm_load_si128(&v104);
  v105.m128_i8[0] = v94;
  v79 = !v10 + 2;
  v105.m128_u64[1] = v99;
  v106 = v21;
  while ( !v106.m128i_i8[0] )
  {
    if ( v105.m128_i8[0] )
      goto LABEL_41;
LABEL_44:
    v58 = sub_1F58D30((__int64)&v105);
LABEL_42:
    if ( v58 <= 1 )
      goto LABEL_45;
    sub_1D19A30((__int64)&v120, *a1, &v106);
    a3 = _mm_load_si128(&v120);
    v59 = *a1;
    v106 = a3;
    sub_1D19A30((__int64)&v120, v59, &v105);
    a4 = _mm_load_si128(&v120);
    v105 = (__m128)a4;
  }
  if ( !v105.m128_i8[0] )
    goto LABEL_44;
  v22 = a1[1];
  if ( (((int)*(unsigned __int16 *)(v22 + 2 * (v105.m128_u8[0] + 115LL * v106.m128i_u8[0] + 16104)) >> (4 * v79)) & 0xB) != 0 )
  {
LABEL_41:
    v58 = word_42FA680[(unsigned __int8)(v105.m128_i8[0] - 14)];
    goto LABEL_42;
  }
  v23 = v6;
  v107 = *(_QWORD *)(v15 + 72);
  if ( v107 )
    sub_1F6CA20(&v107);
  v108 = *(_DWORD *)(v15 + 64);
  v95 = sub_1D15970(&v104);
  v88 = sub_1D15970(&v106);
  v78 = v95 / v88;
  v26 = sub_1D159A0((char *)&v105, v22, v95 % v88, v88, v24, v25, v68, v70, v72, v74);
  v117 = v119;
  v80 = (unsigned int)(v26 + 7) >> 3;
  v120.m128i_i64[0] = (__int64)v121;
  v120.m128i_i64[1] = 0x400000000LL;
  v118 = 0x400000000LL;
  v27 = *(_QWORD *)(v6 + 32);
  v28 = _mm_loadu_si128((const __m128i *)(v27 + 40));
  v101 = *(__int64 **)(v27 + 40);
  v85 = *(_DWORD *)(v27 + 48);
  if ( v95 >= v88 )
  {
    v77 = v8;
    v76 = v15;
    v29 = v81;
    v30 = v6;
    v31 = v28.m128i_u64[1];
    v75 = v6;
    v32 = v73;
    v84 = 0;
    do
    {
      v52 = sub_1E34390(*(_QWORD *)(v30 + 104));
      v36 = (v84 | v52) & (unsigned __int64)-(__int64)(v84 | v52);
      v38 = (_QWORD *)*a1;
      v53 = *(_QWORD *)(v30 + 104);
      a5 = _mm_loadu_si128((const __m128i *)(v53 + 40));
      v111 = a5;
      v112 = *(_QWORD *)(v53 + 56);
      v37 = *(_WORD *)(v53 + 32);
      v54 = *(_QWORD *)v53 & 0xFFFFFFFFFFFFFFF8LL;
      if ( v54 )
      {
        v55 = *(_QWORD *)(v53 + 8) + v84;
        v56 = *(_BYTE *)(v53 + 16);
        if ( (*(_QWORD *)v53 & 4) != 0 )
        {
          *((_QWORD *)&v113 + 1) = *(_QWORD *)(v53 + 8) + v84;
          LOBYTE(v114) = v56;
          *(_QWORD *)&v113 = v54 | 4;
          HIDWORD(v114) = *(_DWORD *)(v54 + 12);
        }
        else
        {
          v57 = *(_QWORD *)v54;
          *(_QWORD *)&v113 = v54;
          *((_QWORD *)&v113 + 1) = v55;
          v10 = *(_BYTE *)(v57 + 8) == 16;
          LOBYTE(v114) = v56;
          if ( v10 )
            v57 = **(_QWORD **)(v57 + 16);
          HIDWORD(v114) = *(_DWORD *)(v57 + 8) >> 8;
        }
      }
      else
      {
        v114 = 0;
        v33 = *(_DWORD *)(v53 + 20);
        v113 = 0u;
        HIDWORD(v114) = v33;
      }
      v34 = *(_QWORD *)(v30 + 72);
      v35 = *(__int128 **)(v30 + 32);
      v109 = v34;
      if ( v34 )
      {
        v69 = v36;
        v71 = v35;
        v82 = v37;
        v96 = v38;
        sub_1623A60((__int64)&v109, v34, 2);
        LODWORD(v36) = v69;
        v35 = v71;
        v37 = v82;
        v38 = v96;
      }
      v110 = *(_DWORD *)(v30 + 64);
      v39 = v85 | v31 & 0xFFFFFFFF00000000LL;
      v40 = sub_1D2B810(
              v38,
              v79,
              (__int64)&v109,
              v106.m128i_u32[0],
              v106.m128i_i64[1],
              v36,
              *v35,
              (__int64)v101,
              v39,
              v113,
              v114,
              v105.m128_i64[0],
              v105.m128_i64[1],
              v37,
              (__int64)&v111);
      v41 = v85;
      v97 = v40;
      if ( v109 )
      {
        sub_161E7C0((__int64)&v109, v109);
        v41 = v85;
      }
      v83 = 16 * v41;
      v42 = v101[5] + 16 * v41;
      LOBYTE(v32) = *(_BYTE *)v42;
      v86 = (__int64 *)*a1;
      *(_QWORD *)&v43 = sub_1D38BB0(
                          *a1,
                          v80,
                          (__int64)&v107,
                          v32,
                          *(const void ***)(v42 + 8),
                          0,
                          a3,
                          *(double *)a4.m128i_i64,
                          a5,
                          0);
      v44 = v101[5] + v83;
      LOBYTE(v29) = *(_BYTE *)v44;
      v101 = sub_1D332F0(
               v86,
               52,
               (__int64)&v107,
               v29,
               *(const void ***)(v44 + 8),
               0,
               *(double *)a3.m128i_i64,
               *(double *)a4.m128i_i64,
               a5,
               (__int64)v101,
               v39,
               v43);
      v85 = v47;
      v31 = v47 | v39 & 0xFFFFFFFF00000000LL;
      v48 = (unsigned int)v118;
      if ( (unsigned int)v118 >= HIDWORD(v118) )
      {
        sub_16CD150((__int64)&v117, v119, 0, 16, v45, v46);
        v48 = (unsigned int)v118;
      }
      v49 = (__int64 *)&v117[16 * v48];
      v49[1] = 0;
      *v49 = v97;
      v50 = v120.m128i_u32[2];
      LODWORD(v118) = v118 + 1;
      if ( v120.m128i_i32[2] >= (unsigned __int32)v120.m128i_i32[3] )
      {
        sub_16CD150((__int64)&v120, v121, 0, 16, v45, v46);
        v50 = v120.m128i_u32[2];
      }
      v51 = (__int64 *)(v120.m128i_i64[0] + 16 * v50);
      v51[1] = 1;
      *v51 = v97;
      ++v103.m128i_i32[0];
      ++v120.m128i_i32[2];
      v84 += v80;
    }
    while ( v78 > v103.m128i_i32[0] );
    v23 = v30;
    v8 = v77;
    v15 = v76;
    v6 = v75;
  }
  *((_QWORD *)&v66 + 1) = v120.m128i_u32[2];
  *(_QWORD *)&v66 = v120.m128i_i64[0];
  v102 = sub_1D359D0(
           (__int64 *)*a1,
           2,
           (__int64)&v107,
           1,
           0,
           0,
           *(double *)a3.m128i_i64,
           *(double *)a4.m128i_i64,
           a5,
           v66);
  v98 = v60;
  *((_QWORD *)&v65 + 1) = (unsigned int)v118;
  *(_QWORD *)&v65 = v117;
  v61.m128i_i64[0] = (__int64)sub_1D359D0(
                                (__int64 *)*a1,
                                107,
                                (__int64)&v107,
                                v104.m128i_u32[0],
                                (const void **)v104.m128i_i64[1],
                                0,
                                *(double *)a3.m128i_i64,
                                *(double *)a4.m128i_i64,
                                a5,
                                v65);
  v103 = v61;
  sub_1F81BC0((__int64)a1, (__int64)v102);
  v111 = _mm_load_si128(&v103);
  sub_1F994A0((__int64)a1, v15, v111.m128i_i64, 1, 1);
  v62 = (unsigned __int8 *)(*(_QWORD *)(v23 + 40) + v90);
  v87 = (__int64 *)*a1;
  v89 = (const void **)*((_QWORD *)v62 + 1);
  v91 = *v62;
  sub_1F80610((__int64)&v111, v6);
  v92 = sub_1D309E0(
          v87,
          145,
          (__int64)&v111,
          v91,
          v89,
          0,
          *(double *)a3.m128i_i64,
          *(double *)a4.m128i_i64,
          *(double *)a5.m128i_i64,
          *(_OWORD *)&v103);
  v93 = v63;
  sub_17CD270(v111.m128i_i64);
  sub_1FA0970(
    (__int64 **)a1,
    (__int64)v115,
    v23,
    v8,
    v103.m128i_i64[0],
    v103.m128i_i64[1],
    (__m128)a3,
    *(double *)a4.m128i_i64,
    a5,
    *(unsigned __int16 *)(v15 + 24));
  sub_1F9A400((__int64)a1, v23, v92, v93, (__int64)v102, v98, 1);
  v64 = v15;
  if ( (_BYTE *)v120.m128i_i64[0] != v121 )
  {
    v103.m128i_i64[0] = v15;
    _libc_free(v120.m128i_u64[0]);
    v64 = v103.m128i_i64[0];
  }
  if ( v117 != v119 )
  {
    v103.m128i_i64[0] = v64;
    _libc_free((unsigned __int64)v117);
    v64 = v103.m128i_i64[0];
  }
  v103.m128i_i64[0] = v64;
  sub_17CD270(&v107);
  result = v103.m128i_i64[0];
LABEL_46:
  if ( (_BYTE *)v115[0] != v116 )
  {
    v103.m128i_i64[0] = result;
    _libc_free(v115[0]);
    return v103.m128i_i64[0];
  }
  return result;
}
