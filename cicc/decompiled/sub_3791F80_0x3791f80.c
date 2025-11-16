// Function: sub_3791F80
// Address: 0x3791f80
//
unsigned __int8 *__fastcall sub_3791F80(__int64 *a1, __int64 a2)
{
  __int64 *v2; // r12
  __int64 *v3; // rax
  __int64 v4; // r15
  __int64 v5; // r13
  int v7; // eax
  __int64 *v8; // rbx
  unsigned __int16 *v9; // rdx
  int v10; // eax
  __int64 v11; // rcx
  __int64 v12; // rdx
  __int64 v13; // r8
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rdx
  __int16 v17; // ax
  __int64 v18; // rdx
  __m128i v19; // rax
  __int64 v20; // rax
  __m128i v21; // xmm0
  __int16 v22; // ax
  __int64 v23; // r8
  __int64 v24; // r9
  __int64 v25; // rsi
  __int64 v26; // rdx
  __int64 v27; // rcx
  __int64 v28; // r8
  unsigned __int64 v29; // rax
  __int64 v30; // r13
  __int64 v31; // rdx
  unsigned __int16 v32; // cx
  bool v33; // di
  unsigned int v34; // esi
  int v35; // esi
  unsigned int v36; // r15d
  __int64 v37; // rdx
  __int64 v38; // rdx
  unsigned __int64 v39; // rax
  __int64 v40; // rdx
  __int64 *v41; // r12
  unsigned __int64 v42; // r15
  __int64 v43; // r13
  int v44; // eax
  __int64 v45; // rsi
  __int64 v46; // rdx
  __int32 v47; // eax
  __int64 v48; // rdx
  __int64 v49; // rdx
  int v50; // eax
  unsigned int v51; // eax
  __int64 v52; // rdx
  unsigned int v53; // eax
  __int64 v54; // rdx
  __int64 v55; // r9
  __int64 v56; // rdx
  __int64 v57; // rsi
  unsigned int v58; // eax
  __int64 *v59; // r14
  __int64 v60; // rdx
  __int64 v61; // r12
  unsigned int v62; // ebx
  __int64 (__fastcall *v63)(__int64, __int64, unsigned int, __int64); // rax
  __int64 v64; // rdx
  _QWORD *v65; // rax
  int v66; // edx
  int v67; // edx
  unsigned int v68; // eax
  __int64 v69; // rdx
  __int64 v70; // rdx
  unsigned int v71; // eax
  __int64 v72; // rsi
  __int64 v73; // rdx
  __int32 v74; // eax
  __int64 v75; // rdx
  unsigned int v76; // edx
  __int64 v77; // r9
  __int64 v78; // rsi
  const __m128i *v79; // rax
  unsigned __int64 v80; // r12
  __m128i v81; // kr00_16
  unsigned int v82; // edx
  __int64 v83; // r9
  unsigned __int8 *v84; // rax
  __int64 v85; // r9
  __int64 v86; // r10
  __int64 v87; // r11
  __int64 v88; // rsi
  unsigned int v89; // edx
  _QWORD *v90; // r12
  unsigned __int8 *v91; // r15
  unsigned int v92; // edx
  unsigned __int64 v93; // r12
  unsigned int v94; // r9d
  __int64 v95; // rsi
  __int64 v96; // rcx
  __int64 v97; // rdx
  __int128 v98; // [rsp-20h] [rbp-180h]
  __int128 v99; // [rsp-10h] [rbp-170h]
  __int16 v100; // [rsp+Ah] [rbp-156h]
  __int64 v101; // [rsp+10h] [rbp-150h]
  __int64 v102; // [rsp+18h] [rbp-148h]
  unsigned int v103; // [rsp+18h] [rbp-148h]
  __int64 *v104; // [rsp+20h] [rbp-140h]
  __int64 v105; // [rsp+20h] [rbp-140h]
  unsigned int v106; // [rsp+20h] [rbp-140h]
  __int64 v107; // [rsp+30h] [rbp-130h]
  unsigned __int16 v108; // [rsp+30h] [rbp-130h]
  __m128i v109; // [rsp+30h] [rbp-130h]
  __int64 v111; // [rsp+30h] [rbp-130h]
  __m128i v112; // [rsp+40h] [rbp-120h]
  __int64 v114; // [rsp+40h] [rbp-120h]
  __int64 v115; // [rsp+90h] [rbp-D0h] BYREF
  __int64 v116; // [rsp+98h] [rbp-C8h]
  __m128i v117; // [rsp+A0h] [rbp-C0h] BYREF
  __m128i v118; // [rsp+B0h] [rbp-B0h] BYREF
  __m128i v119; // [rsp+C0h] [rbp-A0h] BYREF
  __m128i v120; // [rsp+D0h] [rbp-90h] BYREF
  __m128i v121; // [rsp+E0h] [rbp-80h] BYREF
  __m128i v122; // [rsp+F0h] [rbp-70h] BYREF
  __int64 v123; // [rsp+100h] [rbp-60h]
  __int64 v124; // [rsp+108h] [rbp-58h]
  __m128i v125; // [rsp+110h] [rbp-50h] BYREF
  __int64 v126; // [rsp+120h] [rbp-40h]

  v2 = *(__int64 **)(a1[1] + 64);
  v3 = *(__int64 **)(a2 + 40);
  v4 = *v3;
  v5 = v3[1];
  if ( *(_DWORD *)(a2 + 24) != 206 )
    return 0;
  v7 = *(_DWORD *)(v4 + 24);
  v8 = a1;
  if ( v7 > 148 )
  {
    if ( v7 != 208 && (unsigned int)(v7 - 186) > 2 )
      return 0;
  }
  else if ( v7 <= 146 )
  {
    return 0;
  }
  v9 = *(unsigned __int16 **)(v4 + 48);
  v10 = *v9;
  v11 = *((_QWORD *)v9 + 1);
  LOWORD(v115) = v10;
  v116 = v11;
  if ( (_WORD)v10 )
  {
    if ( (unsigned __int16)(v10 - 17) > 0xD3u )
    {
      v125.m128i_i16[0] = v10;
      v125.m128i_i64[1] = v11;
      goto LABEL_30;
    }
    LOWORD(v10) = word_4456580[v10 - 1];
    v37 = 0;
LABEL_34:
    v125.m128i_i16[0] = v10;
    v125.m128i_i64[1] = v37;
    if ( !(_WORD)v10 )
      goto LABEL_8;
LABEL_30:
    if ( (_WORD)v10 == 1 || (unsigned __int16)(v10 - 504) <= 7u )
      BUG();
    v14 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v10 - 16];
    goto LABEL_9;
  }
  v107 = v11;
  if ( sub_30070B0((__int64)&v115) )
  {
    LOWORD(v10) = sub_3009970((__int64)&v115, a2, v12, v107, v13);
    goto LABEL_34;
  }
  v125.m128i_i64[1] = v107;
  v125.m128i_i16[0] = 0;
LABEL_8:
  v14 = sub_3007260((__int64)&v125);
  v123 = v14;
  v124 = v15;
LABEL_9:
  if ( v14 != 1 )
    return 0;
  v16 = *(_QWORD *)(a2 + 48);
  v17 = *(_WORD *)v16;
  v18 = *(_QWORD *)(v16 + 8);
  v117.m128i_i16[0] = v17;
  v117.m128i_i64[1] = v18;
  if ( v17 )
  {
    if ( (unsigned __int16)(v17 - 176) <= 0x34u )
      return 0;
  }
  else if ( sub_3007100((__int64)&v117) )
  {
    return 0;
  }
  v19.m128i_i64[0] = sub_2D5B750((unsigned __int16 *)&v117);
  v125 = v19;
  v20 = sub_CA1930(&v125);
  if ( !v20 || (v20 & (v20 - 1)) != 0 )
    return 0;
  v21 = _mm_loadu_si128(&v117);
  v104 = v2;
  v22 = v117.m128i_i16[0];
  v118 = v21;
  WORD1(v2) = v100;
  v23 = v21.m128i_i64[1];
  v102 = v4;
  v101 = v5;
  while ( 1 )
  {
    LOWORD(v2) = v22;
    v25 = *v8;
    sub_2FE6CC0((__int64)&v125, *v8, *(_QWORD *)(v8[1] + 64), (unsigned int)v2, v23);
    if ( v125.m128i_i8[0] != 6 )
      break;
    LOWORD(v29) = v118.m128i_i16[0];
    if ( v118.m128i_i16[0] )
    {
      v30 = 0;
      v31 = v118.m128i_u16[0] - 1;
      v32 = word_4456580[v31];
LABEL_22:
      v33 = (unsigned __int16)(v29 - 176) <= 0x34u;
      v34 = word_4456340[v31];
      LOBYTE(v29) = v33;
      goto LABEL_23;
    }
    v32 = sub_3009970((__int64)&v118, v25, v26, v27, v28);
    LOWORD(v29) = v118.m128i_i16[0];
    v30 = v38;
    if ( v118.m128i_i16[0] )
    {
      v31 = v118.m128i_u16[0] - 1;
      goto LABEL_22;
    }
    v108 = v32;
    v39 = sub_3007240((__int64)&v118);
    v32 = v108;
    v34 = v39;
    v29 = HIDWORD(v39);
    v33 = v29;
LABEL_23:
    v35 = v34 >> 1;
    v125.m128i_i8[4] = v29;
    v36 = v32;
    v125.m128i_i32[0] = v35;
    if ( v33 )
      v22 = sub_2D43AD0(v32, v35);
    else
      v22 = sub_2D43050(v32, v35);
    v23 = 0;
    if ( !v22 )
    {
      v22 = sub_3009450(v104, v36, v30, v125.m128i_i64[0], 0, v24);
      v23 = v40;
    }
    v118.m128i_i16[0] = v22;
    v118.m128i_i64[1] = v23;
  }
  v41 = v104;
  v42 = v102;
  v43 = v101;
  if ( (unsigned int)sub_3281500(&v118, v25) == 1 )
    return 0;
  v44 = *(_DWORD *)(v102 + 24);
  if ( v44 > 148 )
  {
    if ( v44 != 208 )
      goto LABEL_44;
LABEL_60:
    v58 = sub_3774A40(v102);
    v59 = v8;
    v61 = v60;
    v62 = v58;
    while ( 1 )
    {
      sub_2FE6CC0((__int64)&v125, *v59, (__int64)v104, v62, v61);
      if ( !v125.m128i_i8[0] )
        break;
      v63 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*v59 + 592LL);
      if ( v63 == sub_2D56A50 )
      {
        sub_2FE6CC0((__int64)&v125, *v59, (__int64)v104, v62, v61);
        LOWORD(v62) = v125.m128i_i16[4];
        v61 = v126;
      }
      else
      {
        v62 = v63(*v59, (__int64)v104, v62, v61);
        v61 = v64;
      }
    }
    v94 = v62;
    v8 = v59;
    v95 = v59[1];
    v96 = v61;
    v42 = v102;
    v41 = v104;
    v43 = v101;
    v125.m128i_i32[0] = sub_3774AA0(*v59, v95, v94, v96);
    v125.m128i_i64[1] = v97;
    if ( sub_32844A0((unsigned __int16 *)&v125, v95) == 1 )
      return 0;
    goto LABEL_45;
  }
  if ( v44 > 146 )
    goto LABEL_60;
LABEL_44:
  if ( sub_3281100((unsigned __int16 *)&v115, v25) == 2 )
  {
    while ( 1 )
    {
      v57 = *v8;
      sub_2FE6CC0((__int64)&v125, *v8, (__int64)v104, v115, v116);
      if ( !v125.m128i_i8[0] )
        break;
      LODWORD(v115) = (*(__int64 (__fastcall **)(__int64, __int64 *, _QWORD, __int64))(*(_QWORD *)*v8 + 592LL))(
                        *v8,
                        v104,
                        (unsigned int)v115,
                        v116);
      v116 = v56;
    }
    if ( sub_3281100((unsigned __int16 *)&v115, v57) == 2 )
      return 0;
  }
LABEL_45:
  v45 = *v8;
  sub_2FE6CC0((__int64)&v125, *v8, *(_QWORD *)(v8[1] + 64), v117.m128i_u16[0], v117.m128i_i64[1]);
  if ( v125.m128i_i8[0] == 7 )
  {
    v45 = (__int64)v41;
    v117.m128i_i32[0] = (*(__int64 (__fastcall **)(__int64, __int64 *, _QWORD, __int64))(*(_QWORD *)*v8 + 592LL))(
                          *v8,
                          v41,
                          v117.m128i_u32[0],
                          v117.m128i_i64[1]);
    v117.m128i_i64[1] = v46;
  }
  v119 = _mm_loadu_si128(&v117);
  LOWORD(v47) = sub_3281100((unsigned __int16 *)&v119, v45);
  v125.m128i_i32[0] = v47;
  v125.m128i_i64[1] = v48;
  if ( (_WORD)v47 )
  {
    if ( (unsigned __int16)(v47 - 2) > 7u
      && (unsigned __int16)(v47 - 17) > 0x6Cu
      && (unsigned __int16)(v47 - 176) > 0x1Fu )
    {
LABEL_51:
      v119.m128i_i32[0] = sub_327FDF0((unsigned __int16 *)&v119, v45);
      v119.m128i_i64[1] = v49;
    }
  }
  else if ( !sub_3007070((__int64)&v125) )
  {
    goto LABEL_51;
  }
  v50 = *(_DWORD *)(v42 + 24);
  if ( v50 > 148 )
  {
    if ( v50 == 208 )
      goto LABEL_54;
    if ( (unsigned int)(v50 - 186) > 2 )
      return 0;
    v65 = *(_QWORD **)(v42 + 40);
    v66 = *(_DWORD *)(*v65 + 24LL);
    if ( v66 > 148 )
    {
      if ( v66 != 208 )
        return 0;
    }
    else if ( v66 <= 146 )
    {
      return 0;
    }
    v67 = *(_DWORD *)(v65[5] + 24LL);
    if ( v67 > 148 )
    {
      if ( v67 != 208 )
        return 0;
    }
    else if ( v67 <= 146 )
    {
      return 0;
    }
    v105 = v65[5];
    v112 = _mm_loadu_si128((const __m128i *)v65);
    v109 = _mm_loadu_si128((const __m128i *)(v65 + 5));
    v68 = sub_3774A40(*v65);
    v120.m128i_i32[0] = sub_3774AA0(*v8, v8[1], v68, v69);
    v120.m128i_i64[1] = v70;
    v71 = sub_3774A40(v105);
    v72 = v8[1];
    v74 = sub_3774AA0(*v8, v72, v71, v73);
    v121.m128i_i64[1] = v75;
    v121.m128i_i32[0] = v74;
    v106 = sub_32844A0((unsigned __int16 *)&v120, v72);
    v103 = sub_32844A0((unsigned __int16 *)&v121, v72);
    v76 = sub_32844A0((unsigned __int16 *)&v119, v72);
    if ( v106 == v103 )
    {
      v81 = v120;
    }
    else
    {
      if ( v106 >= v103 )
        v122 = _mm_loadu_si128(&v121);
      else
        v122 = _mm_loadu_si128(&v120);
      v78 = v122.m128i_u16[0];
      v79 = &v120;
      if ( v120.m128i_i16[0] == v122.m128i_i16[0] )
      {
        v79 = &v121;
        if ( !v122.m128i_i16[0] && v120.m128i_i64[1] != v122.m128i_i64[1] )
          v79 = &v120;
      }
      v80 = v76;
      v125 = _mm_loadu_si128(v79);
      if ( sub_32844A0((unsigned __int16 *)&v125, v122.m128i_u16[0]) > (unsigned __int64)v76 )
      {
        if ( sub_32844A0((unsigned __int16 *)&v122, v78) < v80 )
          v81 = v119;
        else
          v81 = v122;
      }
      else
      {
        v81 = v125;
      }
    }
    v112.m128i_i64[0] = (__int64)sub_3791380(
                                   (__int64)v8,
                                   v112.m128i_u64[0],
                                   v21,
                                   v112.m128i_i64[1],
                                   v120.m128i_u32[0],
                                   v120.m128i_i64[1],
                                   v77,
                                   v81.m128i_u32[0],
                                   v81.m128i_i64[1]);
    v112.m128i_i64[1] = v82 | v112.m128i_i64[1] & 0xFFFFFFFF00000000LL;
    v84 = sub_3791380(
            (__int64)v8,
            v109.m128i_u64[0],
            v21,
            v109.m128i_i64[1],
            v121.m128i_u32[0],
            v121.m128i_i64[1],
            v83,
            v81.m128i_u32[0],
            v81.m128i_i64[1]);
    v87 = *(_OWORD *)&v81 >> 64;
    v86 = (__int64)v81;
    v109.m128i_i64[0] = (__int64)v84;
    v88 = *(_QWORD *)(v42 + 80);
    v90 = (_QWORD *)v8[1];
    v125.m128i_i64[0] = v88;
    v109.m128i_i64[1] = v89 | v109.m128i_i64[1] & 0xFFFFFFFF00000000LL;
    if ( v88 )
    {
      sub_B96E90((__int64)&v125, v88, 1);
      v87 = *(_OWORD *)&v81 >> 64;
      v86 = (__int64)v81;
    }
    v125.m128i_i32[2] = *(_DWORD *)(v42 + 72);
    v99 = (__int128)v109;
    v98 = (__int128)v112;
    v111 = v86;
    v114 = v87;
    v91 = sub_3406EB0(v90, *(_DWORD *)(v42 + 24), (__int64)&v125, (unsigned int)v86, v87, v85, v98, v99);
    v93 = v92 | v43 & 0xFFFFFFFF00000000LL;
    sub_9C6650(&v125);
    return sub_3791380(
             (__int64)v8,
             (unsigned __int64)v91,
             v21,
             v93,
             (unsigned int)v111,
             v114,
             v111,
             v119.m128i_u32[0],
             v119.m128i_i64[1]);
  }
  else
  {
    if ( v50 <= 146 )
      return 0;
LABEL_54:
    v51 = sub_3774A40(v42);
    v53 = sub_3774AA0(*v8, v8[1], v51, v52);
    return sub_3791380((__int64)v8, v42, v21, v43, v53, v54, v55, v119.m128i_u32[0], v119.m128i_i64[1]);
  }
}
