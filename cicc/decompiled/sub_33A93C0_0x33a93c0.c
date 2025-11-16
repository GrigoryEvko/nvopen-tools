// Function: sub_33A93C0
// Address: 0x33a93c0
//
__int64 __fastcall sub_33A93C0(__int64 a1, __int64 a2)
{
  __int64 v4; // rdx
  __int64 v5; // r14
  __int64 v6; // r12
  __int64 v7; // rsi
  __int64 v8; // r13
  bool v9; // r11
  __int64 v10; // rdx
  _QWORD *v11; // rax
  __int64 v12; // rax
  int v13; // eax
  __int64 v14; // rax
  int v15; // eax
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rcx
  __int64 v21; // r10
  __int64 v22; // rdx
  __int64 v23; // r11
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rsi
  __int8 v27; // r11
  __int64 v28; // rax
  _QWORD *v29; // rsi
  unsigned int v30; // esi
  _QWORD *v31; // r13
  __int64 (__fastcall *v32)(__int64, int); // rax
  __int32 v33; // eax
  __int32 v34; // r10d
  unsigned int v35; // r11d
  unsigned __int16 v37; // r13
  __int64 v38; // rax
  __int64 v39; // rdx
  __int64 v40; // rax
  __int8 v41; // r11
  unsigned __int64 v42; // rdx
  unsigned __int64 v43; // r13
  int v44; // edx
  __int64 v45; // rax
  __int64 v46; // r14
  __int64 v47; // rsi
  __int128 v48; // rax
  __int64 v49; // rax
  __int8 v50; // r11
  __int64 v51; // r13
  __int64 v52; // rdx
  __int64 v53; // r14
  __int64 v54; // rcx
  __int64 v55; // rdx
  unsigned int v56; // eax
  __int64 v57; // rcx
  __int64 (*v58)(); // rax
  __int64 (*v59)(); // rax
  __int64 v60; // rax
  __int64 v61; // rdx
  __int64 v62; // rax
  unsigned __int64 v63; // rdx
  __int64 v64; // rax
  char v65; // cl
  unsigned __int64 v66; // rax
  _QWORD *v67; // rax
  __int8 v68; // r11
  int v69; // eax
  __int64 v70; // rdx
  __int64 v71; // r14
  unsigned int v72; // ecx
  __int64 v73; // rax
  __int64 v74; // rdi
  __int64 v75; // rdx
  __int64 v76; // rax
  __int64 v77; // rdx
  __int64 v78; // rax
  __int64 *v79; // r13
  __int64 v80; // r12
  __int64 v81; // rax
  int v82; // eax
  int v83; // edx
  __int64 v84; // r13
  int v85; // r14d
  __int64 v86; // rax
  int v87; // r8d
  __int8 v88; // r11
  bool v89; // zf
  __int64 v90; // rsi
  __int64 v91; // r14
  __int64 v92; // rdx
  __int64 v93; // r13
  _QWORD *v94; // rax
  __int64 v95; // rax
  __int64 v96; // r8
  __int64 v97; // r9
  __int64 v98; // rax
  __m128i si128; // xmm0
  __int64 v100; // [rsp+0h] [rbp-130h]
  __int8 v101; // [rsp+0h] [rbp-130h]
  __int64 v102; // [rsp+8h] [rbp-128h]
  unsigned int v103; // [rsp+10h] [rbp-120h]
  __int16 v104; // [rsp+12h] [rbp-11Eh]
  __int64 v105; // [rsp+18h] [rbp-118h]
  unsigned __int16 v106; // [rsp+18h] [rbp-118h]
  __int64 v107; // [rsp+20h] [rbp-110h]
  __int8 v108; // [rsp+20h] [rbp-110h]
  __int8 v109; // [rsp+20h] [rbp-110h]
  __int8 v110; // [rsp+20h] [rbp-110h]
  int v111; // [rsp+20h] [rbp-110h]
  unsigned int v112; // [rsp+20h] [rbp-110h]
  __int8 v113; // [rsp+20h] [rbp-110h]
  __int64 v114; // [rsp+28h] [rbp-108h]
  __int64 v115; // [rsp+30h] [rbp-100h]
  __int64 v116; // [rsp+30h] [rbp-100h]
  __int8 v117; // [rsp+30h] [rbp-100h]
  unsigned __int16 v118; // [rsp+30h] [rbp-100h]
  __int64 v119; // [rsp+38h] [rbp-F8h]
  __int64 v120; // [rsp+40h] [rbp-F0h]
  __int128 v121; // [rsp+40h] [rbp-F0h]
  unsigned int v122; // [rsp+40h] [rbp-F0h]
  int v123; // [rsp+40h] [rbp-F0h]
  __m128i v124; // [rsp+50h] [rbp-E0h] BYREF
  __int64 v125; // [rsp+60h] [rbp-D0h]
  __int64 v126; // [rsp+68h] [rbp-C8h]
  __int64 v127; // [rsp+70h] [rbp-C0h]
  __int64 v128; // [rsp+78h] [rbp-B8h]
  __int64 v129; // [rsp+80h] [rbp-B0h]
  __int64 v130; // [rsp+88h] [rbp-A8h]
  __int64 v131; // [rsp+90h] [rbp-A0h] BYREF
  int v132; // [rsp+98h] [rbp-98h]
  unsigned __int64 v133; // [rsp+A0h] [rbp-90h]
  __int64 v134; // [rsp+A8h] [rbp-88h]
  __int64 v135; // [rsp+B0h] [rbp-80h]
  unsigned __int64 v136; // [rsp+C0h] [rbp-70h] BYREF
  __int64 v137; // [rsp+C8h] [rbp-68h]
  __int64 v138; // [rsp+D0h] [rbp-60h]
  __int64 v139; // [rsp+E0h] [rbp-50h] BYREF
  __int64 v140; // [rsp+E8h] [rbp-48h]
  __m128i v141[4]; // [rsp+F0h] [rbp-40h] BYREF

  v4 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  v5 = *(_QWORD *)(a2 - 32 * v4);
  v6 = *(_QWORD *)(a2 + 32 * (1 - v4));
  v124.m128i_i64[0] = *(_QWORD *)(a2 + 32 * (2 - v4));
  v7 = v124.m128i_i64[0];
  v8 = sub_338B750(a1, v124.m128i_i64[0]);
  v9 = *(_DWORD *)(v8 + 24) == 35 || *(_DWORD *)(v8 + 24) == 11;
  if ( v9 )
  {
    v10 = *(_QWORD *)(v8 + 96);
    v11 = *(_QWORD **)(v10 + 24);
    if ( *(_DWORD *)(v10 + 32) > 0x40u )
      v11 = (_QWORD *)*v11;
    if ( !v11 )
    {
      v78 = *(_QWORD *)(a1 + 864);
      v79 = *(__int64 **)(a2 + 8);
      v124.m128i_i8[0] = v9;
      v80 = *(_QWORD *)(v78 + 16);
      v81 = sub_2E79000(*(__int64 **)(v78 + 40));
      v82 = sub_2D5BAE0(v80, v81, v79, 1);
      v139 = 0;
      v84 = *(_QWORD *)(a1 + 864);
      v85 = v82;
      v86 = *(_QWORD *)a1;
      v87 = v83;
      v88 = v124.m128i_i8[0];
      v89 = *(_QWORD *)a1 == 0;
      LODWORD(v140) = *(_DWORD *)(a1 + 848);
      if ( !v89 && &v139 != (__int64 *)(v86 + 48) )
      {
        v90 = *(_QWORD *)(v86 + 48);
        v139 = v90;
        if ( v90 )
        {
          v123 = v83;
          sub_B96E90((__int64)&v139, v90, 1);
          v87 = v123;
          v88 = v124.m128i_i8[0];
        }
      }
      v124.m128i_i8[0] = v88;
      v136 = a2;
      v91 = sub_3400BD0(v84, 0, (unsigned int)&v139, v85, v87, 0, 0);
      v93 = v92;
      v94 = sub_337DC20(a1 + 8, (__int64 *)&v136);
      v130 = v93;
      v35 = v124.m128i_u8[0];
      v129 = v91;
      *v94 = v91;
      *((_DWORD *)v94 + 2) = v130;
      if ( v139 )
      {
        sub_B91220((__int64)&v139, v139);
        return v124.m128i_u8[0];
      }
      return v35;
    }
  }
  else
  {
    v8 = 0;
  }
  v120 = *(_QWORD *)(*(_QWORD *)(a1 + 864) + 8LL);
  v12 = *(_QWORD *)(*(_QWORD *)v120 + 64LL);
  v137 = 0;
  BYTE4(v138) = 0;
  v124.m128i_i64[0] = v12;
  v136 = v6 & 0xFFFFFFFFFFFFFFFBLL;
  v13 = 0;
  if ( v6 )
  {
    v14 = *(_QWORD *)(v6 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v14 + 8) - 17 <= 1 )
      v14 = **(_QWORD **)(v14 + 16);
    v13 = *(_DWORD *)(v14 + 8) >> 8;
  }
  LODWORD(v138) = v13;
  BYTE4(v135) = 0;
  v133 = v5 & 0xFFFFFFFFFFFFFFFBLL;
  v15 = 0;
  v134 = 0;
  if ( v5 )
  {
    v16 = *(_QWORD *)(v5 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v16 + 8) - 17 <= 1 )
      v16 = **(_QWORD **)(v16 + 16);
    v15 = *(_DWORD *)(v16 + 8) >> 8;
  }
  LODWORD(v135) = v15;
  v107 = sub_338B750(a1, v7);
  v114 = v17;
  v115 = sub_338B750(a1, v6);
  v119 = v18;
  v19 = sub_338B750(a1, v5);
  v20 = *(_QWORD *)(a1 + 864);
  v131 = 0;
  v21 = v19;
  v23 = v22;
  v24 = *(_QWORD *)a1;
  v132 = *(_DWORD *)(a1 + 848);
  v25 = v20;
  if ( v24 )
  {
    if ( &v131 != (__int64 *)(v24 + 48) )
    {
      v26 = *(_QWORD *)(v24 + 48);
      v131 = v26;
      if ( v26 )
      {
        v100 = v21;
        v102 = v23;
        v105 = v20;
        sub_B96E90((__int64)&v131, v26, 1);
        v25 = *(_QWORD *)(a1 + 864);
        v21 = v100;
        v23 = v102;
        v20 = v105;
      }
    }
  }
  if ( (_OWORD *(__fastcall *)(_OWORD *))v124.m128i_i64[0] == sub_3364F70 )
  {
    if ( v131 )
      sub_B91220((__int64)&v131, v131);
    goto LABEL_20;
  }
  ((void (__fastcall *)(__int64 *, __int64, __int64, __int64 *, _QWORD, _QWORD, __int64, __int64, __int64, __int64, __int64, __int64, unsigned __int64, __int64, __int64, unsigned __int64, __int64, __int64))v124.m128i_i64[0])(
    &v139,
    v120,
    v25,
    &v131,
    *(_QWORD *)(v20 + 384),
    *(_QWORD *)(v20 + 392),
    v21,
    v23,
    v115,
    v119,
    v107,
    v114,
    v133,
    v134,
    v135,
    v136,
    v137,
    v138);
  v95 = v139;
  if ( v131 )
  {
    v124.m128i_i64[0] = v139;
    sub_B91220((__int64)&v131, v131);
    v95 = v124.m128i_i64[0];
  }
  if ( !v95 )
  {
LABEL_20:
    if ( !v8 )
      return 0;
    v27 = sub_988330(a2);
    if ( !v27 )
      return 0;
    v28 = *(_QWORD *)(v8 + 96);
    v29 = *(_QWORD **)(v28 + 24);
    if ( *(_DWORD *)(v28 + 32) > 0x40u )
      v29 = (_QWORD *)*v29;
    v30 = 8 * (_DWORD)v29;
    if ( v30 == 32 )
    {
      v37 = 7;
    }
    else
    {
      if ( v30 > 0x20 )
      {
        if ( v30 == 256 )
        {
          v31 = *(_QWORD **)(*(_QWORD *)(a1 + 864) + 16LL);
          v32 = *(__int64 (__fastcall **)(__int64, int))(*v31 + 376LL);
          if ( v32 == sub_2FE4C50 )
            return 0;
        }
        else
        {
          if ( ((v30 - 64) & 0xFFFFFFB8) != 0 )
            return 0;
          v31 = *(_QWORD **)(*(_QWORD *)(a1 + 864) + 16LL);
          v32 = *(__int64 (__fastcall **)(__int64, int))(*v31 + 376LL);
          if ( v32 == sub_2FE4C50 )
          {
            if ( v30 == 64 )
            {
              v34 = 8;
              if ( !v31[22] )
                return 0;
            }
            else
            {
              if ( v30 != 128 )
                return 0;
              v34 = 9;
              if ( !v31[23] )
                return 0;
            }
            goto LABEL_44;
          }
        }
        v124.m128i_i8[0] = v27;
        v33 = ((__int64 (__fastcall *)(_QWORD *))v32)(v31);
        v27 = v124.m128i_i8[0];
        v34 = v33;
        if ( !(_WORD)v33 )
          return 0;
LABEL_44:
        v54 = *(_QWORD *)(v5 + 8);
        if ( (unsigned int)*(unsigned __int8 *)(v54 + 8) - 17 <= 1 )
          v54 = **(_QWORD **)(v54 + 16);
        v55 = *(_QWORD *)(v6 + 8);
        if ( (unsigned int)*(unsigned __int8 *)(v55 + 8) - 17 <= 1 )
          v55 = **(_QWORD **)(v55 + 16);
        v111 = (unsigned __int16)v34;
        if ( !v31[(unsigned __int16)v34 + 14] )
          return 0;
        v56 = *(_DWORD *)(v54 + 8);
        v57 = *(_DWORD *)(v55 + 8) >> 8;
        v122 = v56 >> 8;
        v58 = *(__int64 (**)())(*v31 + 808LL);
        if ( v58 == sub_2D56600 )
          return 0;
        v117 = v27;
        v124.m128i_i32[0] = v34;
        if ( !((unsigned __int8 (__fastcall *)(_QWORD *, _QWORD, _QWORD, __int64, _QWORD, _QWORD, _QWORD))v58)(
                v31,
                (unsigned __int16)v34,
                0,
                v57,
                0,
                0,
                0) )
          return 0;
        v59 = *(__int64 (**)())(*v31 + 808LL);
        if ( v59 == sub_2D56600 )
          return 0;
        v101 = v117;
        if ( !((unsigned __int8 (__fastcall *)(_QWORD *, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD))v59)(
                v31,
                v124.m128i_u16[0],
                0,
                v122,
                0,
                0,
                0) )
          return 0;
        v118 = v124.m128i_i16[0];
        v60 = sub_33A8FE0(v5, v124.m128i_u16[0], a1);
        v124.m128i_i64[0] = v61;
        v106 = v118;
        *(_QWORD *)&v121 = v60;
        v62 = sub_33A8FE0(v6, v118, a1);
        v41 = v101;
        v116 = v62;
        v43 = v63;
        if ( (unsigned __int16)(v106 - 17) > 0xD3u )
          goto LABEL_33;
        v64 = 16LL * (v111 - 1);
        v65 = byte_444C4A0[v64 + 8];
        v66 = *(_QWORD *)&byte_444C4A0[v64];
        LOBYTE(v137) = v65;
        v136 = v66;
        v112 = sub_CA1930(&v136);
        v67 = (_QWORD *)sub_BD5C60(v5);
        v68 = v101;
        switch ( v112 )
        {
          case 1u:
            LOWORD(v69) = 2;
            break;
          case 2u:
            LOWORD(v69) = 3;
            break;
          case 4u:
            LOWORD(v69) = 4;
            break;
          case 8u:
            LOWORD(v69) = 5;
            break;
          case 0x10u:
            LOWORD(v69) = 6;
            break;
          case 0x20u:
            LOWORD(v69) = 7;
            break;
          case 0x40u:
            LOWORD(v69) = 8;
            break;
          case 0x80u:
            LOWORD(v69) = 9;
            break;
          default:
            v69 = sub_3007020(v67, v112);
            v68 = v101;
            v104 = HIWORD(v69);
            v71 = v70;
LABEL_63:
            HIWORD(v72) = v104;
            v113 = v68;
            LOWORD(v72) = v69;
            v103 = v72;
            v73 = sub_33FB890(*(_QWORD *)(a1 + 864), v72, v71, v121, v124.m128i_i64[0]);
            v74 = *(_QWORD *)(a1 + 864);
            v128 = v75;
            v127 = v73;
            *(_QWORD *)&v121 = v73;
            v124.m128i_i64[0] = (unsigned int)v75 | v124.m128i_i64[0] & 0xFFFFFFFF00000000LL;
            v76 = sub_33FB890(v74, v103, v71, v116, v43);
            v41 = v113;
            v125 = v76;
            v126 = v77;
            v116 = v76;
            v43 = (unsigned int)v77 | v43 & 0xFFFFFFFF00000000LL;
LABEL_33:
            v44 = *(_DWORD *)(a1 + 848);
            v45 = *(_QWORD *)a1;
            v136 = 0;
            v46 = *(_QWORD *)(a1 + 864);
            LODWORD(v137) = v44;
            if ( v45 )
            {
              if ( &v136 != (unsigned __int64 *)(v45 + 48) )
              {
                v47 = *(_QWORD *)(v45 + 48);
                v136 = v47;
                if ( v47 )
                {
                  v109 = v41;
                  sub_B96E90((__int64)&v136, v47, 1);
                  v41 = v109;
                }
              }
            }
            v110 = v41;
            *((_QWORD *)&v121 + 1) = v124.m128i_i64[0];
            v124.m128i_i64[1] = v43;
            v124.m128i_i64[0] = v116;
            *(_QWORD *)&v48 = sub_33ED040(v46, 22);
            v49 = sub_340F900(v46, 208, (unsigned int)&v136, 2, 0, DWORD2(v121), v121, *(_OWORD *)&v124, v48);
            v50 = v110;
            v51 = v49;
            v53 = v52;
            if ( v136 )
            {
              sub_B91220((__int64)&v136, v136);
              v50 = v110;
            }
            v124.m128i_i8[0] = v50;
            sub_33809B0((__int64 *)a1, a2, v51, v53, 0);
            return v124.m128i_u8[0];
        }
        v71 = 0;
        goto LABEL_63;
      }
      v37 = 6;
      if ( v30 != 16 )
        return 0;
    }
    v108 = v27;
    v38 = sub_33A8FE0(v5, v37, a1);
    v124.m128i_i64[0] = v39;
    *(_QWORD *)&v121 = v38;
    v40 = sub_33A8FE0(v6, v37, a1);
    v41 = v108;
    v116 = v40;
    v43 = v42;
    goto LABEL_33;
  }
  sub_33809B0((__int64 *)a1, a2, v139, v140, 1);
  v98 = *(unsigned int *)(a1 + 136);
  si128 = _mm_load_si128(v141);
  if ( v98 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 140) )
  {
    v124 = si128;
    sub_C8D5F0(a1 + 128, (const void *)(a1 + 144), v98 + 1, 0x10u, v96, v97);
    v98 = *(unsigned int *)(a1 + 136);
    si128 = _mm_load_si128(&v124);
  }
  v35 = 1;
  *(__m128i *)(*(_QWORD *)(a1 + 128) + 16 * v98) = si128;
  ++*(_DWORD *)(a1 + 136);
  return v35;
}
