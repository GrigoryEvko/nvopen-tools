// Function: sub_2FEA900
// Address: 0x2fea900
//
void __fastcall sub_2FEA900(unsigned int a1, __int64 *a2, __int64 a3, __int64 a4, _WORD *a5, int a6)
{
  __int64 v7; // rsi
  unsigned int v8; // r15d
  unsigned __int16 v9; // r14
  _QWORD *v10; // rbx
  __int64 (__fastcall *v11)(__int64, __int64, __int64, __int64); // rax
  unsigned __int16 v12; // cx
  __int64 v13; // rax
  __int64 (__fastcall *v14)(__int64, __int64, __int64, __int64, unsigned __int64); // rax
  __int64 (__fastcall *v15)(__int64, __int64, __int64, unsigned __int64); // rax
  __int64 v16; // rdx
  __int64 (__fastcall *v17)(__int64, __int64, __int64, __int64, unsigned __int64); // rax
  __int64 v18; // rax
  unsigned __int64 v19; // r14
  __int64 (__fastcall *v20)(__int64, __int64, unsigned int, __int64); // rax
  unsigned __int64 v21; // rdx
  __int64 v22; // rax
  unsigned __int64 v23; // r14
  unsigned int v24; // r8d
  __int64 v25; // rbx
  unsigned __int8 v26; // al
  __int64 v27; // r8
  __int64 v28; // rcx
  unsigned __int8 v29; // al
  __int64 v30; // rdx
  int v31; // r11d
  __int64 v32; // r9
  int v33; // r14d
  unsigned __int64 v34; // rax
  __int64 v35; // rdi
  unsigned int v36; // r12d
  unsigned __int64 v37; // rsi
  unsigned __int64 v38; // r10
  const __m128i *v39; // rbx
  unsigned __int64 v40; // rcx
  __m128i *v41; // rdx
  const void *v42; // rsi
  char *v43; // rbx
  _QWORD *v44; // rbx
  unsigned int v45; // eax
  unsigned int v46; // eax
  __int64 v47; // rax
  unsigned __int64 v48; // r15
  __int64 v49; // rdx
  __int64 v50; // rax
  __int64 v51; // rdx
  __int64 v52; // rdi
  unsigned __int64 v53; // rdx
  char v54; // al
  __int64 v55; // rdx
  __int64 v56; // rdx
  int v57; // r14d
  __int64 v58; // rax
  __int64 (__fastcall *v59)(__int64, __int64, unsigned int, __int64); // rax
  __int64 v60; // rdx
  __int64 v61; // rax
  __int64 v62; // rcx
  __int64 (__fastcall *v63)(__int64, __int64, unsigned int, __int64); // rax
  __int64 v64; // rax
  unsigned __int64 v65; // rcx
  unsigned __int16 v66; // dx
  __int64 v67; // rax
  char v68; // cl
  unsigned __int64 v69; // rax
  unsigned int v70; // eax
  __int64 (__fastcall *v71)(__int64, __int64, unsigned int, __int64); // rax
  unsigned __int64 v72; // rdx
  __int64 v73; // rax
  unsigned __int64 v74; // r14
  _WORD *v75; // rsi
  __int64 (__fastcall *v76)(__int64, __int64, unsigned int, __int64); // rax
  unsigned __int64 v77; // rdx
  __int64 v78; // rax
  unsigned __int64 v79; // r14
  __int64 (__fastcall *v80)(__int64, __int64, unsigned int, __int64); // rax
  __int64 v81; // rax
  unsigned __int64 v82; // rcx
  __int32 v83; // eax
  __int64 v84; // rdx
  unsigned __int64 v85; // rdx
  unsigned __int64 v86; // rdx
  __int64 v87; // [rsp-8h] [rbp-218h]
  __int64 v88; // [rsp+0h] [rbp-210h]
  __int64 v89; // [rsp+8h] [rbp-208h]
  __int64 v90; // [rsp+10h] [rbp-200h]
  __int64 v91; // [rsp+18h] [rbp-1F8h]
  __int64 v92; // [rsp+20h] [rbp-1F0h]
  __int64 v93; // [rsp+30h] [rbp-1E0h]
  __int64 v94; // [rsp+38h] [rbp-1D8h]
  unsigned __int64 v95; // [rsp+40h] [rbp-1D0h]
  int v97; // [rsp+4Ch] [rbp-1C4h]
  unsigned int v98; // [rsp+50h] [rbp-1C0h]
  __int64 v99; // [rsp+50h] [rbp-1C0h]
  unsigned __int64 v100; // [rsp+50h] [rbp-1C0h]
  unsigned int v102; // [rsp+68h] [rbp-1A8h]
  __int64 v103; // [rsp+68h] [rbp-1A8h]
  unsigned __int16 v104; // [rsp+68h] [rbp-1A8h]
  unsigned __int16 v105; // [rsp+68h] [rbp-1A8h]
  __int64 v106; // [rsp+68h] [rbp-1A8h]
  __int64 v107; // [rsp+70h] [rbp-1A0h]
  unsigned int v108; // [rsp+70h] [rbp-1A0h]
  __int64 v109; // [rsp+78h] [rbp-198h]
  _QWORD v111[2]; // [rsp+88h] [rbp-188h] BYREF
  unsigned int v112; // [rsp+9Ch] [rbp-174h]
  __m128i v113; // [rsp+A0h] [rbp-170h] BYREF
  __int64 v114; // [rsp+B0h] [rbp-160h]
  __int64 v115; // [rsp+B8h] [rbp-158h]
  __int64 v116; // [rsp+C0h] [rbp-150h]
  __int64 v117; // [rsp+C8h] [rbp-148h]
  __m128i v118; // [rsp+D0h] [rbp-140h] BYREF
  __m128i v119; // [rsp+E0h] [rbp-130h] BYREF
  __int64 v120; // [rsp+F0h] [rbp-120h]
  __int64 v121; // [rsp+F8h] [rbp-118h]
  __int64 v122; // [rsp+100h] [rbp-110h] BYREF
  __int64 v123; // [rsp+108h] [rbp-108h]
  __int64 v124; // [rsp+110h] [rbp-100h] BYREF
  __int64 v125; // [rsp+118h] [rbp-F8h]
  __int64 v126; // [rsp+120h] [rbp-F0h] BYREF
  unsigned __int64 v127; // [rsp+128h] [rbp-E8h]
  __int64 v128; // [rsp+130h] [rbp-E0h] BYREF
  unsigned __int64 v129; // [rsp+138h] [rbp-D8h]
  __m128i v130; // [rsp+140h] [rbp-D0h] BYREF
  unsigned __int64 v131; // [rsp+150h] [rbp-C0h] BYREF
  unsigned __int64 v132; // [rsp+158h] [rbp-B8h]
  unsigned __int64 v133; // [rsp+160h] [rbp-B0h]
  __int16 v134; // [rsp+168h] [rbp-A8h]
  __int64 v135; // [rsp+170h] [rbp-A0h]
  char v136; // [rsp+178h] [rbp-98h]
  __int64 v137; // [rsp+17Ch] [rbp-94h]
  _BYTE *v138; // [rsp+190h] [rbp-80h] BYREF
  __int64 v139; // [rsp+198h] [rbp-78h]
  _BYTE v140[112]; // [rsp+1A0h] [rbp-70h] BYREF

  LOBYTE(v132) = 0;
  v131 = 0;
  v111[0] = a3;
  v138 = v140;
  v139 = 0x400000000LL;
  sub_34B8C80((_DWORD)a5, a6, (_DWORD)a2, (unsigned int)&v138, 0, 0, __PAIR128__(v132, 0));
  if ( (_DWORD)v139 )
  {
    v109 = 0;
    v94 = 16LL * (unsigned int)v139;
    while ( 1 )
    {
      v7 = 0;
      v8 = 213;
      v113 = _mm_loadu_si128((const __m128i *)&v138[v109]);
      if ( !(unsigned __int8)sub_A74710(v111, 0, 54) )
      {
        v7 = 0;
        if ( !(unsigned __int8)sub_A74710(v111, 0, 79) )
        {
          v10 = *(_QWORD **)a5;
          v107 = *a2;
          goto LABEL_13;
        }
        v8 = 214;
      }
      v9 = v113.m128i_i16[0];
      v10 = *(_QWORD **)a5;
      v107 = *a2;
      if ( v113.m128i_i16[0] )
        break;
      if ( (unsigned __int8)sub_3007070(&v113) )
        goto LABEL_8;
      v14 = (__int64 (__fastcall *)(__int64, __int64, __int64, __int64, unsigned __int64))v10[94];
      if ( v14 != sub_2FEA530 )
      {
LABEL_43:
        v44 = a5;
        v7 = v107;
        v45 = v14((__int64)a5, v107, a1, v113.m128i_u32[0], v113.m128i_u64[1]);
        goto LABEL_44;
      }
LABEL_14:
      v15 = (__int64 (__fastcall *)(__int64, __int64, __int64, unsigned __int64))v10[92];
      BYTE2(v112) = 0;
      v16 = v113.m128i_i64[0];
      if ( v15 == sub_2FEA1A0 )
      {
        v118 = v113;
        if ( v113.m128i_i16[0] )
        {
          v102 = (unsigned __int16)a5[v113.m128i_u16[0] + 1152];
          goto LABEL_17;
        }
        if ( (unsigned __int8)sub_30070B0(&v118, v7, v113.m128i_i64[0]) )
        {
          LOWORD(v128) = 0;
          v7 = v107;
          LOWORD(v131) = 0;
          v132 = 0;
          v46 = sub_2FE8D10(
                  (__int64)a5,
                  v107,
                  v118.m128i_u32[0],
                  v118.m128i_u64[1],
                  (__int64 *)&v131,
                  (unsigned int *)&v130,
                  (unsigned __int16 *)&v128);
          v10 = *(_QWORD **)a5;
          v102 = v46;
          v107 = *a2;
          goto LABEL_17;
        }
        if ( !(unsigned __int8)sub_3007070(&v118) )
          goto LABEL_116;
        v120 = sub_3007260(&v118);
        v121 = v55;
        v131 = v120;
        LOBYTE(v132) = v55;
        v57 = sub_CA1930(&v131);
        v58 = v118.m128i_u16[0];
        v119 = v118;
        if ( v118.m128i_i16[0] )
          goto LABEL_71;
        v99 = v118.m128i_i64[1];
        v106 = v118.m128i_i64[0];
        if ( (unsigned __int8)sub_30070B0(&v119, v7, v56) )
        {
          v132 = 0;
          LOWORD(v128) = 0;
          LOWORD(v131) = 0;
          sub_2FE8D10(
            (__int64)a5,
            v107,
            v119.m128i_u32[0],
            v119.m128i_u64[1],
            (__int64 *)&v131,
            (unsigned int *)&v130,
            (unsigned __int16 *)&v128);
          goto LABEL_107;
        }
        if ( !(unsigned __int8)sub_3007070(&v119) )
          goto LABEL_116;
        v7 = (__int64)a5;
        v59 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)a5 + 592LL);
        if ( v59 == sub_2D56A50 )
        {
          sub_2FE6CC0((__int64)&v131, (__int64)a5, v107, v106, v99);
          v61 = v91;
          LOWORD(v61) = v132;
          v62 = v133;
          v91 = v61;
        }
        else
        {
          v7 = v107;
          v91 = v59((__int64)a5, v107, v119.m128i_u32[0], v119.m128i_i64[1]);
          v62 = v60;
        }
        v123 = v62;
        v58 = (unsigned __int16)v91;
        v122 = v91;
        if ( (_WORD)v91 )
        {
LABEL_71:
          v66 = a5[v58 + 1426];
        }
        else
        {
          v100 = v62;
          if ( !(unsigned __int8)sub_30070B0(&v122, v7, v60) )
          {
            if ( !(unsigned __int8)sub_3007070(&v122) )
              goto LABEL_116;
            v63 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)a5 + 592LL);
            if ( v63 == sub_2D56A50 )
            {
              sub_2FE6CC0((__int64)&v131, (__int64)a5, v107, v122, v123);
              v64 = v89;
              LOWORD(v64) = v132;
              v65 = v133;
              v89 = v64;
            }
            else
            {
              v89 = v63((__int64)a5, v107, v122, v100);
              v65 = v86;
            }
            v7 = v107;
            v66 = sub_2FE98B0((__int64)a5, v107, (unsigned int)v89, v65);
            goto LABEL_72;
          }
          v132 = 0;
          LOWORD(v128) = 0;
          LOWORD(v131) = 0;
          sub_2FE8D10(
            (__int64)a5,
            v107,
            (unsigned int)v122,
            v100,
            (__int64 *)&v131,
            (unsigned int *)&v130,
            (unsigned __int16 *)&v128);
LABEL_107:
          v66 = v128;
          v7 = v87;
        }
LABEL_72:
        if ( v66 <= 1u || (unsigned __int16)(v66 - 504) <= 7u )
          goto LABEL_115;
        v67 = 16LL * (v66 - 1);
        v68 = byte_444C4A0[v67 + 8];
        v69 = *(_QWORD *)&byte_444C4A0[v67];
        LOBYTE(v132) = v68;
        v131 = v69;
        v70 = sub_CA1930(&v131);
        v16 = (v57 + v70 - 1) % v70;
        v102 = (v57 + v70 - 1) / v70;
        v10 = *(_QWORD **)a5;
        v107 = *a2;
LABEL_17:
        v17 = (__int64 (__fastcall *)(__int64, __int64, __int64, __int64, unsigned __int64))v10[93];
        if ( v17 != sub_2FE9BB0 )
          goto LABEL_45;
        goto LABEL_18;
      }
      v44 = a5;
      v7 = v107;
      v45 = ((__int64 (__fastcall *)(_WORD *, __int64, __int64, __int64, _QWORD))v15)(
              a5,
              v107,
              v113.m128i_i64[0],
              v113.m128i_i64[1],
              v112);
LABEL_44:
      v102 = v45;
      v10 = (_QWORD *)*v44;
      v107 = *a2;
      v17 = (__int64 (__fastcall *)(__int64, __int64, __int64, __int64, unsigned __int64))v10[93];
      if ( v17 != sub_2FE9BB0 )
      {
LABEL_45:
        v24 = v17((__int64)a5, v107, a1, v113.m128i_u32[0], v113.m128i_u64[1]);
        goto LABEL_25;
      }
LABEL_18:
      v18 = v113.m128i_u16[0];
      v19 = v113.m128i_u64[1];
      LOWORD(v124) = v113.m128i_i16[0];
      v125 = v113.m128i_i64[1];
      if ( v113.m128i_i16[0] )
        goto LABEL_40;
      if ( (unsigned __int8)sub_30070B0(&v124, v7, v16) )
      {
        v132 = 0;
        LOWORD(v131) = 0;
        LOWORD(v128) = 0;
        sub_2FE8D10(
          (__int64)a5,
          v107,
          (unsigned int)v124,
          v19,
          (__int64 *)&v131,
          (unsigned int *)&v130,
          (unsigned __int16 *)&v128);
LABEL_92:
        v24 = (unsigned __int16)v128;
        goto LABEL_25;
      }
      if ( !(unsigned __int8)sub_3007070(&v124) )
        goto LABEL_116;
      v20 = (__int64 (__fastcall *)(__int64, __int64, unsigned int, __int64))v10[74];
      if ( v20 == sub_2D56A50 )
      {
        sub_2FE6CC0((__int64)&v131, (__int64)a5, v107, v124, v125);
        v22 = v93;
        LOWORD(v22) = v132;
        v23 = v133;
        v93 = v22;
      }
      else
      {
        v93 = v20((__int64)a5, v107, v124, v19);
        v23 = v21;
      }
      v127 = v23;
      v126 = v93;
      if ( !(_WORD)v93 )
      {
        if ( (unsigned __int8)sub_30070B0(&v126, v93, v21) )
        {
          v132 = 0;
          LOWORD(v131) = 0;
          LOWORD(v128) = 0;
          sub_2FE8D10(
            (__int64)a5,
            v107,
            (unsigned int)v126,
            v23,
            (__int64 *)&v131,
            (unsigned int *)&v130,
            (unsigned __int16 *)&v128);
          goto LABEL_92;
        }
        if ( !(unsigned __int8)sub_3007070(&v126) )
          goto LABEL_116;
        v71 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)a5 + 592LL);
        if ( v71 == sub_2D56A50 )
        {
          sub_2FE6CC0((__int64)&v131, (__int64)a5, v107, v126, v127);
          v73 = v92;
          LOWORD(v73) = v132;
          v74 = v133;
          v92 = v73;
        }
        else
        {
          v92 = v71((__int64)a5, v107, v126, v23);
          v74 = v72;
        }
        v129 = v74;
        v18 = (unsigned __int16)v92;
        v128 = v92;
        if ( !(_WORD)v92 )
        {
          if ( (unsigned __int8)sub_30070B0(&v128, v92, v72) )
          {
            LOWORD(v131) = 0;
            v132 = 0;
            LOWORD(v122) = 0;
            sub_2FE8D10(
              (__int64)a5,
              v107,
              (unsigned int)v128,
              v74,
              (__int64 *)&v131,
              (unsigned int *)&v130,
              (unsigned __int16 *)&v122);
            v24 = (unsigned __int16)v122;
            goto LABEL_25;
          }
          if ( !(unsigned __int8)sub_3007070(&v128) )
            goto LABEL_116;
          v75 = a5;
          v76 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)a5 + 592LL);
          if ( v76 == sub_2D56A50 )
          {
            sub_2FE6CC0((__int64)&v131, (__int64)a5, v107, v128, v129);
            v78 = v90;
            LOWORD(v78) = v132;
            v79 = v133;
            v90 = v78;
          }
          else
          {
            v75 = (_WORD *)v107;
            v90 = v76((__int64)a5, v107, v128, v74);
            v79 = v77;
          }
          v130.m128i_i64[1] = v79;
          v18 = (unsigned __int16)v90;
          v130.m128i_i64[0] = v90;
          if ( !(_WORD)v90 )
          {
            if ( (unsigned __int8)sub_30070B0(&v130, v75, v77) )
            {
              v119.m128i_i16[0] = 0;
              LOWORD(v131) = 0;
              v132 = 0;
              sub_2FE8D10(
                (__int64)a5,
                v107,
                v130.m128i_u32[0],
                v79,
                (__int64 *)&v131,
                (unsigned int *)&v122,
                (unsigned __int16 *)&v119);
              v24 = v119.m128i_u16[0];
            }
            else
            {
              if ( !(unsigned __int8)sub_3007070(&v130) )
LABEL_116:
                BUG();
              v80 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)a5 + 592LL);
              if ( v80 == sub_2D56A50 )
              {
                sub_2FE6CC0((__int64)&v131, (__int64)a5, v107, v130.m128i_i64[0], v130.m128i_i64[1]);
                v81 = v88;
                LOWORD(v81) = v132;
                v82 = v133;
                v88 = v81;
              }
              else
              {
                v88 = v80((__int64)a5, v107, v130.m128i_u32[0], v79);
                v82 = v85;
              }
              v24 = sub_2FE98B0((__int64)a5, v107, (unsigned int)v88, v82);
            }
            goto LABEL_25;
          }
        }
LABEL_40:
        v24 = (unsigned __int16)a5[v18 + 1426];
        goto LABEL_25;
      }
      v24 = (unsigned __int16)a5[(unsigned __int16)v93 + 1426];
LABEL_25:
      v108 = v24;
      v25 = (unsigned __int8)sub_A74710(v111, 0, 15);
      v26 = sub_A74710(v111, 0, 54);
      v27 = v108;
      v28 = v26;
      v29 = 0;
      if ( !(_BYTE)v28 )
      {
        v29 = sub_A74710(v111, 0, 79);
        v27 = v108;
        v28 = 0;
      }
      if ( v102 )
      {
        v30 = *(unsigned int *)(a4 + 8);
        v31 = v113.m128i_u16[0];
        v32 = v113.m128i_i64[1];
        v33 = 0;
        v34 = ((8 * v25) | v29 | (unsigned __int64)(2 * v28)) & 0x7FFFFFFFFLL;
        v35 = a4;
        v36 = v102;
        do
        {
          v37 = *(unsigned int *)(v35 + 12);
          v38 = v30 + 1;
          v132 = 0;
          v39 = (const __m128i *)&v131;
          v131 = v34 | v131 & 0xFFFFFFF800000000LL;
          v40 = *(_QWORD *)v35;
          v136 = 1;
          v137 = 0;
          LOWORD(v133) = v27;
          v134 = v31;
          v135 = v32;
          if ( v30 + 1 > v37 )
          {
            v95 = v34;
            v42 = (const void *)(v35 + 16);
            v97 = v31;
            v98 = v27;
            v103 = v32;
            if ( v40 > (unsigned __int64)&v131 || (unsigned __int64)&v131 >= v40 + 56 * v30 )
            {
              v39 = (const __m128i *)&v131;
              sub_C8D5F0(v35, v42, v38, 0x38u, v27, v32);
              v34 = v95;
              v31 = v97;
              v27 = v98;
              v40 = *(_QWORD *)v35;
              v30 = *(unsigned int *)(v35 + 8);
              v32 = v103;
            }
            else
            {
              v43 = (char *)&v131 - v40;
              sub_C8D5F0(v35, v42, v38, 0x38u, v27, v32);
              v32 = v103;
              v27 = v98;
              v31 = v97;
              v40 = *(_QWORD *)v35;
              v30 = *(unsigned int *)(v35 + 8);
              v34 = v95;
              v39 = (const __m128i *)&v43[*(_QWORD *)v35];
            }
          }
          ++v33;
          v41 = (__m128i *)(v40 + 56 * v30);
          *v41 = _mm_loadu_si128(v39);
          v41[1] = _mm_loadu_si128(v39 + 1);
          v41[2] = _mm_loadu_si128(v39 + 2);
          v41[3].m128i_i64[0] = v39[3].m128i_i64[0];
          v30 = (unsigned int)(*(_DWORD *)(v35 + 8) + 1);
          *(_DWORD *)(v35 + 8) = v30;
        }
        while ( v36 != v33 );
        a4 = v35;
      }
      v109 += 16;
      if ( v109 == v94 )
        goto LABEL_33;
    }
    if ( (unsigned __int16)(v113.m128i_i16[0] - 2) <= 7u
      || (unsigned __int16)(v113.m128i_i16[0] - 17) <= 0x6Cu
      || (unsigned __int16)(v113.m128i_i16[0] - 176) <= 0x1Fu )
    {
LABEL_8:
      v11 = (__int64 (__fastcall *)(__int64, __int64, __int64, __int64))v10[295];
      if ( v11 == sub_2FE53A0 )
      {
        v12 = a5[1433];
        v130 = _mm_loadu_si128(&v113);
        if ( v9 == v12 )
        {
          v13 = v130.m128i_i64[1];
          if ( v9 || !v130.m128i_i64[1] )
            goto LABEL_11;
          v132 = 0;
          LOWORD(v131) = 0;
        }
        else
        {
          LOWORD(v131) = v12;
          v132 = 0;
          if ( v12 )
          {
            if ( v12 == 1 || (unsigned __int16)(v12 - 504) <= 7u )
LABEL_115:
              BUG();
            v48 = *(_QWORD *)&byte_444C4A0[16 * v12 - 16];
            v7 = (unsigned __int8)byte_444C4A0[16 * v12 - 8];
            goto LABEL_52;
          }
        }
        v104 = v12;
        v47 = sub_3007260(&v131);
        v12 = v104;
        v116 = v47;
        v48 = v47;
        v117 = v49;
        v7 = (unsigned __int8)v49;
LABEL_52:
        if ( v9 )
        {
          if ( v9 == 1 || (unsigned __int16)(v9 - 504) <= 7u )
            goto LABEL_115;
          v53 = *(_QWORD *)&byte_444C4A0[16 * v9 - 16];
          v54 = byte_444C4A0[16 * v9 - 8];
        }
        else
        {
          v105 = v12;
          v50 = sub_3007260(&v130);
          v7 = (unsigned __int8)v7;
          v12 = v105;
          v52 = v51;
          v114 = v50;
          v53 = v50;
          v115 = v52;
          v54 = v52;
        }
        if ( (!v54 || (_BYTE)v7) && v53 < v48 )
        {
          v13 = 0;
          goto LABEL_12;
        }
        v13 = v130.m128i_i64[1];
LABEL_11:
        v12 = v9;
LABEL_12:
        v113.m128i_i16[0] = v12;
        v113.m128i_i64[1] = v13;
        goto LABEL_13;
      }
      v7 = v107;
      v83 = ((__int64 (__fastcall *)(_WORD *, __int64, _QWORD, __int64, _QWORD))v11)(
              a5,
              v107,
              v113.m128i_u32[0],
              v113.m128i_i64[1],
              v8);
      v10 = *(_QWORD **)a5;
      v113.m128i_i32[0] = v83;
      v113.m128i_i64[1] = v84;
      v107 = *a2;
    }
LABEL_13:
    v14 = (__int64 (__fastcall *)(__int64, __int64, __int64, __int64, unsigned __int64))v10[94];
    if ( v14 != sub_2FEA530 )
      goto LABEL_43;
    goto LABEL_14;
  }
LABEL_33:
  if ( v138 != v140 )
    _libc_free((unsigned __int64)v138);
}
