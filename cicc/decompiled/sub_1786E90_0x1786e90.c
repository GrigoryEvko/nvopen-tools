// Function: sub_1786E90
// Address: 0x1786e90
//
unsigned __int8 *__fastcall sub_1786E90(
        __m128i *a1,
        __int64 a2,
        __m128i a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  unsigned __int8 *v11; // r12
  __m128i v13; // xmm1
  __m128i v14; // xmm2
  char v15; // al
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rbx
  __int64 v19; // r13
  __int64 v20; // r15
  _QWORD *v21; // rax
  double v22; // xmm4_8
  double v23; // xmm5_8
  __int64 v25; // rax
  double v26; // xmm4_8
  double v27; // xmm5_8
  __int64 v28; // r15
  __int64 v29; // rdx
  __int64 v30; // rcx
  __int64 v31; // rax
  __int64 *v32; // rsi
  __int64 v33; // rdx
  int v34; // edi
  __int64 v35; // rdx
  __int64 v36; // rcx
  __int64 v37; // rdx
  __int64 v38; // rcx
  __int64 v39; // rax
  __int64 v40; // rax
  _QWORD *v41; // rdx
  double v42; // xmm4_8
  double v43; // xmm5_8
  __int64 *v44; // r9
  __int64 v45; // r10
  __int64 v46; // rax
  __int64 v47; // r10
  __int64 v48; // rsi
  __int64 v49; // rdx
  __int64 v50; // rcx
  __int64 *v51; // r14
  __int64 v52; // rax
  __int64 v53; // rdx
  __int64 v54; // rcx
  __int64 v55; // rdi
  __int64 v56; // rcx
  __int64 v57; // rdx
  __int64 v58; // rsi
  __int64 *v59; // rax
  __int64 *v60; // rsi
  __int64 *v61; // r12
  __int64 v62; // rdx
  __int64 v63; // rcx
  __int64 v64; // rax
  __int64 v65; // rax
  __int64 v66; // rax
  __int64 v67; // rbx
  __int64 v68; // r14
  _QWORD *v69; // rax
  __int64 v70; // rdi
  __int64 v71; // rax
  __int64 *v72; // rsi
  __int64 *v73; // rbx
  __int64 v74; // rdx
  __int64 v75; // rcx
  char v76; // al
  __int64 v77; // rax
  __int64 v78; // rdi
  __int64 v79; // rbx
  __int64 *v80; // rax
  __int64 v81; // rax
  __int64 v82; // rsi
  __int64 v83; // rbx
  __int64 v84; // rdx
  __int64 v85; // rcx
  __int64 v86; // rax
  __int64 v87; // rdi
  __int64 *v88; // rbx
  _QWORD *v89; // rax
  __int64 v90; // rdx
  __int64 v91; // rcx
  __int64 v92; // rbx
  __int64 v93; // rax
  __int64 v94; // rdx
  __int64 v95; // r14
  __int64 v96; // rbx
  _QWORD *v97; // rax
  __int64 v98; // rdi
  __int64 v99; // rax
  __int64 v100; // rax
  __int64 v101; // rdi
  __int64 *v102; // rax
  __int64 v103; // rdi
  __int64 *v104; // rax
  __int64 *v105; // rdi
  __int64 *v106; // [rsp+0h] [rbp-B0h]
  unsigned __int128 v107; // [rsp+8h] [rbp-A8h]
  __int64 v108; // [rsp+10h] [rbp-A0h]
  __int64 v109; // [rsp+18h] [rbp-98h]
  __int64 v110; // [rsp+18h] [rbp-98h]
  __int64 *v111; // [rsp+18h] [rbp-98h]
  __int64 *v112; // [rsp+20h] [rbp-90h] BYREF
  __int64 v113; // [rsp+28h] [rbp-88h] BYREF
  __int64 *v114; // [rsp+30h] [rbp-80h] BYREF
  __int64 **v115; // [rsp+38h] [rbp-78h]
  __int64 **v116; // [rsp+40h] [rbp-70h]
  unsigned __int128 v117; // [rsp+50h] [rbp-60h] BYREF
  __m128i v118; // [rsp+60h] [rbp-50h]
  __int64 v119; // [rsp+70h] [rbp-40h]

  v11 = (unsigned __int8 *)a2;
  v13 = _mm_loadu_si128(a1 + 167);
  v119 = a2;
  v14 = _mm_loadu_si128(a1 + 168);
  v117 = (unsigned __int128)v13;
  v118 = v14;
  v15 = sub_15F24E0(a2);
  v16 = sub_13D1D40(*(unsigned __int8 **)(a2 - 48), *(unsigned __int8 **)(a2 - 24), v15, &v117);
  if ( !v16 )
  {
    if ( (unsigned __int8)sub_170D400(a1, a2, v17, a3, *(double *)v13.m128i_i64, v14) )
      return v11;
    v25 = (__int64)sub_1707490(
                     (__int64)a1,
                     (unsigned __int8 *)a2,
                     *(double *)a3.m128i_i64,
                     *(double *)v13.m128i_i64,
                     *(double *)v14.m128i_i64);
    if ( v25 )
      return (unsigned __int8 *)v25;
    v25 = sub_1713A90(
            a1->m128i_i64,
            (_BYTE *)a2,
            (__m128)a3,
            *(double *)v13.m128i_i64,
            *(double *)v14.m128i_i64,
            a6,
            v26,
            v27,
            a9,
            a10);
    if ( v25 )
      return (unsigned __int8 *)v25;
    v109 = *(_QWORD *)(a2 - 48);
    v28 = *(_QWORD *)(a2 - 24);
    *(_QWORD *)&v117 = 0xBFF0000000000000LL;
    if ( (unsigned __int8)sub_13D6AF0((double *)&v117, v28) )
    {
      v31 = sub_15A1390(*(_QWORD *)v109, v28, v29, v30);
      v118.m128i_i16[0] = 257;
      v32 = (__int64 *)v31;
      v33 = v109;
      v34 = 14;
LABEL_16:
      v11 = (unsigned __int8 *)sub_15FB440(v34, v32, v33, (__int64)&v117, 0);
      sub_15F2530(v11, a2, 1);
      return v11;
    }
    *((_QWORD *)&v107 + 1) = &v112;
    v115 = &v112;
    *(_QWORD *)&v107 = &v114;
    if ( (unsigned __int8)sub_171FB50((__int64)&v114, v109, v29, v30) )
    {
      *((_QWORD *)&v117 + 1) = &v113;
      if ( (unsigned __int8)sub_171FB50((__int64)&v117, v28, (__int64)&v113, v36) )
      {
        v33 = v113;
        v118.m128i_i16[0] = 257;
        goto LABEL_22;
      }
    }
    *((_QWORD *)&v117 + 1) = &v112;
    if ( (unsigned __int8)sub_171FB50((__int64)&v117, v109, v35, v36) && *(_BYTE *)(v28 + 16) <= 0x10u )
    {
      v118.m128i_i16[0] = 257;
      v33 = sub_15A2BF0(
              (__int64 *)v28,
              v109,
              v37,
              257,
              *(double *)a3.m128i_i64,
              *(double *)v13.m128i_i64,
              *(double *)v14.m128i_i64);
LABEL_22:
      v32 = v112;
      v34 = 16;
      goto LABEL_16;
    }
    *((_QWORD *)&v117 + 1) = &v112;
    v39 = *(_QWORD *)(v109 + 8);
    if ( v39 && !*(_QWORD *)(v39 + 8) && (unsigned __int8)sub_171FB50((__int64)&v117, v109, v37, v38) )
    {
      v44 = v112;
      v45 = a1->m128i_i64[1];
      LOWORD(v116) = 257;
      if ( *((_BYTE *)v112 + 16) <= 0x10u && *(_BYTE *)(v28 + 16) <= 0x10u )
      {
        v110 = v45;
        v106 = v112;
        v46 = sub_15A2A30(
                (__int64 *)0x10,
                v112,
                v28,
                0,
                0,
                *(double *)a3.m128i_i64,
                *(double *)v13.m128i_i64,
                *(double *)v14.m128i_i64);
        v47 = v110;
        v111 = (__int64 *)v46;
        v48 = *(_QWORD *)(v47 + 96);
        v108 = v47;
        v51 = (__int64 *)sub_14DBA30(v46, v48, 0);
        if ( v51 )
        {
LABEL_35:
          v52 = sub_15A1390(*v51, v48, v49, v50);
          v33 = (__int64)v51;
          v32 = (__int64 *)v52;
          v34 = 14;
          v118.m128i_i16[0] = 257;
          goto LABEL_16;
        }
        v50 = (__int64)v111;
        v45 = v108;
        v44 = v106;
        if ( v111 )
        {
          v51 = v111;
          goto LABEL_35;
        }
      }
      v48 = (__int64)v44;
      v51 = sub_177F170(v45, v44, v28, (__int64)v11, (__int64 *)&v114);
      goto LABEL_35;
    }
    *((_QWORD *)&v117 + 1) = &v112;
    v40 = *(_QWORD *)(v28 + 8);
    if ( v40 && !*(_QWORD *)(v40 + 8) && (unsigned __int8)sub_171FB50((__int64)&v117, v28, v37, v38) )
    {
      LOWORD(v116) = 257;
      v60 = v112;
      v61 = sub_1780A30(
              a1->m128i_i64[1],
              (__int64)v112,
              v109,
              (__int64)v11,
              (__int64 *)&v114,
              *(double *)a3.m128i_i64,
              *(double *)v13.m128i_i64,
              *(double *)v14.m128i_i64);
      v64 = sub_15A1390(*v61, (__int64)v60, v62, v63);
      v33 = (__int64)v61;
      v118.m128i_i16[0] = 257;
      v32 = (__int64 *)v64;
      v34 = 14;
      goto LABEL_16;
    }
    if ( v28 == v109 )
    {
      LODWORD(v117) = 96;
      DWORD2(v117) = 0;
      v118.m128i_i64[0] = (__int64)&v112;
      if ( (unsigned __int8)sub_173F0F0((__int64)&v117, v28) )
      {
        v118.m128i_i16[0] = 257;
        return sub_1780390(16, v112, (__int64)v112, a2, (__int64)&v117);
      }
    }
    v41 = sub_1707FD0(a1, (unsigned __int8 *)a2, v109, v28);
    if ( v41 )
      return (unsigned __int8 *)sub_170E100(
                                  a1->m128i_i64,
                                  (__int64)v11,
                                  (__int64)v41,
                                  (__m128)a3,
                                  *(double *)v13.m128i_i64,
                                  *(double *)v14.m128i_i64,
                                  a6,
                                  v42,
                                  v43,
                                  a9,
                                  a10);
    if ( !sub_15F24A0(a2) )
      goto LABEL_51;
    if ( *(_BYTE *)(v28 + 16) > 0x10u || !(unsigned __int8)sub_15A0B20(v28, a2, v53, v54) )
    {
LABEL_40:
      if ( sub_15F24B0((__int64)v11) )
      {
        LODWORD(v114) = 196;
        LODWORD(v115) = 0;
        v116 = &v112;
        if ( (unsigned __int8)sub_1781890((__int64)&v114, v109) )
        {
          v118.m128i_i64[0] = (__int64)&v113;
          LODWORD(v117) = 196;
          DWORD2(v117) = 0;
          if ( (unsigned __int8)sub_1781890((__int64)&v117, v28) )
          {
            v103 = a1->m128i_i64[1];
            v118.m128i_i16[0] = 257;
            v104 = sub_1780A30(
                     v103,
                     (__int64)v112,
                     v113,
                     (__int64)v11,
                     (__int64 *)&v117,
                     *(double *)a3.m128i_i64,
                     *(double *)v13.m128i_i64,
                     *(double *)v14.m128i_i64);
            v105 = (__int64 *)a1->m128i_i64[1];
            v118.m128i_i16[0] = 257;
            v114 = v104;
            v41 = sub_15E8450(v105, 196, &v114, 1, (__int64)v11, (int)&v117);
            return (unsigned __int8 *)sub_170E100(
                                        a1->m128i_i64,
                                        (__int64)v11,
                                        (__int64)v41,
                                        (__m128)a3,
                                        *(double *)v13.m128i_i64,
                                        *(double *)v14.m128i_i64,
                                        a6,
                                        v42,
                                        v43,
                                        a9,
                                        a10);
          }
        }
      }
      *((_QWORD *)&v117 + 1) = &v113;
      *(_QWORD *)&v117 = v28;
      if ( (unsigned __int8)sub_1781900((__int64)&v117, v109) && v113 != v28 )
      {
        v118.m128i_i16[0] = 257;
        v55 = a1->m128i_i64[1];
        v56 = (__int64)v11;
        v57 = v28;
        v58 = v28;
LABEL_44:
        v59 = sub_1780A30(
                v55,
                v58,
                v57,
                v56,
                (__int64 *)&v117,
                *(double *)a3.m128i_i64,
                *(double *)v13.m128i_i64,
                *(double *)v14.m128i_i64);
        v118.m128i_i16[0] = 257;
        return sub_1780390(16, v59, v113, (__int64)v11, (__int64)&v117);
      }
      v117 = __PAIR128__(&v113, v109);
      if ( (unsigned __int8)sub_1781900((__int64)&v117, v28) )
      {
        v57 = v109;
        if ( v113 != v109 )
        {
          v118.m128i_i16[0] = 257;
          v55 = a1->m128i_i64[1];
          v56 = (__int64)v11;
          v58 = v109;
          goto LABEL_44;
        }
      }
LABEL_51:
      if ( !sub_15F2480((__int64)v11) )
      {
LABEL_52:
        v65 = *((_QWORD *)v11 - 3);
        if ( *(_BYTE *)(v65 + 16) == 66 )
        {
          v95 = *(_QWORD *)(v65 - 24);
          if ( *(_BYTE *)(v95 + 16) == 61 )
          {
            v96 = **(_QWORD **)(v95 - 24);
            v97 = (_QWORD *)sub_16498A0(v95);
            if ( v96 == sub_1643320(v97) )
            {
              v98 = *(_QWORD *)v11;
              v118.m128i_i16[0] = 257;
              v99 = sub_15A10B0(v98, 0.0);
              return (unsigned __int8 *)sub_14EDD70(*(_QWORD *)(v95 - 24), (_QWORD *)v109, v99, (__int64)&v117, 0, 0);
            }
          }
        }
        v66 = *((_QWORD *)v11 - 6);
        if ( *(_BYTE *)(v66 + 16) == 66 )
        {
          v67 = *(_QWORD *)(v66 - 24);
          if ( *(_BYTE *)(v67 + 16) == 61 )
          {
            v68 = **(_QWORD **)(v67 - 24);
            v69 = (_QWORD *)sub_16498A0(v67);
            if ( v68 == sub_1643320(v69) )
            {
              v70 = *(_QWORD *)v11;
              v118.m128i_i16[0] = 257;
              v71 = sub_15A10B0(v70, 0.0);
              return (unsigned __int8 *)sub_14EDD70(*(_QWORD *)(v67 - 24), (_QWORD *)v28, v71, (__int64)&v117, 0, 0);
            }
          }
        }
        return 0;
      }
      LODWORD(v117) = 124;
      DWORD2(v117) = 0;
      v118.m128i_i64[1] = 0x3FE0000000000000LL;
      v118.m128i_i64[0] = (__int64)&v112;
      if ( (unsigned __int8)sub_1781A00((__int64)&v117, v109) )
      {
        v113 = v28;
        LODWORD(v117) = 124;
        DWORD2(v117) = 0;
        v118.m128i_i64[0] = (__int64)&v112;
        v118.m128i_i64[1] = 0x3FE0000000000000LL;
        if ( !(unsigned __int8)sub_1781A00((__int64)&v117, v28) )
        {
LABEL_84:
          sub_1593B40((_QWORD *)(v109 - 24LL * (*(_DWORD *)(v109 + 20) & 0xFFFFFFF)), (__int64)v112);
          sub_15F2500(v109, (__int64)v11);
          v101 = a1->m128i_i64[1];
          v118.m128i_i16[0] = 257;
          v102 = sub_1780A30(
                   v101,
                   v109,
                   v113,
                   (__int64)v11,
                   (__int64 *)&v117,
                   0.5,
                   *(double *)v13.m128i_i64,
                   *(double *)v14.m128i_i64);
          v118.m128i_i16[0] = 257;
          return sub_1780390(14, v102, v113, (__int64)v11, (__int64)&v117);
        }
      }
      else
      {
        LODWORD(v117) = 124;
        DWORD2(v117) = 0;
        v118.m128i_i64[0] = (__int64)&v112;
        v118.m128i_i64[1] = 0x3FE0000000000000LL;
        if ( !(unsigned __int8)sub_1781A00((__int64)&v117, v28) )
          goto LABEL_52;
      }
      v100 = v109;
      v109 = v28;
      v113 = v100;
      goto LABEL_84;
    }
    v117 = v107;
    if ( (unsigned __int8)sub_17816B0((_QWORD **)&v117, v109) )
    {
      v72 = v114;
      v73 = (__int64 *)sub_15A2C50(
                         (__int64 *)v28,
                         (__int64)v114,
                         *(double *)a3.m128i_i64,
                         *(double *)v13.m128i_i64,
                         *(double *)v14.m128i_i64);
      if ( sub_15A0C20((__int64)v73, (__int64)v72, v74, v75) )
      {
        v118.m128i_i16[0] = 257;
        return sub_1780390(19, v73, (__int64)v112, (__int64)v11, (__int64)&v117);
      }
    }
    v76 = *(_BYTE *)(v109 + 16);
    if ( v76 == 43 )
    {
      if ( !*(_QWORD *)(v109 - 48) )
        goto LABEL_62;
      v112 = *(__int64 **)(v109 - 48);
      v81 = *(_QWORD *)(v109 - 24);
      if ( *(_BYTE *)(v81 + 16) > 0x10u )
        goto LABEL_62;
    }
    else if ( v76 != 5
           || *(_WORD *)(v109 + 18) != 19
           || (v94 = *(_DWORD *)(v109 + 20) & 0xFFFFFFF, !*(_QWORD *)(v109 - 24 * v94))
           || (v112 = *(__int64 **)(v109 - 24 * v94), (v81 = *(_QWORD *)(v109 + 24 * (1 - v94))) == 0) )
    {
LABEL_62:
      *(_QWORD *)&v117 = &v112;
      *((_QWORD *)&v117 + 1) = &v114;
      if ( (unsigned __int8)sub_1781750((_QWORD **)&v117, v109) )
      {
        v77 = sub_15A2C50(
                (__int64 *)v28,
                (__int64)v114,
                *(double *)a3.m128i_i64,
                *(double *)v13.m128i_i64,
                *(double *)v14.m128i_i64);
        v78 = a1->m128i_i64[1];
        v79 = v77;
        v118.m128i_i16[0] = 257;
        v80 = sub_1780A30(
                v78,
                (__int64)v112,
                v28,
                (__int64)v11,
                (__int64 *)&v117,
                *(double *)a3.m128i_i64,
                *(double *)v13.m128i_i64,
                *(double *)v14.m128i_i64);
        v118.m128i_i16[0] = 257;
        return sub_1780390(12, v80, v79, (__int64)v11, (__int64)&v117);
      }
      v117 = v107;
      if ( (unsigned __int8)sub_17817F0((_QWORD **)&v117, v109) )
      {
        v86 = sub_15A2C50(
                (__int64 *)v28,
                (__int64)v114,
                *(double *)a3.m128i_i64,
                *(double *)v13.m128i_i64,
                *(double *)v14.m128i_i64);
        v87 = a1->m128i_i64[1];
        v88 = (__int64 *)v86;
        v118.m128i_i16[0] = 257;
        v89 = sub_1780A30(
                v87,
                (__int64)v112,
                v28,
                (__int64)v11,
                (__int64 *)&v117,
                *(double *)a3.m128i_i64,
                *(double *)v13.m128i_i64,
                *(double *)v14.m128i_i64);
        v118.m128i_i16[0] = 257;
        return sub_1780390(14, v88, (__int64)v89, (__int64)v11, (__int64)&v117);
      }
      goto LABEL_40;
    }
    v114 = (__int64 *)v81;
    v82 = v81;
    v83 = sub_15A2CB0((__int64 *)v28, v81, *(double *)a3.m128i_i64, *(double *)v13.m128i_i64, *(double *)v14.m128i_i64);
    if ( sub_15A0C20(v83, v82, v84, v85) )
    {
      v118.m128i_i16[0] = 257;
      return sub_1780390(16, v112, v83, (__int64)v11, (__int64)&v117);
    }
    v92 = sub_15A2CB0(v114, v28, *(double *)a3.m128i_i64, *(double *)v13.m128i_i64, *(double *)v14.m128i_i64);
    v93 = *(_QWORD *)(v109 + 8);
    if ( v93 && !*(_QWORD *)(v93 + 8) && sub_15A0C20(v92, v28, v90, v91) )
    {
      v118.m128i_i16[0] = 257;
      return sub_1780390(19, v112, v92, (__int64)v11, (__int64)&v117);
    }
    goto LABEL_62;
  }
  v18 = *(_QWORD *)(a2 + 8);
  if ( !v18 )
    return 0;
  v19 = a1->m128i_i64[0];
  v20 = v16;
  do
  {
    v21 = sub_1648700(v18);
    sub_170B990(v19, (__int64)v21);
    v18 = *(_QWORD *)(v18 + 8);
  }
  while ( v18 );
  if ( a2 == v20 )
    v20 = sub_1599EF0(*(__int64 ***)a2);
  sub_164D160(a2, v20, (__m128)a3, *(double *)v13.m128i_i64, *(double *)v14.m128i_i64, a6, v22, v23, a9, a10);
  return v11;
}
