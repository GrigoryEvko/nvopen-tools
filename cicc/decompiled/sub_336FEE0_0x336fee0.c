// Function: sub_336FEE0
// Address: 0x336fee0
//
__int64 *__fastcall sub_336FEE0(__int64 a1, __int64 a2, __int64 a3, int a4, int a5, int a6, __int64 a7)
{
  __int64 v9; // rbx
  __int64 v10; // rax
  _QWORD *v11; // rdi
  __int64 v12; // r9
  __int64 *v13; // rdx
  __int64 *result; // rax
  __int64 v15; // r14
  __int64 v16; // r11
  unsigned __int64 v17; // r15
  __int64 v18; // rax
  __int64 (__fastcall *v19)(__int64, __int64, __int64, __int64, unsigned __int64); // r9
  __int64 (__fastcall *v20)(__int64, __int64, __int64, unsigned __int64); // rax
  __int64 v21; // r8
  unsigned int v22; // eax
  unsigned int v23; // r13d
  char v24; // al
  __int64 v25; // rdx
  __int64 (__fastcall *v26)(__int64, __int64, __int64, __int64, unsigned __int64); // rax
  __int64 v27; // r8
  __int64 v28; // rax
  unsigned int v29; // ebx
  int v30; // r15d
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 v33; // rax
  unsigned __int64 v34; // rcx
  __int64 v35; // rax
  __int64 (__fastcall *v36)(__int64, __int64, unsigned int, __int64); // rax
  __int64 v37; // rax
  unsigned __int64 v38; // r15
  __int64 (__fastcall *v39)(__int64, __int64, unsigned int, __int64); // rax
  __int64 v40; // rax
  unsigned __int64 v41; // r15
  __int64 (__fastcall *v42)(__int64, __int64, unsigned int, __int64); // rax
  __int64 v43; // rax
  unsigned __int64 v44; // rcx
  __int64 (__fastcall *v45)(__int64, __int64, unsigned int, __int64); // rax
  __int64 v46; // rax
  unsigned __int64 v47; // r15
  __int64 (__fastcall *v48)(__int64, __int64, unsigned int, __int64); // rax
  __int64 v49; // rax
  unsigned __int64 v50; // r15
  __int64 (__fastcall *v51)(__int64, __int64, unsigned int, __int64); // rax
  __int64 v52; // rax
  unsigned __int64 v53; // r15
  __int64 (__fastcall *v54)(__int64, __int64, unsigned int, __int64); // rax
  __int64 v55; // rax
  unsigned __int64 v56; // rcx
  unsigned int v57; // eax
  __int64 v58; // rdx
  __int64 v59; // r11
  __int64 v60; // rax
  __int64 (__fastcall *v61)(__int64, __int64, unsigned int, __int64); // rax
  __int64 v62; // rax
  __int64 v63; // rcx
  __int64 (__fastcall *v64)(__int64, __int64, unsigned int, __int64); // rax
  __int64 v65; // rax
  unsigned __int64 v66; // rcx
  __int64 v67; // r11
  unsigned __int16 v68; // ax
  unsigned __int16 v69; // dx
  __int64 v70; // rax
  char v71; // cl
  __int64 v72; // rax
  unsigned int v73; // eax
  unsigned int v74; // eax
  unsigned __int64 v75; // rdx
  unsigned __int64 v76; // rdx
  unsigned __int64 v77; // rdx
  __int64 v78; // rax
  __int64 v79; // rdx
  unsigned __int64 v80; // rdx
  unsigned __int64 v81; // rdx
  unsigned __int64 v82; // rdx
  unsigned __int64 v83; // rdx
  __int64 v84; // rax
  unsigned __int64 v85; // rdx
  __int64 v86; // [rsp-10h] [rbp-1D0h]
  __int64 v87; // [rsp+0h] [rbp-1C0h]
  __int64 v88; // [rsp+8h] [rbp-1B8h]
  __int64 v89; // [rsp+10h] [rbp-1B0h]
  __int64 v90; // [rsp+18h] [rbp-1A8h]
  __int64 v91; // [rsp+20h] [rbp-1A0h]
  __int64 v92; // [rsp+20h] [rbp-1A0h]
  __int64 v93; // [rsp+28h] [rbp-198h]
  unsigned __int64 v94; // [rsp+28h] [rbp-198h]
  __int64 v95; // [rsp+38h] [rbp-188h]
  __int64 v96; // [rsp+40h] [rbp-180h]
  __int64 v97; // [rsp+48h] [rbp-178h]
  const void *v98; // [rsp+58h] [rbp-168h]
  __int64 v99; // [rsp+60h] [rbp-160h]
  __int64 v100; // [rsp+78h] [rbp-148h]
  const void *v101; // [rsp+80h] [rbp-140h]
  __int64 *v102; // [rsp+88h] [rbp-138h]
  __int64 v103; // [rsp+90h] [rbp-130h]
  int v104; // [rsp+90h] [rbp-130h]
  __int64 v105; // [rsp+A0h] [rbp-120h]
  __int64 v106; // [rsp+A0h] [rbp-120h]
  unsigned int v107; // [rsp+A0h] [rbp-120h]
  __int64 v108; // [rsp+A0h] [rbp-120h]
  __int64 v109; // [rsp+A0h] [rbp-120h]
  __int64 v110; // [rsp+A0h] [rbp-120h]
  __int64 v111; // [rsp+A0h] [rbp-120h]
  __int64 *v112; // [rsp+A8h] [rbp-118h]
  unsigned int v113; // [rsp+BCh] [rbp-104h]
  __int64 v114; // [rsp+C0h] [rbp-100h] BYREF
  unsigned __int64 v115; // [rsp+C8h] [rbp-F8h]
  __int64 v116; // [rsp+D0h] [rbp-F0h] BYREF
  unsigned __int64 v117; // [rsp+D8h] [rbp-E8h]
  __int64 v118; // [rsp+E0h] [rbp-E0h]
  __int64 v119; // [rsp+E8h] [rbp-D8h]
  __int64 v120; // [rsp+F0h] [rbp-D0h] BYREF
  __int64 v121; // [rsp+F8h] [rbp-C8h]
  __int64 v122; // [rsp+100h] [rbp-C0h] BYREF
  unsigned __int64 v123; // [rsp+108h] [rbp-B8h]
  __int64 v124; // [rsp+110h] [rbp-B0h] BYREF
  unsigned __int64 v125; // [rsp+118h] [rbp-A8h]
  __int64 v126; // [rsp+120h] [rbp-A0h] BYREF
  unsigned __int64 v127; // [rsp+128h] [rbp-98h]
  __int64 v128; // [rsp+130h] [rbp-90h] BYREF
  unsigned __int64 v129; // [rsp+138h] [rbp-88h]
  __int64 v130; // [rsp+140h] [rbp-80h] BYREF
  unsigned __int64 v131; // [rsp+148h] [rbp-78h]
  __int64 v132; // [rsp+150h] [rbp-70h] BYREF
  unsigned __int64 v133; // [rsp+158h] [rbp-68h]
  __int64 v134; // [rsp+160h] [rbp-60h] BYREF
  unsigned __int64 v135; // [rsp+168h] [rbp-58h]
  __int128 v136; // [rsp+170h] [rbp-50h] BYREF
  unsigned __int64 v137; // [rsp+180h] [rbp-40h]

  v9 = a3;
  *(_QWORD *)&v136 = 0;
  BYTE8(v136) = 0;
  v10 = a1 + 16;
  v11 = (_QWORD *)(a1 + 104);
  *(v11 - 13) = v10;
  *(v11 - 12) = 0x400000000LL;
  *(_QWORD *)(a1 + 80) = v11;
  *(_QWORD *)(a1 + 112) = a1 + 128;
  v101 = (const void *)(a1 + 128);
  *(_QWORD *)(a1 + 120) = 0x400000000LL;
  *(_QWORD *)(a1 + 144) = a1 + 160;
  *(_QWORD *)(a1 + 152) = 0x400000000LL;
  *(_QWORD *)(a1 + 88) = 0;
  *(_QWORD *)(a1 + 96) = 4;
  *(_BYTE *)(a1 + 180) = 0;
  v98 = (const void *)(a1 + 160);
  sub_34B8C80(a3, a4, a6, a1, 0, 0, v136);
  v13 = *(__int64 **)a1;
  *(_QWORD *)(a1 + 176) = a7;
  result = &v13[2 * *(unsigned int *)(a1 + 8)];
  v102 = result;
  if ( v13 != result )
  {
    v112 = v13;
    v15 = a1;
    while ( 1 )
    {
      v16 = *v112;
      v17 = v112[1];
      v18 = *(_QWORD *)v9;
      if ( !*(_BYTE *)(v15 + 180) )
        break;
      v19 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, unsigned __int64))(v18 + 752);
      if ( v19 == sub_2FEA530 )
      {
        v20 = *(__int64 (__fastcall **)(__int64, __int64, __int64, unsigned __int64))(v18 + 736);
        BYTE2(v136) = 0;
        v21 = v136;
LABEL_6:
        v105 = v16;
        v22 = ((__int64 (__fastcall *)(__int64, __int64, _QWORD, unsigned __int64, __int64))v20)(
                v9,
                a2,
                (unsigned int)v16,
                v17,
                v21);
        v16 = v105;
        v23 = v22;
        v24 = *(_BYTE *)(v15 + 180);
        goto LABEL_7;
      }
      v111 = *v112;
      v74 = v19(v9, a2, (unsigned int)a7, (unsigned int)v16, v17);
      v16 = v111;
      v23 = v74;
      v24 = *(_BYTE *)(v15 + 180);
LABEL_7:
      if ( v24 )
      {
        v25 = *(_QWORD *)v9;
        v26 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, unsigned __int64))(*(_QWORD *)v9 + 744LL);
        if ( v26 != sub_2FE9BB0 )
        {
          v27 = (unsigned int)v26(v9, a2, (unsigned int)a7, (unsigned int)v16, v17);
          goto LABEL_11;
        }
        LOWORD(v122) = v16;
        v123 = v17;
        if ( (_WORD)v16 )
        {
          v27 = *(unsigned __int16 *)(v9 + 2LL * (unsigned __int16)v16 + 2852);
          goto LABEL_11;
        }
        v103 = v25;
        if ( sub_30070B0((__int64)&v122) )
        {
          *((_QWORD *)&v136 + 1) = 0;
          LOWORD(v136) = 0;
          LOWORD(v132) = 0;
          sub_2FE8D10(
            v9,
            a2,
            (unsigned int)v122,
            v17,
            (__int64 *)&v136,
            (unsigned int *)&v134,
            (unsigned __int16 *)&v132);
        }
        else
        {
          if ( !sub_3007070((__int64)&v122) )
            goto LABEL_104;
          v45 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(v103 + 592);
          if ( v45 == sub_2D56A50 )
          {
            sub_2FE6CC0((__int64)&v136, v9, a2, v122, v123);
            v46 = v100;
            LOWORD(v46) = WORD4(v136);
            v47 = v137;
            v100 = v46;
          }
          else
          {
            v100 = v45(v9, a2, v122, v17);
            v47 = v75;
          }
          v125 = v47;
          v35 = (unsigned __int16)v100;
          v124 = v100;
          if ( (_WORD)v100 )
            goto LABEL_42;
          if ( sub_30070B0((__int64)&v124) )
          {
            *((_QWORD *)&v136 + 1) = 0;
            LOWORD(v136) = 0;
            LOWORD(v132) = 0;
            sub_2FE8D10(
              v9,
              a2,
              (unsigned int)v124,
              v47,
              (__int64 *)&v136,
              (unsigned int *)&v134,
              (unsigned __int16 *)&v132);
            goto LABEL_85;
          }
          if ( !sub_3007070((__int64)&v124) )
            goto LABEL_104;
          v48 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v9 + 592LL);
          if ( v48 == sub_2D56A50 )
          {
            sub_2FE6CC0((__int64)&v136, v9, a2, v124, v125);
            v49 = v97;
            LOWORD(v49) = WORD4(v136);
            v50 = v137;
            v97 = v49;
          }
          else
          {
            v97 = v48(v9, a2, v124, v47);
            v50 = v77;
          }
          v127 = v50;
          v35 = (unsigned __int16)v97;
          v126 = v97;
          if ( (_WORD)v97 )
            goto LABEL_42;
          if ( sub_30070B0((__int64)&v126) )
          {
            LOWORD(v136) = 0;
            LOWORD(v132) = 0;
            *((_QWORD *)&v136 + 1) = 0;
            sub_2FE8D10(
              v9,
              a2,
              (unsigned int)v126,
              v50,
              (__int64 *)&v136,
              (unsigned int *)&v134,
              (unsigned __int16 *)&v132);
            goto LABEL_85;
          }
          if ( !sub_3007070((__int64)&v126) )
            goto LABEL_104;
          v51 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v9 + 592LL);
          if ( v51 == sub_2D56A50 )
          {
            sub_2FE6CC0((__int64)&v136, v9, a2, v126, v127);
            v52 = v90;
            LOWORD(v52) = WORD4(v136);
            v53 = v137;
            v90 = v52;
          }
          else
          {
            v90 = v51(v9, a2, v126, v50);
            v53 = v82;
          }
          v129 = v53;
          v35 = (unsigned __int16)v90;
          v128 = v90;
          if ( (_WORD)v90 )
          {
LABEL_42:
            v27 = *(unsigned __int16 *)(v9 + 2 * v35 + 2852);
            goto LABEL_11;
          }
          if ( !sub_30070B0((__int64)&v128) )
          {
            if ( !sub_3007070((__int64)&v128) )
              goto LABEL_104;
            v54 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v9 + 592LL);
            if ( v54 == sub_2D56A50 )
            {
              sub_2FE6CC0((__int64)&v136, v9, a2, v128, v129);
              v55 = v88;
              LOWORD(v55) = WORD4(v136);
              v56 = v137;
              v88 = v55;
            }
            else
            {
              v88 = v54(v9, a2, v128, v53);
              v56 = v81;
            }
            v27 = (unsigned int)sub_2FE98B0(v9, a2, (unsigned int)v88, v56);
LABEL_11:
            if ( v23 )
              goto LABEL_12;
            goto LABEL_40;
          }
          LOWORD(v136) = 0;
          LOWORD(v132) = 0;
          *((_QWORD *)&v136 + 1) = 0;
          sub_2FE8D10(
            v9,
            a2,
            (unsigned int)v128,
            v53,
            (__int64 *)&v136,
            (unsigned int *)&v134,
            (unsigned __int16 *)&v132);
        }
LABEL_85:
        v27 = (unsigned __int16)v132;
        goto LABEL_11;
      }
LABEL_24:
      v35 = (unsigned __int16)v16;
      v130 = v16;
      v131 = v17;
      if ( (_WORD)v16 )
        goto LABEL_42;
      if ( sub_30070B0((__int64)&v130) )
      {
        LOWORD(v132) = 0;
        LOWORD(v136) = 0;
        *((_QWORD *)&v136 + 1) = 0;
        sub_2FE8D10(v9, a2, (unsigned int)v130, v17, (__int64 *)&v136, (unsigned int *)&v134, (unsigned __int16 *)&v132);
        goto LABEL_85;
      }
      if ( !sub_3007070((__int64)&v130) )
        goto LABEL_104;
      v36 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v9 + 592LL);
      if ( v36 == sub_2D56A50 )
      {
        sub_2FE6CC0((__int64)&v136, v9, a2, v130, v131);
        v37 = v99;
        LOWORD(v37) = WORD4(v136);
        v38 = v137;
        v99 = v37;
      }
      else
      {
        v99 = v36(v9, a2, v130, v17);
        v38 = v76;
      }
      v133 = v38;
      v35 = (unsigned __int16)v99;
      v132 = v99;
      if ( (_WORD)v99 )
        goto LABEL_42;
      if ( sub_30070B0((__int64)&v132) )
      {
        LOWORD(v136) = 0;
        LOWORD(v128) = 0;
        *((_QWORD *)&v136 + 1) = 0;
        sub_2FE8D10(v9, a2, (unsigned int)v132, v38, (__int64 *)&v136, (unsigned int *)&v134, (unsigned __int16 *)&v128);
        v12 = v86;
        v27 = (unsigned __int16)v128;
        goto LABEL_11;
      }
      if ( !sub_3007070((__int64)&v132) )
        goto LABEL_104;
      v39 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v9 + 592LL);
      if ( v39 == sub_2D56A50 )
      {
        sub_2FE6CC0((__int64)&v136, v9, a2, v132, v133);
        v40 = v96;
        LOWORD(v40) = WORD4(v136);
        v41 = v137;
        v96 = v40;
      }
      else
      {
        v96 = v39(v9, a2, v132, v38);
        v41 = v80;
      }
      v135 = v41;
      v35 = (unsigned __int16)v96;
      v134 = v96;
      if ( (_WORD)v96 )
        goto LABEL_42;
      if ( sub_30070B0((__int64)&v134) )
      {
        LOWORD(v136) = 0;
        LOWORD(v126) = 0;
        *((_QWORD *)&v136 + 1) = 0;
        sub_2FE8D10(v9, a2, (unsigned int)v134, v41, (__int64 *)&v136, (unsigned int *)&v128, (unsigned __int16 *)&v126);
        v27 = (unsigned __int16)v126;
        goto LABEL_11;
      }
      if ( !sub_3007070((__int64)&v134) )
        goto LABEL_104;
      v42 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v9 + 592LL);
      if ( v42 == sub_2D56A50 )
      {
        sub_2FE6CC0((__int64)&v136, v9, a2, v134, v135);
        v43 = v89;
        LOWORD(v43) = WORD4(v136);
        v44 = v137;
        v89 = v43;
      }
      else
      {
        v89 = v42(v9, a2, v134, v41);
        v44 = v83;
      }
      v27 = (unsigned int)sub_2FE98B0(v9, a2, (unsigned int)v89, v44);
      if ( v23 )
      {
LABEL_12:
        v28 = *(unsigned int *)(v15 + 120);
        v106 = v9;
        v29 = v27;
        v30 = v23 + a5;
        do
        {
          if ( v28 + 1 > (unsigned __int64)*(unsigned int *)(v15 + 124) )
          {
            sub_C8D5F0(v15 + 112, v101, v28 + 1, 4u, v27, v12);
            v28 = *(unsigned int *)(v15 + 120);
          }
          *(_DWORD *)(*(_QWORD *)(v15 + 112) + 4 * v28) = a5++;
          v28 = (unsigned int)(*(_DWORD *)(v15 + 120) + 1);
          *(_DWORD *)(v15 + 120) = v28;
        }
        while ( a5 != v30 );
        v31 = *(_QWORD *)(v15 + 88);
        v27 = v29;
        v9 = v106;
        v32 = v31 + 1;
        if ( (unsigned __int64)(v31 + 1) <= *(_QWORD *)(v15 + 96) )
          goto LABEL_17;
LABEL_41:
        v107 = v27;
        sub_C8D290(v15 + 80, v11, v32, 2u, v27, v12);
        v31 = *(_QWORD *)(v15 + 88);
        v27 = v107;
        goto LABEL_17;
      }
LABEL_40:
      v31 = *(_QWORD *)(v15 + 88);
      v30 = a5;
      v32 = v31 + 1;
      if ( (unsigned __int64)(v31 + 1) > *(_QWORD *)(v15 + 96) )
        goto LABEL_41;
LABEL_17:
      *(_WORD *)(*(_QWORD *)(v15 + 80) + 2 * v31) = v27;
      v33 = *(unsigned int *)(v15 + 152);
      v34 = *(unsigned int *)(v15 + 156);
      ++*(_QWORD *)(v15 + 88);
      if ( v33 + 1 > v34 )
      {
        sub_C8D5F0(v15 + 144, v98, v33 + 1, 4u, v27, v12);
        v33 = *(unsigned int *)(v15 + 152);
      }
      v112 += 2;
      a5 = v30;
      *(_DWORD *)(*(_QWORD *)(v15 + 144) + 4 * v33) = v23;
      result = v112;
      ++*(_DWORD *)(v15 + 152);
      if ( v102 == v112 )
        return result;
    }
    v20 = *(__int64 (__fastcall **)(__int64, __int64, __int64, unsigned __int64))(v18 + 736);
    BYTE2(v113) = 0;
    if ( v20 != sub_2FEA1A0 )
    {
      v21 = v113;
      goto LABEL_6;
    }
    v114 = *v112;
    v115 = v17;
    if ( (_WORD)v16 )
    {
      v23 = *(unsigned __int16 *)(v9 + 2LL * (unsigned __int16)v16 + 2304);
      goto LABEL_24;
    }
    v108 = v16;
    if ( sub_30070B0((__int64)&v114) )
    {
      LOWORD(v136) = 0;
      LOWORD(v132) = 0;
      *((_QWORD *)&v136 + 1) = 0;
      v57 = sub_2FE8D10(
              v9,
              a2,
              (unsigned int)v114,
              v17,
              (__int64 *)&v136,
              (unsigned int *)&v134,
              (unsigned __int16 *)&v132);
      v16 = v108;
      v23 = v57;
      v24 = *(_BYTE *)(v15 + 180);
      goto LABEL_7;
    }
    if ( !sub_3007070((__int64)&v114) )
      goto LABEL_104;
    v118 = sub_3007260((__int64)&v114);
    v119 = v58;
    *(_QWORD *)&v136 = v118;
    BYTE8(v136) = v58;
    v104 = sub_CA1930(&v136);
    v59 = v108;
    v60 = (unsigned __int16)v114;
    v116 = v114;
    v117 = v115;
    if ( !(_WORD)v114 )
    {
      v91 = v115;
      v93 = v114;
      if ( sub_30070B0((__int64)&v116) )
      {
        LOWORD(v136) = 0;
        LOWORD(v132) = 0;
        *((_QWORD *)&v136 + 1) = 0;
        sub_2FE8D10(
          v9,
          a2,
          (unsigned int)v116,
          v117,
          (__int64 *)&v136,
          (unsigned int *)&v134,
          (unsigned __int16 *)&v132);
LABEL_93:
        v69 = v132;
        v59 = v108;
LABEL_79:
        if ( v69 <= 1u || (unsigned __int16)(v69 - 504) <= 7u )
          BUG();
        v110 = v59;
        v70 = 16LL * (v69 - 1);
        v71 = byte_444C4A0[v70 + 8];
        v72 = *(_QWORD *)&byte_444C4A0[v70];
        BYTE8(v136) = v71;
        *(_QWORD *)&v136 = v72;
        v73 = sub_CA1930(&v136);
        v16 = v110;
        v23 = (v104 + v73 - 1) / v73;
        v24 = *(_BYTE *)(v15 + 180);
        goto LABEL_7;
      }
      if ( !sub_3007070((__int64)&v116) )
        goto LABEL_104;
      v61 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v9 + 592LL);
      if ( v61 == sub_2D56A50 )
      {
        sub_2FE6CC0((__int64)&v136, v9, a2, v93, v91);
        v62 = v95;
        LOWORD(v62) = WORD4(v136);
        v63 = v137;
        v59 = v108;
        v95 = v62;
      }
      else
      {
        v78 = v61(v9, a2, v116, v117);
        v59 = v108;
        v95 = v78;
        v63 = v79;
      }
      v121 = v63;
      v60 = (unsigned __int16)v95;
      v120 = v95;
      if ( !(_WORD)v95 )
      {
        v92 = v59;
        v94 = v63;
        if ( !sub_30070B0((__int64)&v120) )
        {
          if ( !sub_3007070((__int64)&v120) )
LABEL_104:
            BUG();
          v64 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v9 + 592LL);
          if ( v64 == sub_2D56A50 )
          {
            sub_2FE6CC0((__int64)&v136, v9, a2, v120, v121);
            v65 = v87;
            LOWORD(v65) = WORD4(v136);
            v66 = v137;
            v67 = v92;
            v87 = v65;
          }
          else
          {
            v84 = v64(v9, a2, v120, v94);
            v67 = v92;
            v87 = v84;
            v66 = v85;
          }
          v109 = v67;
          v68 = sub_2FE98B0(v9, a2, (unsigned int)v87, v66);
          v59 = v109;
          v69 = v68;
          goto LABEL_79;
        }
        v108 = v92;
        LOWORD(v136) = 0;
        LOWORD(v132) = 0;
        *((_QWORD *)&v136 + 1) = 0;
        sub_2FE8D10(v9, a2, (unsigned int)v120, v94, (__int64 *)&v136, (unsigned int *)&v134, (unsigned __int16 *)&v132);
        goto LABEL_93;
      }
    }
    v69 = *(_WORD *)(v9 + 2 * v60 + 2852);
    goto LABEL_79;
  }
  return result;
}
