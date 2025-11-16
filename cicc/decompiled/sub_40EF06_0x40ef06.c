// Function: sub_40EF06
// Address: 0x40ef06
//
__int64 __fastcall sub_40EF06(unsigned int *a1, unsigned int a2, unsigned __int64 a3)
{
  int v4; // edx
  int v5; // ecx
  int v6; // r8d
  int v7; // r9d
  int v8; // edx
  int v9; // ecx
  int v10; // r8d
  int v11; // r9d
  __int64 v12; // rcx
  __int64 v13; // rcx
  __int64 v14; // rcx
  __int64 v15; // rcx
  __int64 v16; // rcx
  __int64 v17; // rcx
  __int64 v18; // rcx
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r8
  __int64 v22; // r9
  __int64 v23; // rcx
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // r8
  __int64 v27; // r9
  __int64 v28; // rcx
  __int64 v29; // rcx
  __int64 v30; // r8
  __int64 v31; // r8
  __int64 v32; // r9
  __int64 v33; // rcx
  __int64 v34; // rcx
  int v35; // edx
  int v36; // r8d
  int v37; // r9d
  __int64 v38; // rdx
  int v39; // ecx
  int v40; // r8d
  int v41; // r9d
  unsigned __int64 v42; // rdi
  unsigned __int64 v43; // rdi
  unsigned __int64 v44; // rdi
  __int64 *v45; // rsi
  int v46; // r8d
  int v47; // r9d
  int v48; // ecx
  unsigned int v49; // r15d
  unsigned __int64 v50; // rdx
  unsigned __int64 v51; // rdi
  unsigned __int64 v52; // rdi
  unsigned __int64 v53; // rdi
  int v54; // edx
  int v55; // ecx
  int v56; // r8d
  int v57; // r9d
  unsigned __int64 v58; // r14
  unsigned __int64 v59; // rdi
  unsigned __int64 v60; // rdi
  __int64 v61; // r8
  int v62; // edx
  int v63; // ecx
  int v64; // r8d
  int v65; // r9d
  unsigned __int64 v66; // rax
  unsigned __int64 v67; // rax
  unsigned __int64 v68; // rax
  __int64 result; // rax
  int v70; // edx
  int v71; // ecx
  int v72; // r8d
  int v73; // r9d
  char v74; // [rsp+0h] [rbp-480h]
  unsigned __int64 v75; // [rsp+10h] [rbp-470h]
  bool v76; // [rsp+1Eh] [rbp-462h]
  bool v77; // [rsp+1Fh] [rbp-461h]
  int v79; // [rsp+30h] [rbp-450h] BYREF
  unsigned int v80; // [rsp+34h] [rbp-44Ch] BYREF
  __int64 v81; // [rsp+38h] [rbp-448h] BYREF
  __int64 v82; // [rsp+40h] [rbp-440h] BYREF
  unsigned __int64 v83; // [rsp+48h] [rbp-438h] BYREF
  unsigned __int64 v84; // [rsp+50h] [rbp-430h] BYREF
  unsigned __int64 v85; // [rsp+58h] [rbp-428h] BYREF
  __int64 v86; // [rsp+60h] [rbp-420h] BYREF
  char *v87; // [rsp+68h] [rbp-418h] BYREF
  __int64 v88; // [rsp+70h] [rbp-410h] BYREF
  __int64 v89; // [rsp+78h] [rbp-408h] BYREF
  __int64 v90; // [rsp+80h] [rbp-400h] BYREF
  __int64 v91; // [rsp+88h] [rbp-3F8h] BYREF
  __int64 v92; // [rsp+90h] [rbp-3F0h] BYREF
  __int64 v93; // [rsp+98h] [rbp-3E8h] BYREF
  __int64 v94; // [rsp+A0h] [rbp-3E0h] BYREF
  __int64 v95; // [rsp+A8h] [rbp-3D8h] BYREF
  __int64 v96; // [rsp+B0h] [rbp-3D0h] BYREF
  __int64 v97; // [rsp+B8h] [rbp-3C8h] BYREF
  __int64 v98; // [rsp+C0h] [rbp-3C0h] BYREF
  int v99; // [rsp+C8h] [rbp-3B8h]
  __int64 v100; // [rsp+D0h] [rbp-3B0h]
  __int64 v101; // [rsp+E8h] [rbp-398h] BYREF
  int v102; // [rsp+F0h] [rbp-390h]
  char *v103; // [rsp+F8h] [rbp-388h]
  __int64 v104; // [rsp+110h] [rbp-370h] BYREF
  int v105; // [rsp+118h] [rbp-368h]
  int v106; // [rsp+120h] [rbp-360h]
  __int64 v107; // [rsp+138h] [rbp-348h] BYREF
  int v108; // [rsp+140h] [rbp-340h]
  char *v109; // [rsp+148h] [rbp-338h]
  __int64 v110; // [rsp+160h] [rbp-320h] BYREF
  int v111; // [rsp+168h] [rbp-318h]
  __int64 v112; // [rsp+170h] [rbp-310h]
  __int64 v113; // [rsp+188h] [rbp-2F8h] BYREF
  int v114; // [rsp+190h] [rbp-2F0h]
  char *v115; // [rsp+198h] [rbp-2E8h]
  __int64 v116; // [rsp+1B0h] [rbp-2D0h] BYREF
  int v117; // [rsp+1B8h] [rbp-2C8h]
  unsigned __int64 v118; // [rsp+1C0h] [rbp-2C0h]
  __int64 v119; // [rsp+1D8h] [rbp-2A8h] BYREF
  int v120; // [rsp+1E0h] [rbp-2A0h]
  char *v121; // [rsp+1E8h] [rbp-298h]
  __int64 v122; // [rsp+200h] [rbp-280h] BYREF
  int v123; // [rsp+208h] [rbp-278h]
  unsigned __int64 v124; // [rsp+210h] [rbp-270h]
  __int64 v125; // [rsp+228h] [rbp-258h] BYREF
  int v126; // [rsp+230h] [rbp-250h]
  __int64 v127; // [rsp+238h] [rbp-248h]
  __int64 v128; // [rsp+250h] [rbp-230h] BYREF
  int v129; // [rsp+258h] [rbp-228h]
  unsigned __int64 v130; // [rsp+260h] [rbp-220h]
  __int64 v131; // [rsp+278h] [rbp-208h] BYREF
  int v132; // [rsp+280h] [rbp-200h]
  char *v133; // [rsp+288h] [rbp-1F8h]
  __int64 v134; // [rsp+2A0h] [rbp-1E0h] BYREF
  int v135; // [rsp+2A8h] [rbp-1D8h]
  unsigned __int64 v136; // [rsp+2B0h] [rbp-1D0h]
  __int64 v137; // [rsp+2C8h] [rbp-1B8h] BYREF
  int v138; // [rsp+2D0h] [rbp-1B0h]
  __int64 v139; // [rsp+2D8h] [rbp-1A8h]
  __int64 v140; // [rsp+2F0h] [rbp-190h] BYREF
  int v141; // [rsp+2F8h] [rbp-188h]
  unsigned __int64 v142; // [rsp+300h] [rbp-180h]
  __int64 v143; // [rsp+318h] [rbp-168h] BYREF
  int v144; // [rsp+320h] [rbp-160h]
  const char *v145; // [rsp+328h] [rbp-158h]
  __int64 v146; // [rsp+340h] [rbp-140h] BYREF
  int v147; // [rsp+348h] [rbp-138h]
  unsigned __int64 v148; // [rsp+350h] [rbp-130h]
  __int64 v149; // [rsp+368h] [rbp-118h] BYREF
  int v150; // [rsp+370h] [rbp-110h]
  __int64 v151; // [rsp+378h] [rbp-108h]
  __int64 v152; // [rsp+390h] [rbp-F0h] BYREF
  int v153; // [rsp+398h] [rbp-E8h]
  const char *v154; // [rsp+3A0h] [rbp-E0h]
  __int64 v155; // [rsp+3B8h] [rbp-C8h] BYREF
  int v156; // [rsp+3C0h] [rbp-C0h]
  const char *v157; // [rsp+3C8h] [rbp-B8h]
  _BYTE v158[16]; // [rsp+3E0h] [rbp-A0h] BYREF
  __int64 v159; // [rsp+3F0h] [rbp-90h]
  __int64 v160; // [rsp+400h] [rbp-80h]
  _QWORD v161[13]; // [rsp+418h] [rbp-68h] BYREF

  v161[0] = 4;
  if ( (unsigned int)sub_1308610("arenas.nbins", &v79, v161, 0, 0) )
  {
    sub_130ACF0(
      (unsigned int)"<jemalloc>: Failure in xmallctl(\"%s\", ...)\n",
      (unsigned int)"arenas.nbins",
      v4,
      v5,
      v6,
      v7);
    abort();
  }
  v161[0] = 4;
  if ( (unsigned int)sub_1308610("arenas.nlextents", &v80, v161, 0, 0) )
  {
    sub_130ACF0(
      (unsigned int)"<jemalloc>: Failure in xmallctl(\"%s\", ...)\n",
      (unsigned int)"arenas.nlextents",
      v8,
      v9,
      v10,
      v11);
    abort();
  }
  v81 = 0;
  v82 = 0;
  sub_40E2CD((__int64)&v98, (__int64)&v82);
  v98 = v12;
  v99 = 6;
  sub_40E2CD((__int64)&v101, (__int64)&v81);
  v103 = "size";
  v101 = v13;
  v102 = 9;
  sub_40E2CD((__int64)&v104, (__int64)&v82);
  v104 = v14;
  v105 = 3;
  sub_40E2CD((__int64)&v107, (__int64)&v81);
  v109 = "ind";
  v107 = v15;
  v108 = 9;
  sub_40E2CD((__int64)&v110, (__int64)&v82);
  v110 = v16;
  v111 = 6;
  sub_40E2CD((__int64)&v113, (__int64)&v81);
  v115 = "allocated";
  v113 = v17;
  v114 = 9;
  sub_40E2CD((__int64)&v116, (__int64)&v82);
  v116 = v18;
  v117 = 5;
  sub_40E2CD((__int64)&v119, (__int64)&v81);
  v121 = "nmalloc";
  v119 = v19;
  v120 = 9;
  sub_40E2CD((__int64)&v122, (__int64)&v82);
  v122 = v20;
  v123 = 5;
  sub_40E2CD((__int64)&v125, (__int64)&v81);
  v125 = v21;
  v127 = v22;
  v126 = 9;
  sub_40E2CD((__int64)&v128, (__int64)&v82);
  v128 = v23;
  v129 = 5;
  sub_40E2CD((__int64)&v131, (__int64)&v81);
  v133 = "ndalloc";
  v131 = v24;
  v132 = 9;
  sub_40E2CD((__int64)&v134, (__int64)&v82);
  v134 = v25;
  v135 = 5;
  sub_40E2CD((__int64)&v137, (__int64)&v81);
  v137 = v26;
  v139 = v27;
  v138 = 9;
  sub_40E2CD((__int64)&v140, (__int64)&v82);
  v140 = v28;
  v141 = 5;
  sub_40E2CD((__int64)&v143, (__int64)&v81);
  v145 = "nrequests";
  v143 = v29;
  v144 = 9;
  sub_40E2CD((__int64)&v146, (__int64)&v82);
  v146 = v30;
  v147 = 5;
  sub_40E2CD((__int64)&v149, (__int64)&v81);
  v149 = v31;
  v151 = v32;
  v150 = 9;
  sub_40E2CD((__int64)&v152, (__int64)&v82);
  v152 = v33;
  v153 = 6;
  sub_40E2CD((__int64)&v155, (__int64)&v81);
  v157 = "curlextents";
  HIDWORD(v101) -= 6;
  v155 = v34;
  v156 = 9;
  sub_130F1C0((_DWORD)a1, (unsigned int)"large:", v35, v34, v36, v37);
  if ( *a1 == 2 )
    sub_40ECF5((int)a1, &v81, v38, v39, v40, v41, v74);
  sub_40EDA0((__int64)a1, (__int64)"lextents", v38);
  v95 = 7;
  v42 = __readfsqword(0) - 2664;
  if ( __readfsbyte(0xFFFFF8C8) )
    v42 = sub_1313D30(v42, 0);
  if ( (unsigned int)sub_133D570(v42, v158, 0, "stats.arenas", &v95) )
    goto LABEL_10;
  v96 = 7;
  v159 = a2;
  v43 = __readfsqword(0) - 2664;
  if ( __readfsbyte(0xFFFFF8C8) )
    v43 = sub_1313D30(v43, 0);
  if ( (unsigned int)sub_133D570(v43, v158, 3, "lextents", &v96) )
    goto LABEL_10;
  v97 = 7;
  v44 = __readfsqword(0) - 2664;
  if ( __readfsbyte(0xFFFFF8C8) )
    v44 = sub_1313D30(v44, 0);
  v45 = v161;
  if ( (unsigned int)sub_133D570(v44, v161, 0, "arenas.lextent", &v97) )
  {
LABEL_10:
    sub_130AA40("<jemalloc>: Failure in ctl_mibnametomib()\n");
    abort();
  }
  v48 = 1000000000;
  v49 = 0;
  v77 = 0;
  v50 = a3 % 0x3B9ACA00;
  v75 = a3 / 0x3B9ACA00;
  while ( v80 > v49 )
  {
    v88 = 7;
    v160 = v49;
    v161[2] = v49;
    v51 = __readfsqword(0) - 2664;
    v89 = 8;
    if ( __readfsbyte(0xFFFFF8C8) )
      LODWORD(v51) = sub_1313D30(v51, 0);
    if ( (unsigned int)sub_133D620(
                         v51,
                         (unsigned int)v158,
                         5,
                         (unsigned int)"nmalloc",
                         (unsigned int)&v88,
                         (unsigned int)&v83,
                         (__int64)&v89,
                         0,
                         0) )
      goto LABEL_22;
    v90 = 7;
    v91 = 8;
    v52 = __readfsqword(0) - 2664;
    if ( __readfsbyte(0xFFFFF8C8) )
      LODWORD(v52) = sub_1313D30(v52, 0);
    if ( (unsigned int)sub_133D620(
                         v52,
                         (unsigned int)v158,
                         5,
                         (unsigned int)"ndalloc",
                         (unsigned int)&v90,
                         (unsigned int)&v84,
                         (__int64)&v91,
                         0,
                         0) )
      goto LABEL_22;
    v92 = 7;
    v93 = 8;
    v53 = __readfsqword(0) - 2664;
    if ( __readfsbyte(0xFFFFF8C8) )
      LODWORD(v53) = sub_1313D30(v53, 0);
    if ( (unsigned int)sub_133D620(
                         v53,
                         (unsigned int)v158,
                         5,
                         (unsigned int)"nrequests",
                         (unsigned int)&v92,
                         (unsigned int)&v85,
                         (__int64)&v93,
                         0,
                         0) )
      goto LABEL_22;
    v58 = v85;
    v76 = v85 == 0;
    if ( v85 && v77 )
      sub_130F1C0((_DWORD)a1, (unsigned int)"                     ---\n", v54, v55, v56, v57);
    v94 = 7;
    v95 = 8;
    v59 = __readfsqword(0) - 2664;
    if ( __readfsbyte(0xFFFFF8C8) )
      LODWORD(v59) = sub_1313D30(v59, 0);
    if ( (unsigned int)sub_133D620(
                         v59,
                         (unsigned int)v161,
                         3,
                         (unsigned int)"size",
                         (unsigned int)&v94,
                         (unsigned int)&v86,
                         (__int64)&v95,
                         0,
                         0) )
      goto LABEL_22;
    v96 = 7;
    v97 = 8;
    v60 = __readfsqword(0) - 2664;
    if ( __readfsbyte(0xFFFFF8C8) )
      LODWORD(v60) = sub_1313D30(v60, 0);
    if ( (unsigned int)sub_133D620(
                         v60,
                         (unsigned int)v158,
                         5,
                         (unsigned int)"curlextents",
                         (unsigned int)&v96,
                         (unsigned int)&v87,
                         (__int64)&v97,
                         0,
                         0) )
    {
LABEL_22:
      sub_130AA40("<jemalloc>: Failure in ctl_bymibname()\n");
      abort();
    }
    if ( *a1 <= 1 )
      sub_130F360(a1);
    sub_40EDDD((__int64)a1, (__int64)"curlextents", 6, (const char **)&v87, v61);
    sub_40E56D(a1, (__int64)"curlextents", v62, v63, v64, v65, v74);
    v45 = (__int64 *)v87;
    v100 = v86;
    v50 = v49 + v79;
    v106 = v49 + v79;
    LOBYTE(v48) = a3 == 0;
    v112 = (_QWORD)v87 * v86;
    v66 = v83;
    v118 = v83;
    if ( v83 && a3 )
    {
      if ( a3 > 0x3B9AC9FF )
      {
        v66 = v83 / v75;
        v50 = v83 % v75;
      }
    }
    else
    {
      v66 = 0;
    }
    v124 = v66;
    v67 = v84;
    v130 = v84;
    if ( v84 && a3 )
    {
      if ( a3 > 0x3B9AC9FF )
      {
        v67 = v84 / v75;
        v50 = v84 % v75;
      }
    }
    else
    {
      v67 = 0;
    }
    v136 = v67;
    v68 = v85;
    v142 = v85;
    if ( v85 && a3 )
    {
      if ( a3 > 0x3B9AC9FF )
      {
        v68 = v85 / v75;
        v50 = v85 % v75;
      }
    }
    else
    {
      v68 = 0;
    }
    v148 = v68;
    v154 = v87;
    if ( v58 )
    {
      if ( *a1 == 2 )
      {
        v45 = &v82;
        sub_40ECF5((int)a1, &v82, v50, v48, v46, v47, v74);
      }
    }
    ++v49;
    v77 = v76;
  }
  result = sub_40E525(a1, (__int64)v45, v50, v48, v46, v47, v74);
  if ( v77 )
    return sub_130F1C0((_DWORD)a1, (unsigned int)"                     ---\n", v70, v71, v72, v73);
  return result;
}
