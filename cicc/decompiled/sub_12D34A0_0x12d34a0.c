// Function: sub_12D34A0
// Address: 0x12d34a0
//
__int64 __fastcall sub_12D34A0(
        unsigned int a1,
        const char **a2,
        int a3,
        int *a4,
        __int64 *a5,
        int *a6,
        __int64 *a7,
        int *a8,
        __int64 *a9,
        int *a10,
        __int64 *a11,
        int *a12,
        __int64 *a13,
        _DWORD *a14)
{
  unsigned int v14; // r12d
  _BYTE v16[8]; // [rsp+8h] [rbp-6A0h] BYREF
  int v17; // [rsp+10h] [rbp-698h]
  char *v18; // [rsp+18h] [rbp-690h]
  __int64 v19; // [rsp+20h] [rbp-688h]
  char v20; // [rsp+28h] [rbp-680h] BYREF
  char *v21; // [rsp+38h] [rbp-670h]
  __int64 v22; // [rsp+40h] [rbp-668h]
  char v23; // [rsp+48h] [rbp-660h] BYREF
  char *v24; // [rsp+58h] [rbp-650h]
  __int64 v25; // [rsp+60h] [rbp-648h]
  char v26; // [rsp+68h] [rbp-640h] BYREF
  char *v27; // [rsp+78h] [rbp-630h]
  __int64 v28; // [rsp+80h] [rbp-628h]
  char v29; // [rsp+88h] [rbp-620h] BYREF
  char *v30; // [rsp+98h] [rbp-610h]
  __int64 v31; // [rsp+A0h] [rbp-608h]
  char v32; // [rsp+A8h] [rbp-600h] BYREF
  char *v33; // [rsp+B8h] [rbp-5F0h]
  __int64 v34; // [rsp+C0h] [rbp-5E8h]
  char v35; // [rsp+C8h] [rbp-5E0h] BYREF
  __int64 v36; // [rsp+D8h] [rbp-5D0h]
  __int64 v37; // [rsp+E0h] [rbp-5C8h]
  __int64 v38; // [rsp+E8h] [rbp-5C0h]
  int v39; // [rsp+100h] [rbp-5A8h]
  int v40; // [rsp+110h] [rbp-598h] BYREF
  __int64 v41; // [rsp+118h] [rbp-590h]
  int *v42; // [rsp+120h] [rbp-588h]
  int *v43; // [rsp+128h] [rbp-580h]
  __int64 v44; // [rsp+130h] [rbp-578h]
  char *v45; // [rsp+138h] [rbp-570h]
  __int64 v46; // [rsp+140h] [rbp-568h]
  char v47; // [rsp+148h] [rbp-560h] BYREF
  char *v48; // [rsp+158h] [rbp-550h]
  __int64 v49; // [rsp+160h] [rbp-548h]
  char v50; // [rsp+168h] [rbp-540h] BYREF
  char *v51; // [rsp+178h] [rbp-530h]
  __int64 v52; // [rsp+180h] [rbp-528h]
  char v53; // [rsp+188h] [rbp-520h] BYREF
  char *v54; // [rsp+198h] [rbp-510h]
  __int64 v55; // [rsp+1A0h] [rbp-508h]
  char v56; // [rsp+1A8h] [rbp-500h] BYREF
  char *v57; // [rsp+1B8h] [rbp-4F0h]
  __int64 v58; // [rsp+1C0h] [rbp-4E8h]
  char v59; // [rsp+1C8h] [rbp-4E0h] BYREF
  char *v60; // [rsp+1D8h] [rbp-4D0h]
  __int64 v61; // [rsp+1E0h] [rbp-4C8h]
  char v62; // [rsp+1E8h] [rbp-4C0h] BYREF
  char *v63; // [rsp+1F8h] [rbp-4B0h]
  __int64 v64; // [rsp+200h] [rbp-4A8h]
  char v65; // [rsp+208h] [rbp-4A0h] BYREF
  char *v66; // [rsp+218h] [rbp-490h]
  __int64 v67; // [rsp+220h] [rbp-488h]
  char v68; // [rsp+228h] [rbp-480h] BYREF
  char *v69; // [rsp+238h] [rbp-470h]
  __int64 v70; // [rsp+240h] [rbp-468h]
  char v71; // [rsp+248h] [rbp-460h] BYREF
  char *v72; // [rsp+258h] [rbp-450h]
  __int64 v73; // [rsp+260h] [rbp-448h]
  char v74; // [rsp+268h] [rbp-440h] BYREF
  char *v75; // [rsp+278h] [rbp-430h]
  __int64 v76; // [rsp+280h] [rbp-428h]
  char v77; // [rsp+288h] [rbp-420h] BYREF
  char *v78; // [rsp+298h] [rbp-410h]
  __int64 v79; // [rsp+2A0h] [rbp-408h]
  char v80; // [rsp+2A8h] [rbp-400h] BYREF
  char *v81; // [rsp+2B8h] [rbp-3F0h]
  __int64 v82; // [rsp+2C0h] [rbp-3E8h]
  char v83; // [rsp+2C8h] [rbp-3E0h] BYREF
  char *v84; // [rsp+2D8h] [rbp-3D0h]
  __int64 v85; // [rsp+2E0h] [rbp-3C8h]
  char v86; // [rsp+2E8h] [rbp-3C0h] BYREF
  char *v87; // [rsp+2F8h] [rbp-3B0h]
  __int64 v88; // [rsp+300h] [rbp-3A8h]
  char v89; // [rsp+308h] [rbp-3A0h] BYREF
  char *v90; // [rsp+318h] [rbp-390h]
  __int64 v91; // [rsp+320h] [rbp-388h]
  char v92; // [rsp+328h] [rbp-380h] BYREF
  char *v93; // [rsp+338h] [rbp-370h]
  __int64 v94; // [rsp+340h] [rbp-368h]
  char v95; // [rsp+348h] [rbp-360h] BYREF
  char *v96; // [rsp+358h] [rbp-350h]
  __int64 v97; // [rsp+360h] [rbp-348h]
  char v98; // [rsp+368h] [rbp-340h] BYREF
  char *v99; // [rsp+378h] [rbp-330h]
  __int64 v100; // [rsp+380h] [rbp-328h]
  char v101; // [rsp+388h] [rbp-320h] BYREF
  char *v102; // [rsp+398h] [rbp-310h]
  __int64 v103; // [rsp+3A0h] [rbp-308h]
  char v104; // [rsp+3A8h] [rbp-300h] BYREF
  char *v105; // [rsp+3B8h] [rbp-2F0h]
  __int64 v106; // [rsp+3C0h] [rbp-2E8h]
  char v107; // [rsp+3C8h] [rbp-2E0h] BYREF
  char *v108; // [rsp+3D8h] [rbp-2D0h]
  __int64 v109; // [rsp+3E0h] [rbp-2C8h]
  char v110; // [rsp+3E8h] [rbp-2C0h] BYREF
  char *v111; // [rsp+3F8h] [rbp-2B0h]
  __int64 v112; // [rsp+400h] [rbp-2A8h]
  char v113; // [rsp+408h] [rbp-2A0h] BYREF
  char *v114; // [rsp+418h] [rbp-290h]
  __int64 v115; // [rsp+420h] [rbp-288h]
  char v116; // [rsp+428h] [rbp-280h] BYREF
  char *v117; // [rsp+438h] [rbp-270h]
  __int64 v118; // [rsp+440h] [rbp-268h]
  char v119; // [rsp+448h] [rbp-260h] BYREF
  char *v120; // [rsp+458h] [rbp-250h]
  __int64 v121; // [rsp+460h] [rbp-248h]
  char v122; // [rsp+468h] [rbp-240h] BYREF
  char *v123; // [rsp+478h] [rbp-230h]
  __int64 v124; // [rsp+480h] [rbp-228h]
  char v125; // [rsp+488h] [rbp-220h] BYREF
  char *v126; // [rsp+498h] [rbp-210h]
  __int64 v127; // [rsp+4A0h] [rbp-208h]
  char v128; // [rsp+4A8h] [rbp-200h] BYREF
  char *v129; // [rsp+4B8h] [rbp-1F0h]
  __int64 v130; // [rsp+4C0h] [rbp-1E8h]
  char v131; // [rsp+4C8h] [rbp-1E0h] BYREF
  char *v132; // [rsp+4D8h] [rbp-1D0h]
  __int64 v133; // [rsp+4E0h] [rbp-1C8h]
  char v134; // [rsp+4E8h] [rbp-1C0h] BYREF
  char *v135; // [rsp+4F8h] [rbp-1B0h]
  __int64 v136; // [rsp+500h] [rbp-1A8h]
  char v137; // [rsp+508h] [rbp-1A0h] BYREF
  char *v138; // [rsp+518h] [rbp-190h]
  __int64 v139; // [rsp+520h] [rbp-188h]
  char v140; // [rsp+528h] [rbp-180h] BYREF
  char *v141; // [rsp+538h] [rbp-170h]
  __int64 v142; // [rsp+540h] [rbp-168h]
  char v143; // [rsp+548h] [rbp-160h] BYREF
  char *v144; // [rsp+558h] [rbp-150h]
  __int64 v145; // [rsp+560h] [rbp-148h]
  char v146; // [rsp+568h] [rbp-140h] BYREF
  char *v147; // [rsp+578h] [rbp-130h]
  __int64 v148; // [rsp+580h] [rbp-128h]
  char v149; // [rsp+588h] [rbp-120h] BYREF
  char *v150; // [rsp+598h] [rbp-110h]
  __int64 v151; // [rsp+5A0h] [rbp-108h]
  char v152; // [rsp+5A8h] [rbp-100h] BYREF
  char *v153; // [rsp+5B8h] [rbp-F0h]
  __int64 v154; // [rsp+5C0h] [rbp-E8h]
  char v155; // [rsp+5C8h] [rbp-E0h] BYREF
  char *v156; // [rsp+5D8h] [rbp-D0h]
  __int64 v157; // [rsp+5E0h] [rbp-C8h]
  char v158; // [rsp+5E8h] [rbp-C0h] BYREF
  char *v159; // [rsp+5F8h] [rbp-B0h]
  __int64 v160; // [rsp+600h] [rbp-A8h]
  char v161; // [rsp+608h] [rbp-A0h] BYREF
  char *v162; // [rsp+618h] [rbp-90h]
  __int64 v163; // [rsp+620h] [rbp-88h]
  char v164; // [rsp+628h] [rbp-80h] BYREF
  char *v165; // [rsp+638h] [rbp-70h]
  __int64 v166; // [rsp+640h] [rbp-68h]
  char v167; // [rsp+648h] [rbp-60h] BYREF
  char *v168; // [rsp+658h] [rbp-50h]
  __int64 v169; // [rsp+660h] [rbp-48h]
  char v170; // [rsp+668h] [rbp-40h] BYREF

  v18 = &v20;
  v21 = &v23;
  v24 = &v26;
  v27 = &v29;
  v30 = &v32;
  v33 = &v35;
  v38 = 0x1000000000LL;
  v42 = &v40;
  v17 = 0;
  v19 = 0;
  v20 = 0;
  v22 = 0;
  v23 = 0;
  v25 = 0;
  v26 = 0;
  v28 = 0;
  v29 = 0;
  v31 = 0;
  v32 = 0;
  v34 = 0;
  v35 = 0;
  v36 = 0;
  v37 = 0;
  v39 = 0;
  v40 = 0;
  v41 = 0;
  v43 = &v40;
  v45 = &v47;
  v48 = &v50;
  v51 = &v53;
  v54 = &v56;
  v57 = &v59;
  v60 = &v62;
  v63 = &v65;
  v66 = &v68;
  v69 = &v71;
  v72 = &v74;
  v75 = &v77;
  v44 = 0;
  v46 = 0;
  v47 = 0;
  v49 = 0;
  v50 = 0;
  v52 = 0;
  v53 = 0;
  v55 = 0;
  v56 = 0;
  v58 = 0;
  v59 = 0;
  v61 = 0;
  v62 = 0;
  v64 = 0;
  v65 = 0;
  v67 = 0;
  v68 = 0;
  v70 = 0;
  v71 = 0;
  v73 = 0;
  v74 = 0;
  v76 = 0;
  v78 = &v80;
  v81 = &v83;
  v84 = &v86;
  v87 = &v89;
  v90 = &v92;
  v93 = &v95;
  v96 = &v98;
  v99 = &v101;
  v102 = &v104;
  v105 = &v107;
  v108 = &v110;
  v77 = 0;
  v79 = 0;
  v80 = 0;
  v82 = 0;
  v83 = 0;
  v85 = 0;
  v86 = 0;
  v88 = 0;
  v89 = 0;
  v91 = 0;
  v92 = 0;
  v94 = 0;
  v95 = 0;
  v97 = 0;
  v98 = 0;
  v100 = 0;
  v101 = 0;
  v103 = 0;
  v104 = 0;
  v106 = 0;
  v107 = 0;
  v109 = 0;
  v111 = &v113;
  v114 = &v116;
  v117 = &v119;
  v120 = &v122;
  v123 = &v125;
  v126 = &v128;
  v129 = &v131;
  v132 = &v134;
  v135 = &v137;
  v138 = &v140;
  v141 = &v143;
  v110 = 0;
  v112 = 0;
  v113 = 0;
  v115 = 0;
  v116 = 0;
  v118 = 0;
  v119 = 0;
  v121 = 0;
  v122 = 0;
  v124 = 0;
  v125 = 0;
  v127 = 0;
  v128 = 0;
  v130 = 0;
  v131 = 0;
  v133 = 0;
  v134 = 0;
  v136 = 0;
  v137 = 0;
  v139 = 0;
  v140 = 0;
  v142 = 0;
  v144 = &v146;
  v147 = &v149;
  v150 = &v152;
  v153 = &v155;
  v156 = &v158;
  v159 = &v161;
  v162 = &v164;
  v165 = &v167;
  v168 = &v170;
  v143 = 0;
  v145 = 0;
  v146 = 0;
  v148 = 0;
  v149 = 0;
  v151 = 0;
  v152 = 0;
  v154 = 0;
  v155 = 0;
  v157 = 0;
  v158 = 0;
  v160 = 0;
  v161 = 0;
  v163 = 0;
  v164 = 0;
  v166 = 0;
  v167 = 0;
  v169 = 0;
  v170 = 0;
  v14 = sub_12D1E20((__int64)v16, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, 0);
  if ( !v14 && a14 )
    *a14 = v17;
  sub_12C73A0((__int64)v16, a1);
  return v14;
}
