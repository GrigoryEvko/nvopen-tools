// Function: sub_2AD74E0
// Address: 0x2ad74e0
//
__int64 __fastcall sub_2AD74E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r9
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  __int64 v24; // rdx
  __int64 v25; // rdx
  __int64 v26; // rdx
  __int64 v27; // rdx
  __int64 v28; // rcx
  __int64 v29; // r8
  __int64 v30; // r9
  __int64 v31; // r8
  __int64 v32; // r9
  __int64 v33; // rcx
  __int64 v34; // rsi
  __int64 v35; // rdx
  __int64 v36; // rcx
  __int64 v37; // r8
  __int64 v38; // r9
  __int64 v39; // rcx
  __int64 v40; // r8
  __int64 v41; // r9
  __int64 v42; // rcx
  __int64 v43; // r8
  __int64 v44; // r9
  __int64 v45; // rcx
  __int64 v46; // r8
  __int64 v47; // r9
  __int64 v48; // rcx
  __int64 v49; // r8
  __int64 v50; // r9
  __int64 v51; // rdx
  __int64 v52; // rdx
  __int64 v53; // rdx
  __int64 v54; // rdx
  __int64 v55; // rcx
  __int64 v56; // r8
  __int64 v57; // r9
  __int64 v58; // r8
  __int64 v59; // r9
  __int64 v60; // rcx
  __int64 v61; // rsi
  __int64 v62; // rdx
  __int64 v63; // rdx
  __int64 v64; // rdx
  __int64 v65; // rdx
  __int64 v66; // rdx
  __int64 v67; // rdx
  __int64 v68; // rdx
  __int64 v69; // rdx
  __int64 v70; // rdx
  __int64 v71; // rdx
  __int64 v72; // rdx
  __int64 v73; // rdx
  __int64 v74; // rdx
  __int64 v75; // rdx
  __int64 v76; // rdx
  __int64 v77; // rdx
  char v79; // al
  char v80; // al
  _QWORD v81[15]; // [rsp+70h] [rbp-7F0h] BYREF
  unsigned __int16 v82; // [rsp+E8h] [rbp-778h]
  _QWORD v83[15]; // [rsp+F0h] [rbp-770h] BYREF
  unsigned __int16 v84; // [rsp+168h] [rbp-6F8h]
  _QWORD v85[15]; // [rsp+170h] [rbp-6F0h] BYREF
  unsigned __int16 v86; // [rsp+1E8h] [rbp-678h]
  _QWORD v87[15]; // [rsp+1F0h] [rbp-670h] BYREF
  unsigned __int16 v88; // [rsp+268h] [rbp-5F8h]
  _QWORD v89[15]; // [rsp+270h] [rbp-5F0h] BYREF
  unsigned __int16 v90; // [rsp+2E8h] [rbp-578h]
  _QWORD v91[15]; // [rsp+2F0h] [rbp-570h] BYREF
  unsigned __int16 v92; // [rsp+368h] [rbp-4F8h]
  _QWORD v93[15]; // [rsp+370h] [rbp-4F0h] BYREF
  unsigned __int16 v94; // [rsp+3E8h] [rbp-478h]
  _QWORD v95[4]; // [rsp+3F0h] [rbp-470h] BYREF
  _BYTE v96[64]; // [rsp+410h] [rbp-450h] BYREF
  __int64 v97; // [rsp+450h] [rbp-410h]
  __int64 v98; // [rsp+458h] [rbp-408h]
  __int64 v99; // [rsp+460h] [rbp-400h]
  unsigned __int16 v100; // [rsp+468h] [rbp-3F8h]
  _QWORD v101[4]; // [rsp+470h] [rbp-3F0h] BYREF
  char v102[64]; // [rsp+490h] [rbp-3D0h] BYREF
  __int64 v103; // [rsp+4D0h] [rbp-390h]
  __int64 v104; // [rsp+4D8h] [rbp-388h]
  __int64 v105; // [rsp+4E0h] [rbp-380h]
  unsigned __int16 v106; // [rsp+4E8h] [rbp-378h]
  _BYTE v107[32]; // [rsp+500h] [rbp-360h] BYREF
  _BYTE v108[64]; // [rsp+520h] [rbp-340h] BYREF
  __int64 v109; // [rsp+560h] [rbp-300h]
  __int64 v110; // [rsp+568h] [rbp-2F8h]
  __int64 v111; // [rsp+570h] [rbp-2F0h]
  unsigned __int16 v112; // [rsp+578h] [rbp-2E8h]
  _QWORD v113[4]; // [rsp+580h] [rbp-2E0h] BYREF
  char v114[64]; // [rsp+5A0h] [rbp-2C0h] BYREF
  __int64 v115; // [rsp+5E0h] [rbp-280h]
  __int64 v116; // [rsp+5E8h] [rbp-278h]
  __int64 v117; // [rsp+5F0h] [rbp-270h]
  unsigned __int16 v118; // [rsp+5F8h] [rbp-268h]
  _QWORD v119[4]; // [rsp+610h] [rbp-250h] BYREF
  _BYTE v120[64]; // [rsp+630h] [rbp-230h] BYREF
  __int64 v121; // [rsp+670h] [rbp-1F0h]
  __int64 v122; // [rsp+678h] [rbp-1E8h]
  __int64 v123; // [rsp+680h] [rbp-1E0h]
  unsigned __int16 v124; // [rsp+688h] [rbp-1D8h]
  _BYTE v125[32]; // [rsp+690h] [rbp-1D0h] BYREF
  _BYTE v126[64]; // [rsp+6B0h] [rbp-1B0h] BYREF
  __int64 v127; // [rsp+6F0h] [rbp-170h]
  __int64 v128; // [rsp+6F8h] [rbp-168h]
  __int64 v129; // [rsp+700h] [rbp-160h]
  unsigned __int16 v130; // [rsp+708h] [rbp-158h]
  _QWORD v131[4]; // [rsp+720h] [rbp-140h] BYREF
  _BYTE v132[64]; // [rsp+740h] [rbp-120h] BYREF
  __int64 v133; // [rsp+780h] [rbp-E0h]
  __int64 v134; // [rsp+788h] [rbp-D8h]
  __int64 v135; // [rsp+790h] [rbp-D0h]
  unsigned __int16 v136; // [rsp+798h] [rbp-C8h]
  _BYTE v137[32]; // [rsp+7A0h] [rbp-C0h] BYREF
  _BYTE v138[64]; // [rsp+7C0h] [rbp-A0h] BYREF
  __int64 v139; // [rsp+800h] [rbp-60h]
  __int64 v140; // [rsp+808h] [rbp-58h]
  __int64 v141; // [rsp+810h] [rbp-50h]
  unsigned __int16 v142; // [rsp+818h] [rbp-48h]

  sub_2ABCC20(v81, a2, a3, a4, a5, a6);
  v82 = *(_WORD *)(a2 + 120);
  sub_2ABCC20(v83, a2 + 128, v82, v6, v7, v8);
  v84 = *(_WORD *)(a2 + 248);
  sub_2ABCC20(v91, (__int64)v83, v84, v9, v10, v11);
  v92 = v84;
  sub_2ABCC20(v89, (__int64)v83, v84, v12, v13, v14);
  v90 = v84;
  sub_2ABCC20(v119, (__int64)v91, v84, v15, v16, v17);
  v124 = v92;
  sub_2ABCC20(v95, (__int64)v89, v92, v18, v19, v20);
  v100 = v90;
  sub_2ABCC20(v131, (__int64)v95, v90, v21, v22, v23);
  v136 = v100;
  sub_C8CF70((__int64)v107, v108, 8, (__int64)v132, (__int64)v131);
  v24 = v133;
  v133 = 0;
  v109 = v24;
  v25 = v134;
  v134 = 0;
  v110 = v25;
  v26 = v135;
  v135 = 0;
  v111 = v26;
  v112 = v136;
  sub_2AB1B50((__int64)v131);
  sub_2ABCC20(v113, (__int64)v119, v27, v28, v29, v30);
  v118 = v124;
  while ( 1 )
  {
    v33 = v109;
    v34 = v115;
    if ( v110 - v109 != v116 - v115 )
      goto LABEL_3;
    if ( v109 == v110 )
      break;
    while ( *(_QWORD *)v33 == *(_QWORD *)v34 )
    {
      v79 = *(_BYTE *)(v33 + 24);
      if ( v79 != *(_BYTE *)(v34 + 24)
        || v79 && (*(_QWORD *)(v33 + 8) != *(_QWORD *)(v34 + 8) || *(_QWORD *)(v33 + 16) != *(_QWORD *)(v34 + 16)) )
      {
        break;
      }
      v33 += 32;
      v34 += 32;
      if ( v110 == v33 )
        goto LABEL_4;
    }
LABEL_3:
    if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v110 - 32) + 8LL) - 1 <= 1 )
      break;
    sub_2AD7320((__int64)v107, v34, v110, v33, v31, v32);
  }
LABEL_4:
  sub_2AB1B50((__int64)v95);
  sub_2AB1B50((__int64)v119);
  sub_2ABCC20(v87, (__int64)v83, v35, v36, v37, v38);
  v88 = v84;
  sub_2ABCC20(v85, (__int64)v81, v84, v39, v40, v41);
  v86 = v82;
  sub_2ABCC20(v119, (__int64)v87, v82, v42, v43, v44);
  v124 = v88;
  sub_2ABCC20(v93, (__int64)v85, v88, v45, v46, v47);
  v94 = v86;
  sub_2ABCC20(v131, (__int64)v93, v86, v48, v49, v50);
  v136 = v94;
  sub_C8CF70((__int64)v95, v96, 8, (__int64)v132, (__int64)v131);
  v51 = v133;
  v133 = 0;
  v97 = v51;
  v52 = v134;
  v134 = 0;
  v98 = v52;
  v53 = v135;
  v135 = 0;
  v99 = v53;
  v100 = v136;
  sub_2AB1B50((__int64)v131);
  sub_2ABCC20(v101, (__int64)v119, v54, v55, v56, v57);
  v106 = v124;
  while ( 1 )
  {
    v60 = v97;
    v61 = v103;
    if ( v98 - v97 != v104 - v103 )
      goto LABEL_6;
    if ( v97 == v98 )
      break;
    while ( *(_QWORD *)v60 == *(_QWORD *)v61 )
    {
      v80 = *(_BYTE *)(v60 + 24);
      if ( v80 != *(_BYTE *)(v61 + 24)
        || v80 && (*(_QWORD *)(v60 + 8) != *(_QWORD *)(v61 + 8) || *(_QWORD *)(v60 + 16) != *(_QWORD *)(v61 + 16)) )
      {
        break;
      }
      v60 += 32;
      v61 += 32;
      if ( v98 == v60 )
        goto LABEL_7;
    }
LABEL_6:
    if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v98 - 32) + 8LL) - 1 <= 1 )
      break;
    sub_2AD7320((__int64)v95, v61, v98, v60, v58, v59);
  }
LABEL_7:
  sub_2AB1B50((__int64)v93);
  sub_2AB1B50((__int64)v119);
  sub_C8CF70((__int64)v131, v132, 8, (__int64)v108, (__int64)v107);
  v133 = v109;
  v109 = 0;
  v134 = v110;
  v110 = 0;
  v135 = v111;
  v111 = 0;
  v136 = v112;
  sub_C8CF70((__int64)v137, v138, 8, (__int64)v114, (__int64)v113);
  v62 = v115;
  v115 = 0;
  v139 = v62;
  v140 = v116;
  v116 = 0;
  v141 = v117;
  v117 = 0;
  v142 = v118;
  sub_C8CF70((__int64)v119, v120, 8, (__int64)v96, (__int64)v95);
  v121 = v97;
  v122 = v98;
  v123 = v99;
  v99 = 0;
  v124 = v100;
  v98 = 0;
  v97 = 0;
  sub_C8CF70((__int64)v125, v126, 8, (__int64)v102, (__int64)v101);
  v63 = v103;
  v103 = 0;
  v127 = v63;
  v64 = v104;
  v104 = 0;
  v128 = v64;
  v65 = v105;
  v105 = 0;
  v129 = v65;
  v130 = v106;
  sub_C8CF70(a1, (void *)(a1 + 32), 8, (__int64)v120, (__int64)v119);
  v66 = v121;
  v121 = 0;
  *(_QWORD *)(a1 + 96) = v66;
  v67 = v122;
  v122 = 0;
  *(_QWORD *)(a1 + 104) = v67;
  v68 = v123;
  v123 = 0;
  *(_QWORD *)(a1 + 112) = v68;
  *(_WORD *)(a1 + 120) = v124;
  sub_C8CF70(a1 + 128, (void *)(a1 + 160), 8, (__int64)v126, (__int64)v125);
  v69 = v127;
  v127 = 0;
  *(_QWORD *)(a1 + 224) = v69;
  v70 = v128;
  v128 = 0;
  *(_QWORD *)(a1 + 232) = v70;
  v71 = v129;
  v129 = 0;
  *(_QWORD *)(a1 + 240) = v71;
  *(_WORD *)(a1 + 248) = v130;
  sub_C8CF70(a1 + 264, (void *)(a1 + 296), 8, (__int64)v132, (__int64)v131);
  v72 = v133;
  v133 = 0;
  *(_QWORD *)(a1 + 360) = v72;
  v73 = v134;
  v134 = 0;
  *(_QWORD *)(a1 + 368) = v73;
  v74 = v135;
  v135 = 0;
  *(_QWORD *)(a1 + 376) = v74;
  *(_WORD *)(a1 + 384) = v136;
  sub_C8CF70(a1 + 392, (void *)(a1 + 424), 8, (__int64)v138, (__int64)v137);
  v75 = v139;
  v139 = 0;
  *(_QWORD *)(a1 + 488) = v75;
  v76 = v140;
  v140 = 0;
  *(_QWORD *)(a1 + 496) = v76;
  v77 = v141;
  v141 = 0;
  *(_QWORD *)(a1 + 504) = v77;
  *(_WORD *)(a1 + 512) = v142;
  sub_2AB1B50((__int64)v125);
  sub_2AB1B50((__int64)v119);
  sub_2AB1B50((__int64)v137);
  sub_2AB1B50((__int64)v131);
  sub_2AB1B50((__int64)v101);
  sub_2AB1B50((__int64)v95);
  sub_2AB1B50((__int64)v85);
  sub_2AB1B50((__int64)v87);
  sub_2AB1B50((__int64)v113);
  sub_2AB1B50((__int64)v107);
  sub_2AB1B50((__int64)v89);
  sub_2AB1B50((__int64)v91);
  sub_2AB1B50((__int64)v83);
  sub_2AB1B50((__int64)v81);
  return a1;
}
