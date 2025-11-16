// Function: sub_2ABCD20
// Address: 0x2abcd20
//
__int64 __fastcall sub_2ABCD20(__int64 a1, _WORD *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // rdx
  __int64 v11; // rdx
  __int64 v12; // rdx
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // rdx
  __int64 v30; // rdx
  __int64 v31; // rdx
  __int64 v32; // rdx
  __int64 v33; // rdx
  __int64 v34; // rdx
  __int64 v35; // rdx
  __int64 v36; // rdx
  __int64 v37; // rdx
  __int64 v38; // rax
  __int64 v39; // rax
  __int64 v40; // rax
  __int64 v41; // rax
  __int64 v42; // rax
  __int64 v43; // rax
  __int64 v44; // rax
  __int64 v45; // rax
  __int64 v46; // rax
  __int64 v47; // rax
  __int64 v48; // rax
  __int64 v49; // rax
  __int64 v50; // rax
  __int64 v51; // rax
  __int64 v52; // rax
  __int64 v53; // rax
  __int64 v54; // rax
  __int64 v55; // rax
  _QWORD v57[4]; // [rsp+80h] [rbp-690h] BYREF
  _BYTE v58[64]; // [rsp+A0h] [rbp-670h] BYREF
  __int64 v59; // [rsp+E0h] [rbp-630h]
  __int64 v60; // [rsp+E8h] [rbp-628h]
  __int64 v61; // [rsp+F0h] [rbp-620h]
  unsigned __int16 v62; // [rsp+F8h] [rbp-618h]
  _QWORD v63[4]; // [rsp+100h] [rbp-610h] BYREF
  _BYTE v64[64]; // [rsp+120h] [rbp-5F0h] BYREF
  __int64 v65; // [rsp+160h] [rbp-5B0h]
  __int64 v66; // [rsp+168h] [rbp-5A8h]
  __int64 v67; // [rsp+170h] [rbp-5A0h]
  __int16 v68; // [rsp+178h] [rbp-598h]
  _QWORD v69[4]; // [rsp+190h] [rbp-580h] BYREF
  _BYTE v70[64]; // [rsp+1B0h] [rbp-560h] BYREF
  __int64 v71; // [rsp+1F0h] [rbp-520h]
  __int64 v72; // [rsp+1F8h] [rbp-518h]
  __int64 v73; // [rsp+200h] [rbp-510h]
  unsigned __int16 v74; // [rsp+208h] [rbp-508h]
  _QWORD v75[4]; // [rsp+210h] [rbp-500h] BYREF
  _BYTE v76[64]; // [rsp+230h] [rbp-4E0h] BYREF
  __int64 v77; // [rsp+270h] [rbp-4A0h]
  __int64 v78; // [rsp+278h] [rbp-498h]
  __int64 v79; // [rsp+280h] [rbp-490h]
  __int16 v80; // [rsp+288h] [rbp-488h]
  _BYTE v81[32]; // [rsp+2A0h] [rbp-470h] BYREF
  _BYTE v82[64]; // [rsp+2C0h] [rbp-450h] BYREF
  __int64 v83; // [rsp+300h] [rbp-410h]
  __int64 v84; // [rsp+308h] [rbp-408h]
  __int64 v85; // [rsp+310h] [rbp-400h]
  unsigned __int16 v86; // [rsp+318h] [rbp-3F8h]
  _BYTE v87[32]; // [rsp+320h] [rbp-3F0h] BYREF
  _BYTE v88[64]; // [rsp+340h] [rbp-3D0h] BYREF
  __int64 v89; // [rsp+380h] [rbp-390h]
  __int64 v90; // [rsp+388h] [rbp-388h]
  __int64 v91; // [rsp+390h] [rbp-380h]
  __int16 v92; // [rsp+398h] [rbp-378h]
  __int16 v93; // [rsp+3A8h] [rbp-368h]
  _BYTE v94[32]; // [rsp+3B0h] [rbp-360h] BYREF
  _BYTE v95[64]; // [rsp+3D0h] [rbp-340h] BYREF
  __int64 v96; // [rsp+410h] [rbp-300h]
  __int64 v97; // [rsp+418h] [rbp-2F8h]
  __int64 v98; // [rsp+420h] [rbp-2F0h]
  unsigned __int16 v99; // [rsp+428h] [rbp-2E8h]
  _BYTE v100[32]; // [rsp+430h] [rbp-2E0h] BYREF
  _BYTE v101[64]; // [rsp+450h] [rbp-2C0h] BYREF
  __int64 v102; // [rsp+490h] [rbp-280h]
  __int64 v103; // [rsp+498h] [rbp-278h]
  __int64 v104; // [rsp+4A0h] [rbp-270h]
  __int16 v105; // [rsp+4A8h] [rbp-268h]
  __int16 v106; // [rsp+4B8h] [rbp-258h]
  _BYTE v107[32]; // [rsp+4C0h] [rbp-250h] BYREF
  _BYTE v108[64]; // [rsp+4E0h] [rbp-230h] BYREF
  __int64 v109; // [rsp+520h] [rbp-1F0h]
  __int64 v110; // [rsp+528h] [rbp-1E8h]
  __int64 v111; // [rsp+530h] [rbp-1E0h]
  unsigned __int16 v112; // [rsp+538h] [rbp-1D8h]
  _BYTE v113[32]; // [rsp+540h] [rbp-1D0h] BYREF
  _BYTE v114[64]; // [rsp+560h] [rbp-1B0h] BYREF
  __int64 v115; // [rsp+5A0h] [rbp-170h]
  __int64 v116; // [rsp+5A8h] [rbp-168h]
  __int64 v117; // [rsp+5B0h] [rbp-160h]
  __int16 v118; // [rsp+5B8h] [rbp-158h]
  __int16 v119; // [rsp+5C8h] [rbp-148h]
  _BYTE v120[32]; // [rsp+5D0h] [rbp-140h] BYREF
  _BYTE v121[64]; // [rsp+5F0h] [rbp-120h] BYREF
  __int64 v122; // [rsp+630h] [rbp-E0h]
  __int64 v123; // [rsp+638h] [rbp-D8h]
  __int64 v124; // [rsp+640h] [rbp-D0h]
  unsigned __int16 v125; // [rsp+648h] [rbp-C8h]
  _BYTE v126[32]; // [rsp+650h] [rbp-C0h] BYREF
  _BYTE v127[64]; // [rsp+670h] [rbp-A0h] BYREF
  __int64 v128; // [rsp+6B0h] [rbp-60h]
  __int64 v129; // [rsp+6B8h] [rbp-58h]
  __int64 v130; // [rsp+6C0h] [rbp-50h]
  __int16 v131; // [rsp+6C8h] [rbp-48h]
  __int16 v132; // [rsp+6D8h] [rbp-38h]

  sub_2ABCC20(v69, (__int64)(a2 + 132), a3, a4, a5, a6);
  v74 = a2[192];
  sub_2ABCC20(v75, (__int64)(a2 + 196), v74, v7, v8, v9);
  v80 = a2[256];
  sub_C8CF70((__int64)v107, v108, 8, (__int64)v70, (__int64)v69);
  v109 = v71;
  v110 = v72;
  v72 = 0;
  v111 = v73;
  v73 = 0;
  v112 = v74;
  v71 = 0;
  sub_C8CF70((__int64)v113, v114, 8, (__int64)v76, (__int64)v75);
  v115 = v77;
  v77 = 0;
  v116 = v78;
  v78 = 0;
  v117 = v79;
  v79 = 0;
  v118 = v80;
  sub_C8CF70((__int64)v120, v121, 8, (__int64)v108, (__int64)v107);
  v10 = v109;
  v109 = 0;
  v122 = v10;
  v123 = v110;
  v110 = 0;
  v124 = v111;
  v111 = 0;
  v125 = v112;
  sub_C8CF70((__int64)v126, v127, 8, (__int64)v114, (__int64)v113);
  v128 = v115;
  v129 = v116;
  v116 = 0;
  v130 = v117;
  v117 = 0;
  v131 = v118;
  v115 = 0;
  sub_C8CF70((__int64)v94, v95, 8, (__int64)v121, (__int64)v120);
  v96 = v122;
  v97 = v123;
  v123 = 0;
  v98 = v124;
  v124 = 0;
  v99 = v125;
  v122 = 0;
  sub_C8CF70((__int64)v100, v101, 8, (__int64)v127, (__int64)v126);
  v11 = v128;
  v128 = 0;
  v102 = v11;
  v12 = v129;
  v129 = 0;
  v103 = v12;
  v13 = v130;
  v130 = 0;
  v104 = v13;
  v105 = v131;
  sub_2AB1B50((__int64)v126);
  sub_2AB1B50((__int64)v120);
  v106 = 256;
  sub_2AB1B50((__int64)v113);
  sub_2AB1B50((__int64)v107);
  sub_2ABCC20(v57, (__int64)a2, (__int64)v57, v14, v15, v16);
  v62 = a2[60];
  sub_2ABCC20(v63, (__int64)(a2 + 64), v62, v17, v18, (__int64)v63);
  v68 = a2[124];
  sub_C8CF70((__int64)v107, v108, 8, (__int64)v58, (__int64)v57);
  v19 = v59;
  v59 = 0;
  v109 = v19;
  v20 = v60;
  v60 = 0;
  v110 = v20;
  v21 = v61;
  v61 = 0;
  v111 = v21;
  v112 = v62;
  sub_C8CF70((__int64)v113, v114, 8, (__int64)v64, (__int64)v63);
  v22 = v65;
  v65 = 0;
  v115 = v22;
  v23 = v66;
  v66 = 0;
  v116 = v23;
  v24 = v67;
  v67 = 0;
  v117 = v24;
  v118 = v68;
  sub_C8CF70((__int64)v120, v121, 8, (__int64)v108, (__int64)v107);
  v25 = v109;
  v109 = 0;
  v122 = v25;
  v26 = v110;
  v110 = 0;
  v123 = v26;
  v27 = v111;
  v111 = 0;
  v124 = v27;
  v125 = v112;
  sub_C8CF70((__int64)v126, v127, 8, (__int64)v114, (__int64)v113);
  v28 = v115;
  v115 = 0;
  v128 = v28;
  v129 = v116;
  v116 = 0;
  v130 = v117;
  v117 = 0;
  v131 = v118;
  sub_C8CF70((__int64)v81, v82, 8, (__int64)v121, (__int64)v120);
  v83 = v122;
  v84 = v123;
  v123 = 0;
  v85 = v124;
  v124 = 0;
  v86 = v125;
  v122 = 0;
  sub_C8CF70((__int64)v87, v88, 8, (__int64)v127, (__int64)v126);
  v29 = v128;
  v128 = 0;
  v89 = v29;
  v30 = v129;
  v129 = 0;
  v90 = v30;
  v31 = v130;
  v130 = 0;
  v91 = v31;
  v92 = v131;
  sub_2AB1B50((__int64)v126);
  sub_2AB1B50((__int64)v120);
  v93 = 256;
  sub_2AB1B50((__int64)v113);
  sub_2AB1B50((__int64)v107);
  sub_C8CF70((__int64)v120, v121, 8, (__int64)v95, (__int64)v94);
  v32 = v96;
  v96 = 0;
  v122 = v32;
  v33 = v97;
  v97 = 0;
  v123 = v33;
  v34 = v98;
  v98 = 0;
  v124 = v34;
  v125 = v99;
  sub_C8CF70((__int64)v126, v127, 8, (__int64)v101, (__int64)v100);
  v35 = v102;
  v102 = 0;
  v128 = v35;
  v36 = v103;
  v103 = 0;
  v129 = v36;
  v37 = v104;
  v104 = 0;
  v130 = v37;
  v131 = v105;
  v132 = v106;
  sub_C8CF70((__int64)v107, v108, 8, (__int64)v82, (__int64)v81);
  v38 = v83;
  v83 = 0;
  v109 = v38;
  v39 = v84;
  v84 = 0;
  v110 = v39;
  v40 = v85;
  v85 = 0;
  v111 = v40;
  v112 = v86;
  sub_C8CF70((__int64)v113, v114, 8, (__int64)v88, (__int64)v87);
  v41 = v89;
  v89 = 0;
  v115 = v41;
  v42 = v90;
  v90 = 0;
  v116 = v42;
  v43 = v91;
  v91 = 0;
  v117 = v43;
  v118 = v92;
  v119 = v93;
  sub_C8CF70(a1, (void *)(a1 + 32), 8, (__int64)v108, (__int64)v107);
  v44 = v109;
  v109 = 0;
  *(_QWORD *)(a1 + 96) = v44;
  v45 = v110;
  v110 = 0;
  *(_QWORD *)(a1 + 104) = v45;
  v46 = v111;
  v111 = 0;
  *(_QWORD *)(a1 + 112) = v46;
  *(_WORD *)(a1 + 120) = v112;
  sub_C8CF70(a1 + 128, (void *)(a1 + 160), 8, (__int64)v114, (__int64)v113);
  v47 = v115;
  v115 = 0;
  *(_QWORD *)(a1 + 224) = v47;
  v48 = v116;
  v116 = 0;
  *(_QWORD *)(a1 + 232) = v48;
  v49 = v117;
  v117 = 0;
  *(_QWORD *)(a1 + 240) = v49;
  *(_WORD *)(a1 + 248) = v118;
  *(_WORD *)(a1 + 264) = v119;
  sub_C8CF70(a1 + 272, (void *)(a1 + 304), 8, (__int64)v121, (__int64)v120);
  v50 = v122;
  v122 = 0;
  *(_QWORD *)(a1 + 368) = v50;
  v51 = v123;
  v123 = 0;
  *(_QWORD *)(a1 + 376) = v51;
  v52 = v124;
  v124 = 0;
  *(_QWORD *)(a1 + 384) = v52;
  *(_WORD *)(a1 + 392) = v125;
  sub_C8CF70(a1 + 400, (void *)(a1 + 432), 8, (__int64)v127, (__int64)v126);
  v53 = v128;
  v128 = 0;
  *(_QWORD *)(a1 + 496) = v53;
  v54 = v129;
  v129 = 0;
  *(_QWORD *)(a1 + 504) = v54;
  v55 = v130;
  v130 = 0;
  *(_QWORD *)(a1 + 512) = v55;
  *(_WORD *)(a1 + 520) = v131;
  *(_WORD *)(a1 + 536) = v132;
  sub_2AB1B50((__int64)v113);
  sub_2AB1B50((__int64)v107);
  sub_2AB1B50((__int64)v126);
  sub_2AB1B50((__int64)v120);
  sub_2AB1B50((__int64)v87);
  sub_2AB1B50((__int64)v81);
  sub_2AB1B50((__int64)v63);
  sub_2AB1B50((__int64)v57);
  sub_2AB1B50((__int64)v100);
  sub_2AB1B50((__int64)v94);
  sub_2AB1B50((__int64)v75);
  sub_2AB1B50((__int64)v69);
  return a1;
}
