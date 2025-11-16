// Function: sub_17D2020
// Address: 0x17d2020
//
__int64 __fastcall sub_17D2020(__int64 a1, double a2, double a3, double a4)
{
  __int64 result; // rax
  __int64 *v5; // r9
  __int64 v6; // rax
  __int64 v7; // r13
  __int64 *v8; // rax
  __int64 v9; // rax
  _QWORD *v10; // r13
  __int64 v11; // rax
  __int64 v12; // rbx
  __int64 v13; // rax
  _BYTE *v14; // rbx
  __int64 v15; // rax
  __int64 v16; // rax
  _QWORD *v17; // rbx
  __int64 *v18; // r9
  __int64 v19; // r13
  __int64 *v20; // rax
  _QWORD *v21; // r13
  __int64 v22; // rax
  __int64 v23; // rbx
  __int64 v24; // rax
  _QWORD *v25; // rax
  __int64 v26; // rbx
  __int64 v27; // rsi
  __int64 v28; // rsi
  __int64 v29; // r13
  __int64 v30; // rax
  __int64 v31; // rbx
  __int64 **v32; // rdx
  __int64 v33; // r10
  __int64 v34; // rax
  __int64 v35; // rax
  __int64 v36; // rax
  __int64 v37; // rax
  __int64 v38; // rbx
  __int64 v39; // rax
  __int64 **v40; // rdx
  __int64 v41; // r10
  __int64 v42; // rax
  __int64 v43; // rax
  _QWORD *v44; // rbx
  __int64 v45; // rax
  __int64 v46; // rax
  __int64 v47; // rax
  _QWORD *v48; // rax
  __int64 v49; // rax
  __int64 v50; // rax
  __int64 v51; // rbx
  __int64 v52; // rax
  __int64 v53; // rax
  __int64 v54; // rax
  __int64 v55; // rax
  __int64 v56; // rax
  _QWORD *v57; // rbx
  __int64 v58; // rax
  __int64 v59; // rax
  __int64 v60; // rax
  __int64 v61; // rax
  __int64 v62; // rax
  __int64 v63; // rax
  _QWORD *v64; // rax
  __int64 v65; // r13
  __int64 v66; // rax
  __int64 v67; // rbx
  __int64 *v68; // rax
  __int64 v69; // rax
  _BYTE *v70; // rbx
  __int64 v71; // rax
  _QWORD *v72; // rbx
  __int64 v73; // rax
  __int64 v74; // r9
  __int64 v75; // rax
  __int64 v76; // rsi
  __int64 v77; // rax
  __int64 v78; // r9
  __int64 v79; // rax
  __int64 v80; // rsi
  __int64 v81; // r10
  __int64 v82; // rax
  __int64 v83; // rsi
  __int64 v84; // rsi
  __int64 v85; // rdx
  unsigned __int8 *v86; // rsi
  __int64 v87; // r10
  __int64 v88; // rax
  __int64 v89; // rsi
  __int64 v90; // rsi
  __int64 v91; // rdx
  unsigned __int8 *v92; // rsi
  __int64 v93; // rax
  _QWORD *v94; // rax
  __int64 v95; // r13
  __int64 v96; // rax
  __int64 *v97; // rbx
  _QWORD *v98; // rax
  _QWORD *v99; // rax
  __int64 v100; // [rsp+8h] [rbp-168h]
  __int64 v101; // [rsp+10h] [rbp-160h]
  __int64 *v102; // [rsp+10h] [rbp-160h]
  __int64 v103; // [rsp+18h] [rbp-158h]
  __int64 v104; // [rsp+18h] [rbp-158h]
  __int64 v105; // [rsp+18h] [rbp-158h]
  __int64 v106; // [rsp+18h] [rbp-158h]
  __int64 v107; // [rsp+18h] [rbp-158h]
  __int64 *v108; // [rsp+18h] [rbp-158h]
  _QWORD *v109; // [rsp+20h] [rbp-150h]
  __int64 *v110; // [rsp+20h] [rbp-150h]
  __int64 v111; // [rsp+28h] [rbp-148h]
  __int64 v112; // [rsp+28h] [rbp-148h]
  __int64 v113; // [rsp+28h] [rbp-148h]
  __int64 v114; // [rsp+28h] [rbp-148h]
  __int64 *v115; // [rsp+28h] [rbp-148h]
  __int64 v116; // [rsp+28h] [rbp-148h]
  __int64 v117; // [rsp+28h] [rbp-148h]
  __int64 v118; // [rsp+28h] [rbp-148h]
  __int64 v119; // [rsp+28h] [rbp-148h]
  __int64 v120; // [rsp+28h] [rbp-148h]
  __int64 v121; // [rsp+28h] [rbp-148h]
  _BYTE *v122; // [rsp+30h] [rbp-140h]
  _BYTE *v123; // [rsp+30h] [rbp-140h]
  __int64 v124; // [rsp+30h] [rbp-140h]
  __int64 v125; // [rsp+30h] [rbp-140h]
  __int64 v126; // [rsp+30h] [rbp-140h]
  __int64 v127; // [rsp+30h] [rbp-140h]
  _QWORD *v128; // [rsp+30h] [rbp-140h]
  __int64 v129; // [rsp+30h] [rbp-140h]
  __int64 v130; // [rsp+30h] [rbp-140h]
  __int64 v131; // [rsp+38h] [rbp-138h]
  __int64 v132; // [rsp+40h] [rbp-130h]
  __int64 v133; // [rsp+48h] [rbp-128h]
  __int64 v134; // [rsp+68h] [rbp-108h] BYREF
  __int64 v135[2]; // [rsp+70h] [rbp-100h] BYREF
  __int16 v136; // [rsp+80h] [rbp-F0h]
  _BYTE v137[16]; // [rsp+90h] [rbp-E0h] BYREF
  __int16 v138; // [rsp+A0h] [rbp-D0h]
  __int64 v139[2]; // [rsp+B0h] [rbp-C0h] BYREF
  __int16 v140; // [rsp+C0h] [rbp-B0h]
  __int64 v141[2]; // [rsp+D0h] [rbp-A0h] BYREF
  __int16 v142; // [rsp+E0h] [rbp-90h]
  __int64 v143; // [rsp+F0h] [rbp-80h] BYREF
  __int64 v144; // [rsp+F8h] [rbp-78h]
  __int64 *v145; // [rsp+100h] [rbp-70h]
  _QWORD *v146; // [rsp+108h] [rbp-68h]

  if ( *(_DWORD *)(a1 + 56) )
  {
    v93 = sub_157ED20(*(_QWORD *)(*(_QWORD *)(a1 + 24) + 480LL));
    sub_17CE510((__int64)&v143, v93, 0, 0, 0);
    v142 = 257;
    v94 = sub_156E5B0(&v143, *(_QWORD *)(*(_QWORD *)(a1 + 16) + 232LL), (__int64)v141);
    *(_QWORD *)(a1 + 40) = v94;
    v95 = (__int64)v94;
    v142 = 257;
    v96 = sub_15A0680(*(_QWORD *)(*(_QWORD *)(a1 + 16) + 176LL), 192, 0);
    v97 = (__int64 *)sub_12899C0(&v143, v96, v95, (__int64)v141, 0, 0);
    v142 = 257;
    v98 = (_QWORD *)sub_1643330(*(_QWORD **)(*(_QWORD *)(a1 + 16) + 168LL));
    v99 = sub_17CEAE0(&v143, v98, (__int64)v97, v141);
    *(_QWORD *)(a1 + 32) = v99;
    sub_15E7430(&v143, v99, 8u, *(_QWORD **)(*(_QWORD *)(a1 + 16) + 224LL), 8u, v97, 0, 0, 0, 0, 0);
    sub_17CD270(&v143);
  }
  v132 = sub_15A0680(*(_QWORD *)(*(_QWORD *)(a1 + 16) + 176LL), 64, 0);
  v131 = sub_15A0680(*(_QWORD *)(*(_QWORD *)(a1 + 16) + 176LL), 128, 0);
  result = *(unsigned int *)(a1 + 56);
  v100 = result;
  if ( *(_DWORD *)(a1 + 56) )
  {
    v133 = 0;
    do
    {
      v26 = *(_QWORD *)(*(_QWORD *)(a1 + 48) + 8 * v133);
      v27 = *(_QWORD *)(v26 + 32);
      if ( v27 == *(_QWORD *)(v26 + 40) + 40LL || !v27 )
        v28 = 0;
      else
        v28 = v27 - 24;
      sub_17CE510((__int64)&v143, v28, 0, 0, 0);
      v29 = *(_QWORD *)(v26 - 24LL * (*(_DWORD *)(v26 + 20) & 0xFFFFFFF));
      v30 = *(_QWORD *)(a1 + 16);
      v140 = 257;
      v124 = sub_1647230(*(_QWORD **)(v30 + 168), 0);
      v138 = 257;
      v31 = sub_15A0680(*(_QWORD *)(*(_QWORD *)(a1 + 16) + 176LL), 0, 0);
      v136 = 257;
      v32 = *(__int64 ***)(*(_QWORD *)(a1 + 16) + 176LL);
      if ( v32 == *(__int64 ***)v29 )
      {
        v33 = v29;
      }
      else if ( *(_BYTE *)(v29 + 16) > 0x10u )
      {
        v142 = 257;
        v87 = sub_15FDBD0(45, v29, (__int64)v32, (__int64)v141, 0);
        if ( v144 )
        {
          v119 = v87;
          v110 = v145;
          sub_157E9D0(v144 + 40, v87);
          v87 = v119;
          v88 = *(_QWORD *)(v119 + 24);
          v89 = *v110;
          *(_QWORD *)(v119 + 32) = v110;
          v89 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v119 + 24) = v89 | v88 & 7;
          *(_QWORD *)(v89 + 8) = v119 + 24;
          *v110 = *v110 & 7 | (v119 + 24);
        }
        v120 = v87;
        sub_164B780(v87, v135);
        v33 = v120;
        if ( v143 )
        {
          v134 = v143;
          sub_1623A60((__int64)&v134, v143, 2);
          v33 = v120;
          v90 = *(_QWORD *)(v120 + 48);
          v91 = v120 + 48;
          if ( v90 )
          {
            sub_161E7C0(v120 + 48, v90);
            v33 = v120;
            v91 = v120 + 48;
          }
          v92 = (unsigned __int8 *)v134;
          *(_QWORD *)(v33 + 48) = v134;
          if ( v92 )
          {
            v121 = v33;
            sub_1623210((__int64)&v134, v92, v91);
            v33 = v121;
          }
        }
      }
      else
      {
        v33 = sub_15A46C0(45, (__int64 ***)v29, v32, 0);
      }
      v34 = sub_12899C0(&v143, v33, v31, (__int64)v137, 0, 0);
      v35 = sub_12AA3B0(&v143, 0x2Eu, v34, v124, (__int64)v139);
      v142 = 257;
      v109 = sub_156E5B0(&v143, v35, (__int64)v141);
      v36 = *(_QWORD *)(a1 + 16);
      v140 = 257;
      v125 = sub_1647230(*(_QWORD **)(v36 + 168), 0);
      v37 = *(_QWORD *)(a1 + 16);
      v138 = 257;
      v38 = sub_15A0680(*(_QWORD *)(v37 + 176), 8, 0);
      v39 = *(_QWORD *)(a1 + 16);
      v136 = 257;
      v40 = *(__int64 ***)(v39 + 176);
      if ( v40 == *(__int64 ***)v29 )
      {
        v41 = v29;
      }
      else if ( *(_BYTE *)(v29 + 16) > 0x10u )
      {
        v142 = 257;
        v81 = sub_15FDBD0(45, v29, (__int64)v40, (__int64)v141, 0);
        if ( v144 )
        {
          v116 = v81;
          v108 = v145;
          sub_157E9D0(v144 + 40, v81);
          v81 = v116;
          v82 = *(_QWORD *)(v116 + 24);
          v83 = *v108;
          *(_QWORD *)(v116 + 32) = v108;
          v83 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v116 + 24) = v83 | v82 & 7;
          *(_QWORD *)(v83 + 8) = v116 + 24;
          *v108 = *v108 & 7 | (v116 + 24);
        }
        v117 = v81;
        sub_164B780(v81, v135);
        v41 = v117;
        if ( v143 )
        {
          v134 = v143;
          sub_1623A60((__int64)&v134, v143, 2);
          v41 = v117;
          v84 = *(_QWORD *)(v117 + 48);
          v85 = v117 + 48;
          if ( v84 )
          {
            sub_161E7C0(v117 + 48, v84);
            v41 = v117;
            v85 = v117 + 48;
          }
          v86 = (unsigned __int8 *)v134;
          *(_QWORD *)(v41 + 48) = v134;
          if ( v86 )
          {
            v118 = v41;
            sub_1623210((__int64)&v134, v86, v85);
            v41 = v118;
          }
        }
      }
      else
      {
        v41 = sub_15A46C0(45, (__int64 ***)v29, v40, 0);
      }
      v42 = sub_12899C0(&v143, v41, v38, (__int64)v137, 0, 0);
      v43 = sub_12AA3B0(&v143, 0x2Eu, v42, v125, (__int64)v139);
      v142 = 257;
      v44 = sub_156E5B0(&v143, v43, (__int64)v141);
      v142 = 257;
      v111 = sub_1647200(*(_QWORD **)(*(_QWORD *)(a1 + 16) + 168LL), 0);
      v140 = 257;
      v126 = sub_15A0680(*(_QWORD *)(*(_QWORD *)(a1 + 16) + 176LL), 24, 0);
      v138 = 257;
      v45 = sub_12AA3B0(&v143, 0x2Du, v29, *(_QWORD *)(*(_QWORD *)(a1 + 16) + 176LL), (__int64)v137);
      v46 = sub_12899C0(&v143, v45, v126, (__int64)v139, 0, 0);
      v47 = sub_12AA3B0(&v143, 0x2Eu, v46, v111, (__int64)v141);
      v142 = 257;
      v48 = sub_156E5B0(&v143, v47, (__int64)v141);
      v142 = 257;
      v49 = sub_12AA3B0(&v143, 0x26u, (__int64)v48, *(_QWORD *)(*(_QWORD *)(a1 + 16) + 176LL), (__int64)v141);
      v142 = 257;
      v101 = v49;
      v127 = sub_12899C0(&v143, (__int64)v44, v49, (__int64)v141, 0, 0);
      v50 = *(_QWORD *)(a1 + 16);
      v142 = 257;
      v51 = sub_1647230(*(_QWORD **)(v50 + 168), 0);
      v52 = *(_QWORD *)(a1 + 16);
      v140 = 257;
      v112 = sub_15A0680(*(_QWORD *)(v52 + 176), 16, 0);
      v53 = *(_QWORD *)(a1 + 16);
      v138 = 257;
      v54 = sub_12AA3B0(&v143, 0x2Du, v29, *(_QWORD *)(v53 + 176), (__int64)v137);
      v55 = sub_12899C0(&v143, v54, v112, (__int64)v139, 0, 0);
      v56 = sub_12AA3B0(&v143, 0x2Eu, v55, v51, (__int64)v141);
      v142 = 257;
      v57 = sub_156E5B0(&v143, v56, (__int64)v141);
      v58 = *(_QWORD *)(a1 + 16);
      v142 = 257;
      v104 = sub_1647200(*(_QWORD **)(v58 + 168), 0);
      v59 = *(_QWORD *)(a1 + 16);
      v140 = 257;
      v113 = sub_15A0680(*(_QWORD *)(v59 + 176), 28, 0);
      v60 = *(_QWORD *)(a1 + 16);
      v138 = 257;
      v61 = sub_12AA3B0(&v143, 0x2Du, v29, *(_QWORD *)(v60 + 176), (__int64)v137);
      v62 = sub_12899C0(&v143, v61, v113, (__int64)v139, 0, 0);
      v63 = sub_12AA3B0(&v143, 0x2Eu, v62, v104, (__int64)v141);
      v142 = 257;
      v64 = sub_156E5B0(&v143, v63, (__int64)v141);
      v142 = 257;
      v65 = sub_12AA3B0(&v143, 0x26u, (__int64)v64, *(_QWORD *)(*(_QWORD *)(a1 + 16) + 176LL), (__int64)v141);
      v142 = 257;
      v114 = sub_12899C0(&v143, (__int64)v57, v65, (__int64)v141, 0, 0);
      v142 = 257;
      v66 = sub_12899C0(&v143, v132, v101, (__int64)v141, 0, 0);
      v67 = *(_QWORD *)(a1 + 24);
      v105 = v66;
      v68 = (__int64 *)sub_1643330(v146);
      v69 = sub_17CFB40(v67, v127, &v143, v68, 8u);
      v70 = *(_BYTE **)(a1 + 32);
      v128 = (_QWORD *)v69;
      v142 = 257;
      v71 = sub_1643330(v146);
      v72 = (_QWORD *)sub_17CEC00(&v143, v71, v70, v105, v141);
      v140 = 257;
      if ( *(_BYTE *)(v132 + 16) > 0x10u || *(_BYTE *)(v105 + 16) > 0x10u )
      {
        v142 = 257;
        v73 = sub_15FB440(13, (__int64 *)v132, v105, (__int64)v141, 0);
        v74 = v73;
        if ( v144 )
        {
          v106 = v73;
          v102 = v145;
          sub_157E9D0(v144 + 40, v73);
          v74 = v106;
          v75 = *(_QWORD *)(v106 + 24);
          v76 = *v102;
          *(_QWORD *)(v106 + 32) = v102;
          v76 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v106 + 24) = v76 | v75 & 7;
          *(_QWORD *)(v76 + 8) = v106 + 24;
          *v102 = *v102 & 7 | (v106 + 24);
        }
        v107 = v74;
        sub_164B780(v74, v139);
        sub_12A86E0(&v143, v107);
        v5 = (__int64 *)v107;
      }
      else
      {
        v5 = (__int64 *)sub_15A2B60((__int64 *)v132, v105, 0, 0, a2, a3, a4);
      }
      sub_15E7430(&v143, v128, 8u, v72, 8u, v5, 0, 0, 0, 0, 0);
      v142 = 257;
      v6 = sub_12899C0(&v143, v131, v65, (__int64)v141, 0, 0);
      v7 = *(_QWORD *)(a1 + 24);
      v103 = v6;
      v8 = (__int64 *)sub_1643330(v146);
      v9 = sub_17CFB40(v7, v114, &v143, v8, 8u);
      v140 = 257;
      v10 = (_QWORD *)v9;
      v142 = 257;
      v11 = sub_1643350(v146);
      v12 = sub_159C470(v11, 64, 0);
      v122 = *(_BYTE **)(a1 + 32);
      v13 = sub_1643330(v146);
      v14 = (_BYTE *)sub_17CEC00(&v143, v13, v122, v12, v139);
      v15 = sub_1643330(v146);
      v16 = sub_17CEC00(&v143, v15, v14, v103, v141);
      v140 = 257;
      v17 = (_QWORD *)v16;
      if ( *(_BYTE *)(v131 + 16) > 0x10u || *(_BYTE *)(v103 + 16) > 0x10u )
      {
        v142 = 257;
        v77 = sub_15FB440(13, (__int64 *)v131, v103, (__int64)v141, 0);
        v78 = v77;
        if ( v144 )
        {
          v129 = v77;
          v115 = v145;
          sub_157E9D0(v144 + 40, v77);
          v78 = v129;
          v79 = *(_QWORD *)(v129 + 24);
          v80 = *v115;
          *(_QWORD *)(v129 + 32) = v115;
          v80 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v129 + 24) = v80 | v79 & 7;
          *(_QWORD *)(v80 + 8) = v129 + 24;
          *v115 = *v115 & 7 | (v129 + 24);
        }
        v130 = v78;
        sub_164B780(v78, v139);
        sub_12A86E0(&v143, v130);
        v18 = (__int64 *)v130;
      }
      else
      {
        v18 = (__int64 *)sub_15A2B60((__int64 *)v131, v103, 0, 0, a2, a3, a4);
      }
      sub_15E7430(&v143, v10, 8u, v17, 8u, v18, 0, 0, 0, 0, 0);
      v19 = *(_QWORD *)(a1 + 24);
      v20 = (__int64 *)sub_1643330(v146);
      v21 = (_QWORD *)sub_17CFB40(v19, (__int64)v109, &v143, v20, 0x10u);
      v142 = 257;
      v22 = sub_1643350(v146);
      v23 = sub_159C470(v22, 192, 0);
      v123 = *(_BYTE **)(a1 + 32);
      v24 = sub_1643330(v146);
      v25 = (_QWORD *)sub_17CEC00(&v143, v24, v123, v23, v141);
      sub_15E7430(&v143, v21, 0x10u, v25, 0x10u, *(__int64 **)(a1 + 40), 0, 0, 0, 0, 0);
      if ( v143 )
        sub_161E7C0((__int64)&v143, v143);
      result = ++v133;
    }
    while ( v133 != v100 );
  }
  return result;
}
