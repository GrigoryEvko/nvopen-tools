// Function: sub_187EA30
// Address: 0x187ea30
//
__int64 __fastcall sub_187EA30(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, double a5, double a6, double a7)
{
  __int64 *v8; // r12
  __int64 v10; // r13
  __int64 v11; // r15
  __int64 v13; // rax
  unsigned __int8 *v14; // rsi
  __int64 v15; // rax
  __int64 **v16; // rdx
  __int64 **v17; // rsi
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // r9
  __int64 **v21; // rsi
  unsigned __int64 v22; // rdi
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // r9
  __int64 v26; // r13
  __int64 **v27; // r8
  __int64 v28; // r10
  int v29; // eax
  __int64 *v30; // rax
  unsigned __int64 v31; // rax
  __int64 v32; // rax
  __int64 v33; // rsi
  __int64 v34; // rdx
  __int64 v35; // rcx
  __int64 v36; // r15
  __int64 v37; // rcx
  __int64 v38; // r15
  __int64 v39; // rdi
  _QWORD *v40; // r9
  __int64 v41; // r15
  __int64 v42; // rax
  unsigned __int8 *v43; // rsi
  __int64 v44; // rax
  unsigned __int8 *v45; // rsi
  __int64 v46; // r13
  __int64 v47; // rax
  unsigned __int8 *v48; // rsi
  __int64 v49; // rbx
  __int64 v50; // rax
  __int64 v51; // r14
  __int64 *v52; // rbx
  __int64 v53; // rax
  __int64 v54; // rcx
  __int64 v55; // rax
  __int64 v56; // rcx
  __int64 v57; // r8
  __int64 v58; // r9
  __int64 v59; // rcx
  __int64 v60; // r8
  __int64 v61; // r9
  unsigned __int8 *v62; // rsi
  __int64 v63; // rax
  __int64 v64; // r9
  __int64 *v65; // r13
  __int64 v66; // rcx
  __int64 v67; // rax
  __int64 v68; // rax
  __int64 v69; // rsi
  __int64 v70; // rax
  _QWORD *v71; // rax
  _QWORD *v72; // rax
  __int64 v73; // rax
  __int64 *v74; // r15
  __int64 v75; // rax
  __int64 v76; // rcx
  __int64 v77; // rsi
  __int64 v78; // rdx
  unsigned __int8 *v79; // rsi
  __int64 v80; // rax
  __int64 v81; // rsi
  __int64 v82; // rax
  __int64 v83; // rax
  _QWORD *v84; // rax
  __int64 v85; // r9
  __int64 v86; // r10
  __int64 v87; // r11
  __int64 v88; // rdx
  __int64 v89; // rax
  unsigned __int64 v90; // rax
  __int64 v91; // rdx
  __int64 v92; // rax
  __int64 v93; // rdx
  __int64 v94; // rcx
  __int64 v95; // r9
  __int64 v96; // r15
  __int64 v97; // r13
  __int64 v98; // r12
  __int64 v99; // rax
  __int64 v100; // r8
  unsigned int v101; // esi
  __int64 v102; // rdx
  __int64 v103; // rax
  __int64 v104; // rdx
  __int64 v105; // rax
  __int64 v106; // rax
  unsigned __int8 *v107; // rsi
  __int64 v108; // rax
  __int64 v109; // rax
  __int64 v110; // [rsp+0h] [rbp-150h]
  __int64 v111; // [rsp+8h] [rbp-148h]
  __int64 v112; // [rsp+8h] [rbp-148h]
  __int64 v113; // [rsp+8h] [rbp-148h]
  __int64 v114; // [rsp+8h] [rbp-148h]
  __int64 **v115; // [rsp+10h] [rbp-140h]
  __int64 v116; // [rsp+10h] [rbp-140h]
  _QWORD *v117; // [rsp+10h] [rbp-140h]
  __int64 v118; // [rsp+10h] [rbp-140h]
  __int64 v119; // [rsp+10h] [rbp-140h]
  __int64 v120; // [rsp+18h] [rbp-138h]
  __int64 v121; // [rsp+18h] [rbp-138h]
  __int64 v122; // [rsp+18h] [rbp-138h]
  __int64 v123; // [rsp+18h] [rbp-138h]
  __int64 *v124; // [rsp+18h] [rbp-138h]
  __int64 *v125; // [rsp+18h] [rbp-138h]
  _QWORD *v126; // [rsp+18h] [rbp-138h]
  __int64 v127; // [rsp+18h] [rbp-138h]
  __int64 v128; // [rsp+18h] [rbp-138h]
  _QWORD *v129; // [rsp+28h] [rbp-128h]
  __int64 *v130; // [rsp+28h] [rbp-128h]
  __int64 v131[2]; // [rsp+40h] [rbp-110h] BYREF
  __int16 v132; // [rsp+50h] [rbp-100h]
  unsigned __int8 *v133[2]; // [rsp+60h] [rbp-F0h] BYREF
  __int16 v134; // [rsp+70h] [rbp-E0h]
  unsigned __int8 *v135; // [rsp+80h] [rbp-D0h] BYREF
  __int64 v136; // [rsp+88h] [rbp-C8h]
  __int64 *v137; // [rsp+90h] [rbp-C0h]
  __int64 v138; // [rsp+98h] [rbp-B8h]
  __int64 v139; // [rsp+A0h] [rbp-B0h]
  int v140; // [rsp+A8h] [rbp-A8h]
  __int64 v141; // [rsp+B0h] [rbp-A0h]
  __int64 v142; // [rsp+B8h] [rbp-98h]
  unsigned __int8 *v143; // [rsp+D0h] [rbp-80h] BYREF
  __int64 v144; // [rsp+D8h] [rbp-78h]
  __int64 v145; // [rsp+E0h] [rbp-70h]
  __int64 v146; // [rsp+E8h] [rbp-68h]
  __int64 v147; // [rsp+F0h] [rbp-60h]
  int v148; // [rsp+F8h] [rbp-58h]
  __int64 v149; // [rsp+100h] [rbp-50h]
  __int64 v150; // [rsp+108h] [rbp-48h]

  v8 = a1;
  v10 = *(_QWORD *)(a3 - 24LL * (*(_DWORD *)(a3 + 20) & 0xFFFFFFF));
  v11 = sub_1632FA0(*a1);
  if ( (unsigned __int8)sub_187BC10(a2, v11, v10, 0) )
    return sub_159C4F0(*(__int64 **)*a1);
  v129 = *(_QWORD **)(a3 + 40);
  v13 = sub_16498A0(a3);
  v14 = *(unsigned __int8 **)(a3 + 48);
  v135 = 0;
  v138 = v13;
  v15 = *(_QWORD *)(a3 + 40);
  v139 = 0;
  v136 = v15;
  v137 = (__int64 *)(a3 + 24);
  v140 = 0;
  v141 = 0;
  v142 = 0;
  v143 = v14;
  if ( v14 )
  {
    sub_1623A60((__int64)&v143, (__int64)v14, 2);
    v135 = v143;
    if ( v143 )
      sub_1623210((__int64)&v143, v143, (__int64)&v135);
  }
  v16 = (__int64 **)a1[12];
  v134 = 257;
  v17 = *(__int64 ***)v10;
  if ( v16 != *(__int64 ***)v10 )
  {
    if ( *(_BYTE *)(v10 + 16) > 0x10u )
    {
      LOWORD(v145) = 257;
      v80 = sub_15FDBD0(45, v10, (__int64)v16, (__int64)&v143, 0);
      v10 = v80;
      if ( v136 )
      {
        v125 = v137;
        sub_157E9D0(v136 + 40, v80);
        v81 = *v125;
        v82 = *(_QWORD *)(v10 + 24) & 7LL;
        *(_QWORD *)(v10 + 32) = v125;
        v81 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v10 + 24) = v81 | v82;
        *(_QWORD *)(v81 + 8) = v10 + 24;
        *v125 = *v125 & 7 | (v10 + 24);
      }
      sub_164B780(v10, (__int64 *)v133);
      sub_12A86E0((__int64 *)&v135, v10);
      v17 = (__int64 **)a1[12];
    }
    else
    {
      v18 = sub_15A46C0(45, (__int64 ***)v10, v16, 0);
      v17 = (__int64 **)a1[12];
      v10 = v18;
    }
  }
  v19 = sub_15A4180(*(_QWORD *)(a4 + 8), v17, 0);
  if ( *(_DWORD *)a4 == 3 )
  {
    LOWORD(v145) = 257;
    v38 = sub_12AA0C0((__int64 *)&v135, 0x20u, (_BYTE *)v10, v19, (__int64)&v143);
    goto LABEL_43;
  }
  v134 = 257;
  if ( *(_BYTE *)(v10 + 16) > 0x10u || *(_BYTE *)(v19 + 16) > 0x10u )
  {
    LOWORD(v145) = 257;
    v63 = sub_15FB440(13, (__int64 *)v10, v19, (__int64)&v143, 0);
    v64 = v63;
    if ( v136 )
    {
      v65 = v137;
      v122 = v63;
      sub_157E9D0(v136 + 40, v63);
      v64 = v122;
      v66 = *v65;
      v67 = *(_QWORD *)(v122 + 24);
      *(_QWORD *)(v122 + 32) = v65;
      v66 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v122 + 24) = v66 | v67 & 7;
      *(_QWORD *)(v66 + 8) = v122 + 24;
      *v65 = *v65 & 7 | (v122 + 24);
    }
    v123 = v64;
    sub_164B780(v64, (__int64 *)v133);
    sub_12A86E0((__int64 *)&v135, v123);
    v20 = v123;
  }
  else
  {
    v20 = sub_15A2B60((__int64 *)v10, v19, 0, 0, a5, a6, a7);
  }
  v21 = (__int64 **)a1[12];
  v22 = *(_QWORD *)(a4 + 16);
  v120 = v20;
  v134 = 257;
  v23 = sub_15A3CB0(v22, v21, 0);
  if ( *(_BYTE *)(v120 + 16) > 0x10u || *(_BYTE *)(v23 + 16) > 0x10u )
  {
    LOWORD(v145) = 257;
    v71 = (_QWORD *)sub_15FB440(24, (__int64 *)v120, v23, (__int64)&v143, 0);
    v72 = sub_17CF870((__int64 *)&v135, v71, (__int64 *)v133);
    v25 = v120;
    v26 = (__int64)v72;
  }
  else
  {
    v24 = sub_15A2D80((__int64 *)v120, v23, 0, a5, a6, a7);
    v25 = v120;
    v26 = v24;
  }
  v27 = (__int64 **)v8[12];
  v28 = *(_QWORD *)(a4 + 16);
  v111 = v25;
  v134 = 257;
  v115 = v27;
  v121 = v28;
  v29 = sub_15A9520(v11, 0);
  v30 = (__int64 *)sub_159C470(v8[6], (unsigned int)(8 * v29), 0);
  v31 = sub_15A2B60(v30, v121, 0, 0, a5, a6, a7);
  v32 = sub_15A3CB0(v31, v115, 0);
  v33 = v32;
  if ( *(_BYTE *)(v111 + 16) > 0x10u || *(_BYTE *)(v32 + 16) > 0x10u )
  {
    LOWORD(v145) = 257;
    v68 = sub_15FB440(23, (__int64 *)v111, v32, (__int64)&v143, 0);
    v36 = v68;
    if ( v136 )
    {
      v124 = v137;
      sub_157E9D0(v136 + 40, v68);
      v69 = *v124;
      v70 = *(_QWORD *)(v36 + 24) & 7LL;
      *(_QWORD *)(v36 + 32) = v124;
      v69 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v36 + 24) = v69 | v70;
      *(_QWORD *)(v69 + 8) = v36 + 24;
      *v124 = *v124 & 7 | (v36 + 24);
    }
    sub_164B780(v36, (__int64 *)v133);
    v33 = v36;
    sub_12A86E0((__int64 *)&v135, v36);
  }
  else
  {
    v36 = sub_15A2D50((__int64 *)v111, v32, 0, 0, a5, a6, a7);
  }
  v134 = 257;
  if ( *(_BYTE *)(v36 + 16) > 0x10u )
    goto LABEL_55;
  if ( sub_1593BB0(v36, v33, v34, v35) )
    goto LABEL_23;
  if ( *(_BYTE *)(v26 + 16) > 0x10u )
  {
LABEL_55:
    LOWORD(v145) = 257;
    v73 = sub_15FB440(27, (__int64 *)v26, v36, (__int64)&v143, 0);
    v26 = v73;
    if ( v136 )
    {
      v74 = v137;
      sub_157E9D0(v136 + 40, v73);
      v75 = *(_QWORD *)(v26 + 24);
      v76 = *v74;
      *(_QWORD *)(v26 + 32) = v74;
      v76 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v26 + 24) = v76 | v75 & 7;
      *(_QWORD *)(v76 + 8) = v26 + 24;
      *v74 = *v74 & 7 | (v26 + 24);
    }
    sub_164B780(v26, (__int64 *)v133);
    if ( v135 )
    {
      v131[0] = (__int64)v135;
      sub_1623A60((__int64)v131, (__int64)v135, 2);
      v77 = *(_QWORD *)(v26 + 48);
      v78 = v26 + 48;
      if ( v77 )
      {
        sub_161E7C0(v26 + 48, v77);
        v78 = v26 + 48;
      }
      v79 = (unsigned __int8 *)v131[0];
      *(_QWORD *)(v26 + 48) = v131[0];
      if ( v79 )
        sub_1623210((__int64)v131, v79, v78);
    }
  }
  else
  {
    v26 = sub_15A2D10((__int64 *)v26, v36, a5, a6, a7);
  }
LABEL_23:
  v37 = *(_QWORD *)(a4 + 24);
  LOWORD(v145) = 257;
  v38 = sub_12AA0C0((__int64 *)&v135, 0x25u, (_BYTE *)v26, v37, (__int64)&v143);
  if ( *(_DWORD *)a4 != 4 )
  {
    v39 = *(_QWORD *)(a3 + 8);
    if ( v39 )
    {
      if ( !*(_QWORD *)(v39 + 8) )
      {
        v40 = sub_1648700(v39);
        if ( *((_BYTE *)v40 + 16) == 26 )
        {
          v83 = *(_QWORD *)(a3 + 32);
          if ( v83 != *(_QWORD *)(a3 + 40) + 40LL && v83 && v40 == (_QWORD *)(v83 - 24) )
          {
            v126 = v40;
            LOWORD(v145) = 257;
            v112 = sub_157FBF0(v129, (__int64 *)(a3 + 24), (__int64)&v143);
            v116 = (__int64)v126;
            v127 = *(v126 - 6);
            v84 = sub_1648A60(56, 3u);
            v85 = v116;
            v86 = v112;
            v87 = (__int64)v84;
            if ( v84 )
            {
              v110 = v116;
              v117 = v84;
              sub_15F83E0((__int64)v84, v112, v127, v38, 0);
              v85 = v110;
              v86 = v112;
              v87 = (__int64)v117;
            }
            v88 = *(_QWORD *)(v85 + 48);
            if ( v88 || *(__int16 *)(v85 + 18) < 0 )
            {
              v113 = v87;
              v118 = v86;
              v89 = sub_1625790(v85, 2);
              v87 = v113;
              v86 = v118;
              v88 = v89;
            }
            v114 = v86;
            v119 = v87;
            sub_1625C10(v87, 2, v88);
            v90 = sub_157EBA0((__int64)v129);
            sub_1AA6530(v90, v119, v91);
            v92 = sub_157F280(v127);
            if ( v92 != v93 )
            {
              v128 = v26;
              v96 = v93;
              v97 = (__int64)v129;
              v130 = v8;
              v98 = v92;
              do
              {
                v99 = 0x17FFFFFFE8LL;
                v100 = *(_BYTE *)(v98 + 23) & 0x40;
                v101 = *(_DWORD *)(v98 + 20) & 0xFFFFFFF;
                if ( v101 )
                {
                  v95 = v98 - 24LL * v101;
                  v102 = 24LL * *(unsigned int *)(v98 + 56) + 8;
                  v103 = 0;
                  do
                  {
                    v94 = v98 - 24LL * v101;
                    if ( (_BYTE)v100 )
                      v94 = *(_QWORD *)(v98 - 8);
                    if ( v114 == *(_QWORD *)(v94 + v102) )
                    {
                      v99 = 24 * v103;
                      goto LABEL_83;
                    }
                    ++v103;
                    v102 += 8;
                  }
                  while ( v101 != (_DWORD)v103 );
                  v99 = 0x17FFFFFFE8LL;
                }
LABEL_83:
                if ( (_BYTE)v100 )
                {
                  v104 = *(_QWORD *)(v98 - 8);
                }
                else
                {
                  v94 = 24LL * v101;
                  v104 = v98 - v94;
                }
                sub_1704F80(v98, *(_QWORD *)(v104 + v99), v97, v94, v100, v95);
                v105 = *(_QWORD *)(v98 + 32);
                if ( !v105 )
                  BUG();
                v98 = 0;
                if ( *(_BYTE *)(v105 - 8) == 77 )
                  v98 = v105 - 24;
              }
              while ( v96 != v98 );
              v26 = v128;
              v8 = v130;
            }
            v106 = sub_16498A0(a3);
            v107 = *(unsigned __int8 **)(a3 + 48);
            v143 = 0;
            v146 = v106;
            v108 = *(_QWORD *)(a3 + 40);
            v147 = 0;
            v144 = v108;
            v148 = 0;
            v149 = 0;
            v150 = 0;
            v145 = a3 + 24;
            v133[0] = v107;
            if ( v107 )
            {
              sub_1623A60((__int64)v133, (__int64)v107, 2);
              if ( v143 )
                sub_161E7C0((__int64)&v143, (__int64)v143);
              v143 = v133[0];
              if ( v133[0] )
                sub_1623210((__int64)v133, v133[0], (__int64)&v143);
            }
            v109 = sub_187BEA0(v8, (__int64 *)&v143, a4, (__int64 *)v26, a5, a6, a7);
            v62 = v143;
            v38 = v109;
            if ( !v143 )
              goto LABEL_43;
LABEL_42:
            sub_161E7C0((__int64)&v143, (__int64)v62);
            goto LABEL_43;
          }
        }
      }
    }
    v41 = sub_1AA92B0(v38, a3, 0, 0, 0, 0);
    v42 = sub_16498A0(v41);
    v143 = 0;
    v146 = v42;
    v147 = 0;
    v148 = 0;
    v149 = 0;
    v150 = 0;
    v144 = *(_QWORD *)(v41 + 40);
    v145 = v41 + 24;
    v43 = *(unsigned __int8 **)(v41 + 48);
    v133[0] = v43;
    if ( v43 )
    {
      sub_1623A60((__int64)v133, (__int64)v43, 2);
      if ( v143 )
        sub_161E7C0((__int64)&v143, (__int64)v143);
      v143 = v133[0];
      if ( v133[0] )
        sub_1623210((__int64)v133, v133[0], (__int64)&v143);
    }
    v44 = sub_187BEA0(v8, (__int64 *)&v143, a4, (__int64 *)v26, a5, a6, a7);
    v45 = *(unsigned __int8 **)(a3 + 48);
    v46 = v44;
    v47 = *(_QWORD *)(a3 + 40);
    v133[0] = v45;
    v136 = v47;
    v137 = (__int64 *)(a3 + 24);
    if ( v45 )
    {
      sub_1623A60((__int64)v133, (__int64)v45, 2);
      v48 = v135;
      if ( !v135 )
        goto LABEL_35;
    }
    else
    {
      v48 = v135;
      if ( !v135 )
        goto LABEL_37;
    }
    sub_161E7C0((__int64)&v135, (__int64)v48);
LABEL_35:
    v135 = v133[0];
    if ( v133[0] )
      sub_1623210((__int64)v133, v133[0], (__int64)&v135);
LABEL_37:
    v49 = v8[5];
    v132 = 257;
    v134 = 257;
    v50 = sub_1648B60(64);
    v38 = v50;
    if ( v50 )
    {
      v51 = v50;
      sub_15F1EA0(v50, v49, 53, 0, 0, 0);
      *(_DWORD *)(v38 + 56) = 2;
      sub_164B780(v38, (__int64 *)v133);
      sub_1648880(v38, *(_DWORD *)(v38 + 56), 1);
    }
    else
    {
      v51 = 0;
    }
    if ( v136 )
    {
      v52 = v137;
      sub_157E9D0(v136 + 40, v38);
      v53 = *(_QWORD *)(v38 + 24);
      v54 = *v52;
      *(_QWORD *)(v38 + 32) = v52;
      v54 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v38 + 24) = v54 | v53 & 7;
      *(_QWORD *)(v54 + 8) = v38 + 24;
      *v52 = *v52 & 7 | (v38 + 24);
    }
    sub_164B780(v51, v131);
    sub_12A86E0((__int64 *)&v135, v38);
    v55 = sub_159C470(v8[5], 0, 0);
    sub_1704F80(v38, v55, (__int64)v129, v56, v57, v58);
    sub_1704F80(v38, v46, v144, v59, v60, v61);
    v62 = v143;
    if ( !v143 )
      goto LABEL_43;
    goto LABEL_42;
  }
LABEL_43:
  if ( v135 )
    sub_161E7C0((__int64)&v135, (__int64)v135);
  return v38;
}
