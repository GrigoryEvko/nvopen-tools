// Function: sub_1806810
// Address: 0x1806810
//
unsigned __int64 __fastcall sub_1806810(
        __int64 a1,
        unsigned __int64 a2,
        __int64 a3,
        __int64 a4,
        unsigned int a5,
        unsigned __int8 a6,
        double a7,
        double a8,
        double a9,
        __int64 a10,
        char a11,
        unsigned int a12)
{
  __int64 v13; // r14
  _QWORD *v16; // rax
  unsigned __int8 *v17; // rsi
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rax
  __int64 v21; // rdx
  unsigned __int8 *v22; // r13
  unsigned int v24; // esi
  __int64 *v25; // r13
  __int64 v26; // rbx
  __int64 v27; // rdx
  __int64 v28; // rcx
  __int64 v29; // rbx
  __int64 ***v30; // r13
  unsigned __int64 *v31; // rbx
  __int64 **v32; // rax
  unsigned __int64 v33; // rcx
  __int64 v34; // rsi
  __int64 v35; // rdx
  unsigned __int8 *v36; // rsi
  __int64 v37; // rbx
  __int64 v38; // rax
  __int64 v39; // rbx
  __int64 v40; // rax
  unsigned __int8 *v41; // rsi
  __int64 v42; // rax
  unsigned __int8 *v43; // rsi
  int v44; // ecx
  __int64 v45; // rdi
  __int64 v46; // rax
  __int64 v47; // r14
  __int64 **v48; // rsi
  __int64 v49; // rax
  __int64 v50; // r13
  _QWORD *v51; // rax
  unsigned __int8 *v52; // rsi
  __int64 v53; // rdx
  __int64 v54; // rbx
  __int64 v55; // rdx
  unsigned __int64 result; // rax
  unsigned __int8 **v57; // r13
  unsigned __int8 *v58; // rsi
  __int64 v59; // rdx
  __int64 v60; // rsi
  unsigned __int8 *v61; // rsi
  __int64 v62; // r13
  __int64 v63; // rcx
  _QWORD *v64; // rax
  __int64 v65; // r14
  _QWORD *v66; // rax
  _QWORD *v67; // rax
  __int64 v68; // rdx
  _QWORD *v69; // r10
  __int64 v70; // rcx
  __int64 v71; // rdx
  __int64 v72; // rax
  __int64 v73; // rax
  __int64 v74; // rdx
  __int64 v75; // rax
  __int64 v76; // rdx
  __int64 v77; // rdi
  __int64 v78; // rax
  __int64 v79; // rax
  __int64 v80; // r13
  __int64 v81; // rax
  _BYTE *v82; // r13
  __int64 v83; // rdi
  __int64 v84; // rax
  __int64 v85; // r13
  __int64 v86; // rax
  __int64 v87; // rax
  __int64 v88; // rsi
  __int64 v89; // rax
  _QWORD *v90; // rax
  __int64 v91; // [rsp+0h] [rbp-160h]
  __int64 v92; // [rsp+0h] [rbp-160h]
  __int64 v93; // [rsp+8h] [rbp-158h]
  int v95; // [rsp+18h] [rbp-148h]
  __int64 v96; // [rsp+18h] [rbp-148h]
  __int64 v97; // [rsp+18h] [rbp-148h]
  unsigned int v98; // [rsp+20h] [rbp-140h]
  int v99; // [rsp+24h] [rbp-13Ch]
  __int64 v101; // [rsp+30h] [rbp-130h]
  _QWORD *v102; // [rsp+30h] [rbp-130h]
  __int64 *v103; // [rsp+30h] [rbp-130h]
  __int64 v104; // [rsp+40h] [rbp-120h] BYREF
  __int64 v105; // [rsp+48h] [rbp-118h] BYREF
  unsigned __int8 *v106; // [rsp+50h] [rbp-110h] BYREF
  __int64 v107; // [rsp+58h] [rbp-108h]
  __int64 v108; // [rsp+60h] [rbp-100h]
  unsigned __int8 *v109[2]; // [rsp+70h] [rbp-F0h] BYREF
  __int16 v110; // [rsp+80h] [rbp-E0h]
  unsigned __int8 *v111; // [rsp+90h] [rbp-D0h] BYREF
  __int64 v112; // [rsp+98h] [rbp-C8h]
  unsigned __int64 *v113; // [rsp+A0h] [rbp-C0h]
  _QWORD *v114; // [rsp+A8h] [rbp-B8h]
  __int64 v115; // [rsp+B0h] [rbp-B0h]
  int v116; // [rsp+B8h] [rbp-A8h]
  __int64 v117; // [rsp+C0h] [rbp-A0h]
  __int64 v118; // [rsp+C8h] [rbp-98h]
  unsigned __int8 *v119[2]; // [rsp+E0h] [rbp-80h] BYREF
  __int64 v120; // [rsp+F0h] [rbp-70h]
  _QWORD *v121; // [rsp+F8h] [rbp-68h]
  __int64 v122; // [rsp+100h] [rbp-60h]
  int v123; // [rsp+108h] [rbp-58h]
  __int64 v124; // [rsp+110h] [rbp-50h]
  __int64 v125; // [rsp+118h] [rbp-48h]

  v13 = a3;
  v95 = *(_DWORD *)(a1 + 208);
  v16 = (_QWORD *)sub_16498A0(a3);
  v17 = *(unsigned __int8 **)(v13 + 48);
  v111 = 0;
  v114 = v16;
  v18 = *(_QWORD *)(v13 + 40);
  v115 = 0;
  v112 = v18;
  v116 = 0;
  v117 = 0;
  v118 = 0;
  v113 = (unsigned __int64 *)(v13 + 24);
  v119[0] = v17;
  if ( v17 )
  {
    sub_1623A60((__int64)v119, (__int64)v17, 2);
    v111 = v119[0];
    if ( v119[0] )
      sub_1623210((__int64)v119, v119[0], (__int64)&v111);
  }
  v19 = *(_QWORD *)(a1 + 232);
  LOWORD(v120) = 257;
  v20 = sub_12A95D0((__int64 *)&v111, a4, v19, (__int64)v119);
  v21 = 32;
  v104 = v20;
  v22 = (unsigned __int8 *)v20;
  v98 = a5 >> 3;
  __asm { tzcnt   eax, eax }
  if ( a5 > 7 )
    v21 = _EAX;
  v99 = a6;
  v93 = v21;
  if ( a11 )
  {
    LOWORD(v120) = 257;
    if ( a12 )
    {
      v109[0] = v22;
      v75 = sub_1643350(v114);
      v109[1] = (unsigned __int8 *)sub_159C470(v75, a12, 0);
      v76 = *(_QWORD *)(a1 + 8 * (v93 + 10LL * a6 + 63));
      result = sub_1285290((__int64 *)&v111, *(_QWORD *)(v76 + 24), v76, (int)v109, 2, (__int64)v119, 0);
    }
    else
    {
      v59 = *(_QWORD *)(a1 + 8 * (v21 + 10LL * a6 + 58));
      result = sub_1285290((__int64 *)&v111, *(_QWORD *)(v59 + 24), v59, (int)&v104, 1, (__int64)v119, 0);
    }
    goto LABEL_47;
  }
  if ( v95 == 12 )
  {
    LOWORD(v120) = 257;
    v79 = sub_15A0680(*(_QWORD *)v22, -1073741825, 0);
    v110 = 257;
    v80 = sub_1281C00((__int64 *)&v111, (__int64)v22, v79, (__int64)v119);
    v104 = v80;
    v81 = sub_15A0680(*(_QWORD *)v80, 29, 0);
    if ( *(_BYTE *)(v80 + 16) > 0x10u || *(_BYTE *)(v81 + 16) > 0x10u )
    {
      LOWORD(v120) = 257;
      v90 = (_QWORD *)sub_15FB440(24, (__int64 *)v80, v81, (__int64)v119, 0);
      v82 = sub_17CF870((__int64 *)&v111, v90, (__int64 *)v109);
    }
    else
    {
      v82 = (_BYTE *)sub_15A2D80((__int64 *)v80, v81, 0, a7, a8, a9);
    }
    v83 = *(_QWORD *)(a1 + 232);
    LOWORD(v120) = 257;
    v84 = sub_15A0680(v83, 4, 0);
    v85 = sub_12AA0C0((__int64 *)&v111, 0x20u, v82, v84, (__int64)v119);
    v119[0] = *(unsigned __int8 **)(a1 + 160);
    v86 = sub_161BE60(v119, 1u, 0x186A0u);
    v13 = sub_1AA92B0(v85, v13, 0, v86, 0, 0);
    sub_17050D0((__int64 *)&v111, v13);
  }
  v24 = a5 >> *(_DWORD *)(a1 + 240);
  if ( v24 < 8 )
    v24 = 8;
  v25 = (__int64 *)sub_1644900(*(_QWORD **)(a1 + 160), v24);
  v26 = sub_1646BA0(v25, 0);
  v91 = sub_1804A80(a1, v104, (__int64 *)&v111, a7, a8, a9);
  v96 = sub_15A06D0((__int64 **)v25, v104, v27, v28);
  LOWORD(v120) = 257;
  v110 = 257;
  v29 = sub_12AA3B0((__int64 *)&v111, 0x2Eu, v91, v26, (__int64)v109);
  v30 = (__int64 ***)sub_1648A60(64, 1u);
  if ( v30 )
    sub_15F9210((__int64)v30, *(_QWORD *)(*(_QWORD *)v29 + 24LL), v29, 0, 0, 0);
  if ( v112 )
  {
    v31 = v113;
    sub_157E9D0(v112 + 40, (__int64)v30);
    v32 = v30[3];
    v33 = *v31;
    v30[4] = (__int64 **)v31;
    v33 &= 0xFFFFFFFFFFFFFFF8LL;
    v30[3] = (__int64 **)(v33 | (unsigned __int8)v32 & 7);
    *(_QWORD *)(v33 + 8) = v30 + 3;
    *v31 = *v31 & 7 | (unsigned __int64)(v30 + 3);
  }
  sub_164B780((__int64)v30, (__int64 *)v119);
  if ( v111 )
  {
    v106 = v111;
    sub_1623A60((__int64)&v106, (__int64)v111, 2);
    v34 = (__int64)v30[6];
    v35 = (__int64)(v30 + 6);
    if ( v34 )
    {
      sub_161E7C0((__int64)(v30 + 6), v34);
      v35 = (__int64)(v30 + 6);
    }
    v36 = v106;
    v30[6] = (__int64 **)v106;
    if ( v36 )
      sub_1623210((__int64)&v106, v36, v35);
  }
  LOWORD(v120) = 257;
  v37 = sub_12AA0C0((__int64 *)&v111, 0x21u, v30, v96, (__int64)v119);
  if ( !byte_4FA89E0 && a5 >= (unsigned __int64)(8LL << *(_DWORD *)(a1 + 240)) )
  {
    v50 = sub_1AA92B0(v37, v13, *(_BYTE *)(a1 + 229) ^ 1u, 0, 0, 0);
    goto LABEL_33;
  }
  v119[0] = *(unsigned __int8 **)(a1 + 160);
  v38 = sub_161BE60(v119, 1u, 0x186A0u);
  v39 = sub_1AA92B0(v37, v13, 0, v38, 0, 0);
  v40 = sub_15F4DF0(v39, 0);
  v41 = *(unsigned __int8 **)(v39 + 48);
  v97 = v40;
  v42 = *(_QWORD *)(v39 + 40);
  v119[0] = v41;
  v112 = v42;
  v113 = (unsigned __int64 *)(v39 + 24);
  if ( v41 )
  {
    sub_1623A60((__int64)v119, (__int64)v41, 2);
    v43 = v111;
    if ( !v111 )
      goto LABEL_24;
    goto LABEL_23;
  }
  v43 = v111;
  if ( v111 )
  {
LABEL_23:
    sub_161E7C0((__int64)&v111, (__int64)v43);
LABEL_24:
    v111 = v119[0];
    if ( v119[0] )
      sub_1623210((__int64)v119, v119[0], (__int64)&v111);
  }
  v44 = *(_DWORD *)(a1 + 240);
  v45 = *(_QWORD *)(a1 + 232);
  LOWORD(v120) = 257;
  v46 = sub_15A0680(v45, (1LL << v44) - 1, 0);
  v47 = sub_1281C00((__int64 *)&v111, v104, v46, (__int64)v119);
  if ( a5 > 0xF )
  {
    v77 = *(_QWORD *)(a1 + 232);
    LOWORD(v120) = 257;
    v78 = sub_15A0680(v77, v98 - 1, 0);
    v47 = sub_12899C0((__int64 *)&v111, v47, v78, (__int64)v119, 0, 0);
  }
  v110 = 257;
  v48 = *v30;
  if ( *v30 != *(__int64 ***)v47 )
  {
    if ( *(_BYTE *)(v47 + 16) > 0x10u )
    {
      LOWORD(v120) = 257;
      v87 = sub_15FE0A0((_QWORD *)v47, (__int64)v48, 0, (__int64)v119, 0);
      v47 = v87;
      if ( v112 )
      {
        v103 = (__int64 *)v113;
        sub_157E9D0(v112 + 40, v87);
        v88 = *v103;
        v89 = *(_QWORD *)(v47 + 24) & 7LL;
        *(_QWORD *)(v47 + 32) = v103;
        v88 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v47 + 24) = v88 | v89;
        *(_QWORD *)(v88 + 8) = v47 + 24;
        *v103 = *v103 & 7 | (v47 + 24);
      }
      sub_164B780(v47, (__int64 *)v109);
      sub_12A86E0((__int64 *)&v111, v47);
    }
    else
    {
      v47 = sub_15A4750((__int64 ***)v47, v48, 0);
    }
  }
  LOWORD(v120) = 257;
  v49 = sub_12AA0C0((__int64 *)&v111, 0x27u, (_BYTE *)v47, (__int64)v30, (__int64)v119);
  v101 = v49;
  if ( *(_BYTE *)(a1 + 229) )
  {
    v50 = sub_1AA92B0(v49, v39, 0, 0, 0, 0);
  }
  else
  {
    v62 = *(_QWORD *)(a1 + 160);
    v63 = *(_QWORD *)(v97 + 56);
    LOWORD(v120) = 257;
    v92 = v63;
    v64 = (_QWORD *)sub_22077B0(64);
    v65 = (__int64)v64;
    if ( v64 )
      sub_157FB60(v64, v62, (__int64)v119, v92, v97);
    v66 = sub_1648A60(56, 0);
    v50 = (__int64)v66;
    if ( v66 )
      sub_15F82E0((__int64)v66, *(_QWORD *)(a1 + 160), v65);
    v67 = sub_1648A60(56, 3u);
    v69 = v67;
    if ( v67 )
    {
      v70 = v101;
      v102 = v67;
      sub_15F83E0((__int64)v67, v65, v97, v70, 0);
      v69 = v102;
    }
    sub_1AA6530(v39, v69, v68);
  }
LABEL_33:
  v105 = v104;
  v51 = (_QWORD *)sub_16498A0(v50);
  v119[0] = 0;
  v121 = v51;
  v122 = 0;
  v123 = 0;
  v124 = 0;
  v125 = 0;
  v119[1] = *(unsigned __int8 **)(v50 + 40);
  v120 = v50 + 24;
  v52 = *(unsigned __int8 **)(v50 + 48);
  v109[0] = v52;
  if ( v52 )
  {
    sub_1623A60((__int64)v109, (__int64)v52, 2);
    if ( v119[0] )
      sub_161E7C0((__int64)v119, (__int64)v119[0]);
    v119[0] = v109[0];
    if ( v109[0] )
      sub_1623210((__int64)v109, v109[0], (__int64)v119);
  }
  if ( !a12 )
  {
    if ( !a10 )
    {
      v110 = 257;
      v53 = *(_QWORD *)(a1 + 8 * (v93 + 10LL * v99 + 38));
      v54 = sub_1285290((__int64 *)v119, *(_QWORD *)(v53 + 24), v53, (int)&v105, 1, (__int64)v109, 0);
      goto LABEL_41;
    }
    v110 = 257;
    v106 = (unsigned __int8 *)v105;
    v71 = *(_QWORD *)(16LL * v99 + a1 + 624);
    v107 = a10;
    goto LABEL_68;
  }
  v72 = sub_1643350(v121);
  v73 = sub_159C470(v72, a12, 0);
  if ( !a10 )
  {
    v107 = v73;
    v110 = 257;
    v106 = (unsigned __int8 *)v105;
    v71 = *(_QWORD *)(a1 + 8 * (v93 + 10LL * v99 + 43));
LABEL_68:
    v54 = sub_1285290((__int64 *)v119, *(_QWORD *)(v71 + 24), v71, (int)&v106, 2, (__int64)v109, 0);
    goto LABEL_41;
  }
  v110 = 257;
  v107 = a10;
  v106 = (unsigned __int8 *)v105;
  v108 = v73;
  v74 = *(_QWORD *)(a1 + 16LL * v99 + 632);
  v54 = sub_1285290((__int64 *)v119, *(_QWORD *)(v74 + 24), v74, (int)&v106, 3, (__int64)v109, 0);
LABEL_41:
  v55 = *(_QWORD *)(a1 + 712);
  v110 = 257;
  sub_1285290((__int64 *)v119, *(_QWORD *)(*(_QWORD *)v55 + 24LL), v55, 0, 0, (__int64)v109, 0);
  if ( v119[0] )
    sub_161E7C0((__int64)v119, (__int64)v119[0]);
  result = a2;
  v57 = (unsigned __int8 **)(v54 + 48);
  v58 = *(unsigned __int8 **)(a2 + 48);
  v119[0] = v58;
  if ( !v58 )
  {
    if ( v57 == v119 )
      goto LABEL_47;
    v60 = *(_QWORD *)(v54 + 48);
    if ( !v60 )
      goto LABEL_47;
LABEL_57:
    result = sub_161E7C0(v54 + 48, v60);
    goto LABEL_58;
  }
  result = sub_1623A60((__int64)v119, (__int64)v58, 2);
  if ( v57 == v119 )
  {
    if ( v119[0] )
      result = sub_161E7C0((__int64)v119, (__int64)v119[0]);
    goto LABEL_47;
  }
  v60 = *(_QWORD *)(v54 + 48);
  if ( v60 )
    goto LABEL_57;
LABEL_58:
  v61 = v119[0];
  *(unsigned __int8 **)(v54 + 48) = v119[0];
  if ( v61 )
    result = sub_1623210((__int64)v119, v61, v54 + 48);
LABEL_47:
  if ( v111 )
    return sub_161E7C0((__int64)&v111, (__int64)v111);
  return result;
}
