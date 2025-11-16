// Function: sub_36D7E80
// Address: 0x36d7e80
//
__int64 __fastcall sub_36D7E80(__int64 a1, __int64 a2, __m128i a3)
{
  __int64 v5; // rsi
  __int64 *v6; // rax
  __int64 v7; // r13
  __int64 v8; // rdx
  int v9; // eax
  int v10; // eax
  unsigned int v11; // r12d
  int v13; // esi
  bool v14; // al
  __int64 *v15; // rsi
  __int64 v16; // rcx
  int v17; // eax
  __int64 v18; // rax
  unsigned __int64 v19; // r8
  int v20; // eax
  __int64 v21; // rax
  _QWORD *v22; // r13
  unsigned __int64 v23; // r11
  __int64 v24; // r14
  __int64 v25; // rdx
  unsigned __int16 v26; // ax
  __int64 v27; // rdx
  __int64 v28; // rax
  __int64 v29; // rdx
  unsigned __int8 *v30; // rax
  unsigned __int64 v31; // r11
  int v32; // edx
  __int64 v33; // rdx
  unsigned __int16 v34; // ax
  __int64 v35; // rdx
  __int64 v36; // rax
  __int64 v37; // rdx
  unsigned __int64 v38; // rdx
  char v39; // al
  __int64 v40; // rax
  __int64 v41; // r9
  int v42; // edx
  unsigned __int64 v43; // r11
  __int16 v44; // ax
  int v45; // esi
  __int64 v46; // rax
  __int64 v47; // rax
  unsigned __int64 v48; // r14
  __int64 v51; // rsi
  int v52; // edx
  __int64 v53; // rax
  __int64 v54; // rdx
  int v55; // ecx
  __int64 v56; // rcx
  _QWORD *v57; // r13
  unsigned __int64 v58; // r11
  __int64 v59; // rax
  unsigned __int16 v60; // dx
  __int64 v61; // rax
  __int64 v62; // rax
  __int64 v63; // rdx
  int v64; // edx
  __int16 v65; // ax
  _QWORD *v66; // rdi
  unsigned __int64 v67; // rcx
  __int64 v68; // r8
  __int64 v69; // r13
  __int64 v70; // rcx
  __int64 v71; // r8
  __int64 v72; // r9
  __int64 v73; // rsi
  int v74; // eax
  __int64 v75; // rax
  _QWORD *v76; // r14
  unsigned __int64 *v77; // rax
  unsigned __int64 v78; // rcx
  unsigned __int64 v79; // r11
  int v80; // edx
  int v81; // eax
  unsigned __int64 v82; // rt1
  __int64 v83; // rdx
  __int64 v87; // r13
  unsigned __int8 *v88; // rax
  int v89; // edx
  unsigned __int8 *v90; // rax
  __int64 v91; // rcx
  int v92; // edx
  __int64 v96; // rax
  char *v97; // r13
  __int64 v98; // rdx
  unsigned __int16 v99; // ax
  __int64 v100; // rdx
  __int64 v101; // rax
  __int64 v102; // rdx
  __int64 v103; // rax
  __int16 v104; // dx
  __int64 v105; // rax
  char v106; // si
  unsigned __int64 v107; // rax
  __int64 v108; // [rsp+0h] [rbp-100h]
  __int64 v109; // [rsp+10h] [rbp-F0h]
  unsigned __int64 v110; // [rsp+18h] [rbp-E8h]
  unsigned __int64 v111; // [rsp+18h] [rbp-E8h]
  unsigned __int64 v112; // [rsp+18h] [rbp-E8h]
  unsigned __int64 v113; // [rsp+20h] [rbp-E0h]
  unsigned __int64 v114; // [rsp+20h] [rbp-E0h]
  unsigned __int8 *v115; // [rsp+28h] [rbp-D8h]
  int v116; // [rsp+30h] [rbp-D0h]
  unsigned __int64 v117; // [rsp+30h] [rbp-D0h]
  unsigned __int64 v118; // [rsp+38h] [rbp-C8h]
  unsigned __int64 v119; // [rsp+38h] [rbp-C8h]
  __int64 v120; // [rsp+38h] [rbp-C8h]
  unsigned __int8 *v121; // [rsp+38h] [rbp-C8h]
  unsigned __int64 v122; // [rsp+40h] [rbp-C0h]
  int v123; // [rsp+40h] [rbp-C0h]
  unsigned int v124; // [rsp+4Ch] [rbp-B4h]
  __int64 v125; // [rsp+50h] [rbp-B0h] BYREF
  int v126; // [rsp+58h] [rbp-A8h]
  __int64 v127; // [rsp+60h] [rbp-A0h] BYREF
  __int64 v128; // [rsp+68h] [rbp-98h]
  unsigned __int64 v129; // [rsp+70h] [rbp-90h] BYREF
  __int64 v130; // [rsp+78h] [rbp-88h]
  __int64 v131; // [rsp+80h] [rbp-80h]
  __int64 v132; // [rsp+88h] [rbp-78h]
  __int64 v133; // [rsp+90h] [rbp-70h] BYREF
  __int64 v134; // [rsp+98h] [rbp-68h]
  unsigned __int64 v135; // [rsp+A0h] [rbp-60h] BYREF
  __int64 v136; // [rsp+A8h] [rbp-58h]
  unsigned __int8 *v137; // [rsp+B0h] [rbp-50h]
  int v138; // [rsp+B8h] [rbp-48h]
  unsigned __int8 *v139; // [rsp+C0h] [rbp-40h]
  int v140; // [rsp+C8h] [rbp-38h]

  v5 = *(_QWORD *)(a2 + 80);
  v125 = v5;
  if ( v5 )
    sub_B96E90((__int64)&v125, v5, 1);
  v126 = *(_DWORD *)(a2 + 72);
  v6 = *(__int64 **)(a2 + 40);
  v7 = *v6;
  v8 = v6[5];
  v9 = *(_DWORD *)(a2 + 24);
  if ( v9 == 186 )
  {
    v13 = *(_DWORD *)(v7 + 24);
    v14 = *(_DWORD *)(v8 + 24) == 35 || *(_DWORD *)(v8 + 24) == 11;
    if ( v13 == 35 || v13 == 11 )
    {
      if ( !v14 )
      {
        v46 = v7;
        v7 = v8;
        v8 = v46;
      }
    }
    else if ( !v14 )
    {
      goto LABEL_7;
    }
    v47 = *(_QWORD *)(v8 + 96);
    v48 = *(_QWORD *)(v47 + 24);
    if ( *(_DWORD *)(v47 + 32) > 0x40u )
      v48 = *(_QWORD *)v48;
    if ( !v48 || (v48 & (v48 + 1)) != 0 )
      goto LABEL_7;
    if ( ~v48 )
    {
      __asm { tzcnt   r14, r14 }
      _R14 = (int)_R14;
      v51 = (int)_R14;
    }
    else
    {
      v51 = 64;
      _R14 = 64;
    }
    v121 = sub_3400BD0(*(_QWORD *)(a1 + 64), v51, (__int64)&v125, 7, 0, 1u, a3, 0);
    v123 = v52;
    if ( (unsigned int)(*(_DWORD *)(v7 + 24) - 191) > 1 )
      goto LABEL_7;
    v53 = *(_QWORD *)(v7 + 40);
    v54 = *(_QWORD *)(v53 + 40);
    v55 = *(_DWORD *)(v54 + 24);
    if ( v55 != 11 && v55 != 35 )
      goto LABEL_7;
    v56 = *(_QWORD *)(v54 + 96);
    v57 = *(_QWORD **)(v56 + 24);
    if ( *(_DWORD *)(v56 + 32) > 0x40u )
      v57 = (_QWORD *)*v57;
    v58 = *(_QWORD *)v53;
    v124 = *(_DWORD *)(v53 + 8);
    v59 = *(_QWORD *)(v54 + 48) + 16LL * *(unsigned int *)(v53 + 48);
    v60 = *(_WORD *)v59;
    v61 = *(_QWORD *)(v59 + 8);
    LOWORD(v129) = v60;
    v130 = v61;
    if ( v60 )
    {
      if ( v60 == 1 || (unsigned __int16)(v60 - 504) <= 7u )
        goto LABEL_100;
      v63 = 16LL * (v60 - 1);
      v62 = *(_QWORD *)&byte_444C4A0[v63];
      LOBYTE(v63) = byte_444C4A0[v63 + 8];
    }
    else
    {
      v117 = v58;
      v62 = sub_3007260((__int64)&v129);
      v58 = v117;
      v127 = v62;
      v128 = v63;
    }
    v111 = v58;
    v135 = v62;
    LOBYTE(v136) = v63;
    if ( sub_CA1930(&v135) - (__int64)v57 < _R14 )
      goto LABEL_7;
    v115 = sub_3400BD0(*(_QWORD *)(a1 + 64), (__int64)v57, (__int64)&v125, 7, 0, 1u, a3, 0);
    v116 = v64;
    v43 = v111;
    v24 = 16LL * v124;
LABEL_50:
    v65 = *(_WORD *)(*(_QWORD *)(v43 + 48) + v24);
    if ( v65 == 7 )
      goto LABEL_51;
    goto LABEL_71;
  }
  if ( (unsigned int)(v9 - 191) > 1 )
    goto LABEL_7;
  v10 = *(_DWORD *)(v7 + 24);
  if ( v10 == 186 )
  {
    v74 = *(_DWORD *)(v8 + 24);
    if ( v74 != 35 && v74 != 11 )
      goto LABEL_7;
    v75 = *(_QWORD *)(v8 + 96);
    v76 = *(_QWORD **)(v75 + 24);
    if ( *(_DWORD *)(v75 + 32) > 0x40u )
      v76 = (_QWORD *)*v76;
    v77 = *(unsigned __int64 **)(v7 + 40);
    v78 = *v77;
    v79 = v77[5];
    v124 = *((_DWORD *)v77 + 12);
    v80 = *(_DWORD *)(*v77 + 24);
    if ( v80 != 35 && v80 != 11 )
    {
      v124 = *((_DWORD *)v77 + 2);
      v81 = *(_DWORD *)(v79 + 24);
      if ( v81 != 11 && v81 != 35 )
        goto LABEL_7;
      v82 = v79;
      v79 = v78;
      v78 = v82;
    }
    v83 = *(_QWORD *)(v78 + 96);
    _RAX = *(_QWORD *)(v83 + 24);
    if ( *(_DWORD *)(v83 + 32) > 0x40u )
      _RAX = *(_QWORD *)_RAX;
    if ( !_RAX )
      goto LABEL_7;
    if ( (_RAX & (_RAX + 1)) != 0 )
    {
      if ( ((_RAX | (_RAX - 1)) & ((_RAX | (_RAX - 1)) + 1)) != 0 )
        goto LABEL_7;
      __asm { tzcnt   rcx, rax }
      _RAX = ~(_RAX >> _RCX);
      __asm { tzcnt   rdx, rax }
      _RDX = (int)_RDX;
      if ( !_RAX )
        _RDX = 64;
      if ( (unsigned __int64)v76 < (int)_RCX )
        goto LABEL_7;
      v87 = _RDX + (int)_RCX - (_QWORD)v76;
    }
    else
    {
      _RAX = ~_RAX;
      __asm { tzcnt   r13, rax }
      _R13 = (int)_R13;
      if ( !_RAX )
        _R13 = 64;
      v87 = _R13 - (_QWORD)v76;
    }
    v114 = v79;
    v88 = sub_3400BD0(*(_QWORD *)(a1 + 64), (__int64)v76, (__int64)&v125, 7, 0, 1u, a3, 0);
    v116 = v89;
    v115 = v88;
    v90 = sub_3400BD0(*(_QWORD *)(a1 + 64), v87, (__int64)&v125, 7, 0, 1u, a3, 0);
    v41 = v108;
    v43 = v114;
    v121 = v90;
    v91 = 16LL * v124;
    v123 = v92;
    if ( *(_DWORD *)(a2 + 24) != 191 )
    {
      v65 = *(_WORD *)(*(_QWORD *)(v114 + 48) + 16LL * v124);
      if ( v65 == 7 )
        goto LABEL_51;
LABEL_71:
      v45 = 344;
      if ( v65 == 8 )
        goto LABEL_52;
      goto LABEL_7;
    }
    v97 = (char *)v76 + v87;
    v98 = v91 + *(_QWORD *)(v114 + 48);
    v99 = *(_WORD *)v98;
    v100 = *(_QWORD *)(v98 + 8);
    LOWORD(v127) = v99;
    v128 = v100;
    if ( v99 )
    {
      if ( v99 == 1 || (unsigned __int16)(v99 - 504) <= 7u )
        goto LABEL_100;
      v105 = 16LL * (v99 - 1);
      v106 = byte_444C4A0[v105 + 8];
      v107 = *(_QWORD *)&byte_444C4A0[v105];
      LOBYTE(v130) = v106;
      v129 = v107;
    }
    else
    {
      v101 = sub_3007260((__int64)&v127);
      v91 = 16LL * v124;
      v43 = v114;
      v129 = v101;
      v130 = v102;
    }
    v109 = v91;
    v112 = v43;
    v135 = v129;
    LOBYTE(v136) = v130;
    v103 = sub_CA1930(&v135);
    v43 = v112;
    v104 = *(_WORD *)(*(_QWORD *)(v112 + 48) + v109);
    if ( v104 == 7 )
    {
      if ( v97 == (char *)v103 )
        goto LABEL_89;
LABEL_51:
      v45 = 341;
      goto LABEL_52;
    }
    if ( v104 == 8 )
    {
      v45 = 338;
      if ( v97 != (char *)v103 )
        v45 = 344;
      goto LABEL_52;
    }
LABEL_7:
    v11 = 0;
    goto LABEL_8;
  }
  if ( v10 != 190 )
    goto LABEL_7;
  v15 = *(__int64 **)(v7 + 40);
  v16 = v15[5];
  v17 = *(_DWORD *)(v16 + 24);
  if ( v17 != 11 && v17 != 35 )
    goto LABEL_7;
  v18 = *(_QWORD *)(v16 + 96);
  v19 = *(_QWORD *)(v18 + 24);
  if ( *(_DWORD *)(v18 + 32) > 0x40u )
    v19 = **(_QWORD **)(v18 + 24);
  v20 = *(_DWORD *)(v8 + 24);
  if ( v20 != 35 && v20 != 11 )
    goto LABEL_7;
  v21 = *(_QWORD *)(v8 + 96);
  v22 = *(_QWORD **)(v21 + 24);
  if ( *(_DWORD *)(v21 + 32) > 0x40u )
    v22 = (_QWORD *)*v22;
  if ( v19 > (unsigned __int64)v22 )
    goto LABEL_7;
  v23 = *v15;
  v124 = *((_DWORD *)v15 + 2);
  v24 = 16LL * v124;
  v25 = v24 + *(_QWORD *)(*v15 + 48);
  v26 = *(_WORD *)v25;
  v27 = *(_QWORD *)(v25 + 8);
  LOWORD(v133) = v26;
  v134 = v27;
  if ( v26 )
  {
    if ( v26 == 1 || (unsigned __int16)(v26 - 504) <= 7u )
      goto LABEL_100;
    v29 = 16LL * (v26 - 1);
    v28 = *(_QWORD *)&byte_444C4A0[v29];
    LOBYTE(v29) = byte_444C4A0[v29 + 8];
  }
  else
  {
    v118 = v23;
    v122 = v19;
    v28 = sub_3007260((__int64)&v133);
    v23 = v118;
    v19 = v122;
    v131 = v28;
    v132 = v29;
  }
  v113 = v23;
  v119 = v19;
  v135 = v28;
  LOBYTE(v136) = v29;
  if ( (unsigned __int64)v22 >= sub_CA1930(&v135) )
    goto LABEL_7;
  v30 = sub_3400BD0(*(_QWORD *)(a1 + 64), (__int64)v22 - v119, (__int64)&v125, 7, 0, 1u, a3, 0);
  v31 = v113;
  v115 = v30;
  v116 = v32;
  v33 = *(_QWORD *)(v113 + 48) + v24;
  v120 = *(_QWORD *)(a1 + 64);
  v34 = *(_WORD *)v33;
  v35 = *(_QWORD *)(v33 + 8);
  LOWORD(v135) = v34;
  v136 = v35;
  if ( !v34 )
  {
    v36 = sub_3007260((__int64)&v135);
    v31 = v113;
    v133 = v36;
    v134 = v37;
    v38 = v36;
    v39 = v134;
    goto LABEL_29;
  }
  if ( v34 == 1 || (unsigned __int16)(v34 - 504) <= 7u )
LABEL_100:
    BUG();
  v96 = 16LL * (v34 - 1);
  v38 = *(_QWORD *)&byte_444C4A0[v96];
  v39 = byte_444C4A0[v96 + 8];
LABEL_29:
  v110 = v31;
  v135 = v38;
  LOBYTE(v136) = v39;
  v40 = sub_CA1930(&v135);
  v121 = sub_3400BD0(v120, v40 - (_QWORD)v22, (__int64)&v125, 7, 0, 1u, a3, 0);
  v123 = v42;
  v43 = v110;
  if ( *(_DWORD *)(a2 + 24) != 191 )
    goto LABEL_50;
  v44 = *(_WORD *)(*(_QWORD *)(v110 + 48) + 16LL * v124);
  if ( v44 == 7 )
  {
LABEL_89:
    v45 = 335;
    goto LABEL_52;
  }
  if ( v44 != 8 )
    goto LABEL_7;
  v45 = 338;
LABEL_52:
  v66 = *(_QWORD **)(a1 + 64);
  v135 = v43;
  v67 = *(_QWORD *)(a2 + 48);
  LODWORD(v136) = v124;
  v68 = *(unsigned int *)(a2 + 68);
  v137 = v115;
  v138 = v116;
  v139 = v121;
  v140 = v123;
  v69 = sub_33E66D0(v66, v45, (__int64)&v125, v67, v68, v41, &v135, 3);
  sub_34158F0(*(_QWORD *)(a1 + 64), a2, v69, v70, v71, v72);
  sub_3421DB0(v69);
  v73 = a2;
  v11 = 1;
  sub_33ECEA0(*(const __m128i **)(a1 + 64), v73);
LABEL_8:
  if ( v125 )
    sub_B91220((__int64)&v125, v125);
  return v11;
}
