// Function: sub_21C2320
// Address: 0x21c2320
//
__int64 __fastcall sub_21C2320(__int64 a1, __int64 a2, __m128i a3, double a4, __m128i a5)
{
  unsigned int v5; // r13d
  __int64 v7; // rsi
  __int64 *v9; // rax
  __int64 v10; // r13
  __int64 v11; // rdx
  int v12; // eax
  int v13; // eax
  __int64 v14; // rsi
  __int64 v15; // rcx
  int v16; // eax
  int v18; // esi
  bool v19; // al
  int v20; // eax
  __int64 v21; // rax
  _QWORD *v22; // r15
  __int64 v23; // rax
  __int64 v24; // rcx
  __int64 v25; // r13
  unsigned int v26; // r11d
  int v27; // edx
  int v28; // eax
  __int64 v29; // rt0
  __int64 v30; // rdx
  bool v34; // zf
  __int64 v35; // rax
  __int64 v36; // rax
  int v37; // edx
  __int64 v38; // r9
  int v39; // edx
  int v40; // r10d
  unsigned int v41; // r11d
  char v42; // al
  __int16 v43; // si
  __int64 v44; // rcx
  _QWORD *v45; // rdi
  __int64 v46; // r13
  __int64 v47; // rdi
  __int64 v48; // rdx
  __int64 v49; // rcx
  __int64 v50; // r8
  __int64 v51; // r9
  __int64 v52; // rax
  __int64 v53; // rdx
  unsigned __int64 v54; // rax
  __int64 v57; // rdi
  int v58; // edx
  int v59; // r10d
  __int64 v60; // rax
  __int64 v61; // rdx
  int v62; // ecx
  __int64 v63; // rcx
  _QWORD *v64; // rsi
  unsigned int v65; // r11d
  __int64 v66; // rax
  char v67; // di
  __int64 v68; // rax
  unsigned int v69; // eax
  int v70; // r10d
  unsigned int v71; // r11d
  __int64 v72; // rax
  int v73; // edx
  __int64 v74; // r15
  __int64 v75; // rax
  _QWORD *v76; // rcx
  int v77; // eax
  __int64 v78; // rax
  unsigned __int64 v79; // r10
  __int64 v80; // r11
  __int64 v81; // rax
  char v82; // di
  __int64 v83; // rax
  unsigned int v84; // eax
  _QWORD *v85; // rcx
  unsigned __int64 v86; // r10
  unsigned int v87; // r11d
  __int64 v88; // rax
  unsigned __int64 v89; // r10
  int v90; // edx
  __int64 v91; // rax
  char v92; // di
  __int64 v93; // rax
  __int64 v94; // rsi
  unsigned __int64 v95; // r10
  unsigned int v96; // r11d
  __int64 v97; // rax
  int v98; // edx
  unsigned int v102; // eax
  char v103; // al
  unsigned int v104; // [rsp+18h] [rbp-98h]
  unsigned int v105; // [rsp+18h] [rbp-98h]
  unsigned __int64 v106; // [rsp+18h] [rbp-98h]
  unsigned int v107; // [rsp+18h] [rbp-98h]
  unsigned int v108; // [rsp+18h] [rbp-98h]
  __int64 v109; // [rsp+20h] [rbp-90h]
  int v110; // [rsp+20h] [rbp-90h]
  __int64 v111; // [rsp+20h] [rbp-90h]
  unsigned int v112; // [rsp+20h] [rbp-90h]
  int v113; // [rsp+2Ch] [rbp-84h]
  unsigned int v114; // [rsp+2Ch] [rbp-84h]
  int v115; // [rsp+2Ch] [rbp-84h]
  __int64 v116; // [rsp+30h] [rbp-80h]
  unsigned __int64 v117; // [rsp+30h] [rbp-80h]
  __int64 v118; // [rsp+38h] [rbp-78h]
  unsigned int v119; // [rsp+38h] [rbp-78h]
  _QWORD *v120; // [rsp+38h] [rbp-78h]
  unsigned __int64 v121; // [rsp+38h] [rbp-78h]
  __int64 v122; // [rsp+40h] [rbp-70h] BYREF
  int v123; // [rsp+48h] [rbp-68h]
  __int64 v124; // [rsp+50h] [rbp-60h] BYREF
  __int64 v125; // [rsp+58h] [rbp-58h]
  __int64 v126; // [rsp+60h] [rbp-50h]
  int v127; // [rsp+68h] [rbp-48h]
  __int64 v128; // [rsp+70h] [rbp-40h]
  int v129; // [rsp+78h] [rbp-38h]

  v5 = 0;
  if ( *(_DWORD *)(*(_QWORD *)(a1 + 480) + 252LL) > 0x45u )
    return v5;
  v7 = *(_QWORD *)(a2 + 72);
  v122 = v7;
  if ( v7 )
    sub_1623A60((__int64)&v122, v7, 2);
  v123 = *(_DWORD *)(a2 + 64);
  v9 = *(__int64 **)(a2 + 32);
  v10 = *v9;
  v11 = v9[5];
  v12 = *(unsigned __int16 *)(a2 + 24);
  if ( v12 == 118 )
  {
    v18 = *(unsigned __int16 *)(v10 + 24);
    v19 = *(_WORD *)(v11 + 24) == 32 || *(_WORD *)(v11 + 24) == 10;
    if ( v18 == 32 || v18 == 10 )
    {
      if ( !v19 )
      {
        v52 = v10;
        v10 = v11;
        v11 = v52;
      }
    }
    else if ( !v19 )
    {
      goto LABEL_10;
    }
    v53 = *(_QWORD *)(v11 + 88);
    v54 = *(_QWORD *)(v53 + 24);
    if ( *(_DWORD *)(v53 + 32) > 0x40u )
      v54 = *(_QWORD *)v54;
    if ( !v54 || (v54 & (v54 + 1)) != 0 )
      goto LABEL_10;
    _RAX = ~v54;
    __asm { tzcnt   r15, rax }
    _R15 = (int)_R15;
    v57 = *(_QWORD *)(a1 + 272);
    if ( !_RAX )
      _R15 = 64;
    v118 = sub_1D38BB0(v57, _R15, (__int64)&v122, 5, 0, 1, a3, a4, a5, 0);
    v59 = v58;
    if ( (unsigned int)*(unsigned __int16 *)(v10 + 24) - 123 > 1 )
      goto LABEL_10;
    v60 = *(_QWORD *)(v10 + 32);
    v61 = *(_QWORD *)(v60 + 40);
    v62 = *(unsigned __int16 *)(v61 + 24);
    if ( v62 != 10 && v62 != 32 )
      goto LABEL_10;
    v63 = *(_QWORD *)(v61 + 88);
    v64 = *(_QWORD **)(v63 + 24);
    if ( *(_DWORD *)(v63 + 32) > 0x40u )
      v64 = (_QWORD *)*v64;
    v25 = *(_QWORD *)v60;
    v65 = *(_DWORD *)(v60 + 8);
    v66 = *(_QWORD *)(v61 + 40) + 16LL * *(unsigned int *)(v60 + 48);
    v67 = *(_BYTE *)v66;
    v68 = *(_QWORD *)(v66 + 8);
    LOBYTE(v124) = v67;
    v125 = v68;
    if ( v67 )
    {
      v69 = sub_21BD810(v67);
    }
    else
    {
      v112 = v65;
      v115 = v59;
      v69 = sub_1F58D40((__int64)&v124);
      v71 = v112;
      v70 = v115;
    }
    v105 = v71;
    v110 = v70;
    if ( _R15 > v69 - (unsigned __int64)v64 )
      goto LABEL_10;
    v72 = sub_1D38BB0(*(_QWORD *)(a1 + 272), (__int64)v64, (__int64)&v122, 5, 0, 1, a3, a4, a5, 0);
    v40 = v110;
    v116 = v72;
    v41 = v105;
    v113 = v73;
    v74 = 16LL * v105;
    goto LABEL_68;
  }
  if ( (unsigned int)(v12 - 123) > 1 )
    goto LABEL_10;
  v13 = *(unsigned __int16 *)(v10 + 24);
  if ( v13 != 118 )
  {
    if ( v13 != 122 )
      goto LABEL_10;
    v14 = *(_QWORD *)(v10 + 32);
    v15 = *(_QWORD *)(v14 + 40);
    v16 = *(unsigned __int16 *)(v15 + 24);
    if ( v16 != 32 && v16 != 10 )
      goto LABEL_10;
    v75 = *(_QWORD *)(v15 + 88);
    v76 = *(_QWORD **)(v75 + 24);
    if ( *(_DWORD *)(v75 + 32) > 0x40u )
      v76 = (_QWORD *)*v76;
    v77 = *(unsigned __int16 *)(v11 + 24);
    if ( v77 != 10 && v77 != 32 )
      goto LABEL_10;
    v78 = *(_QWORD *)(v11 + 88);
    v79 = *(_QWORD *)(v78 + 24);
    if ( *(_DWORD *)(v78 + 32) > 0x40u )
      v79 = **(_QWORD **)(v78 + 24);
    if ( (unsigned __int64)v76 > v79 )
      goto LABEL_10;
    v25 = *(_QWORD *)v14;
    v80 = *(unsigned int *)(v14 + 8);
    v74 = 16 * v80;
    v81 = 16 * v80 + *(_QWORD *)(*(_QWORD *)v14 + 40LL);
    v82 = *(_BYTE *)v81;
    v83 = *(_QWORD *)(v81 + 8);
    LOBYTE(v124) = v82;
    v125 = v83;
    if ( v82 )
    {
      v84 = sub_21BD810(v82);
    }
    else
    {
      v114 = v80;
      v117 = v79;
      v120 = v76;
      v84 = sub_1F58D40((__int64)&v124);
      v87 = v114;
      v86 = v117;
      v85 = v120;
    }
    v119 = v87;
    if ( v86 >= v84 )
      goto LABEL_10;
    v106 = v86;
    v88 = sub_1D38BB0(*(_QWORD *)(a1 + 272), v86 - (_QWORD)v85, (__int64)&v122, 5, 0, 1, a3, a4, a5, 0);
    v89 = v106;
    v116 = v88;
    v113 = v90;
    v111 = *(_QWORD *)(a1 + 272);
    v91 = v74 + *(_QWORD *)(v25 + 40);
    v92 = *(_BYTE *)v91;
    v93 = *(_QWORD *)(v91 + 8);
    LOBYTE(v124) = v92;
    v125 = v93;
    if ( v92 )
    {
      v94 = (unsigned int)sub_21BD810(v92);
    }
    else
    {
      v108 = v119;
      v121 = v89;
      v102 = sub_1F58D40((__int64)&v124);
      v96 = v108;
      v95 = v121;
      v94 = v102;
    }
    v107 = v96;
    v97 = sub_1D38BB0(v111, v94 - v95, (__int64)&v122, 5, 0, 1, a3, a4, a5, 0);
    v41 = v107;
    v40 = v98;
    v118 = v97;
    if ( *(_WORD *)(a2 + 24) != 123 )
    {
LABEL_68:
      v42 = *(_BYTE *)(*(_QWORD *)(v25 + 40) + v74);
      goto LABEL_35;
    }
    v103 = *(_BYTE *)(*(_QWORD *)(v25 + 40) + v74);
    if ( v103 != 5 )
    {
      if ( v103 == 6 )
      {
        v43 = 155;
        goto LABEL_37;
      }
LABEL_10:
      v5 = 0;
      goto LABEL_11;
    }
LABEL_88:
    v43 = 152;
    goto LABEL_37;
  }
  v20 = *(unsigned __int16 *)(v11 + 24);
  if ( v20 != 10 && v20 != 32 )
    goto LABEL_10;
  v21 = *(_QWORD *)(v11 + 88);
  v22 = *(_QWORD **)(v21 + 24);
  if ( *(_DWORD *)(v21 + 32) > 0x40u )
    v22 = (_QWORD *)*v22;
  v23 = *(_QWORD *)(v10 + 32);
  v24 = *(_QWORD *)v23;
  v25 = *(_QWORD *)(v23 + 40);
  v26 = *(_DWORD *)(v23 + 48);
  v27 = *(unsigned __int16 *)(*(_QWORD *)v23 + 24LL);
  if ( v27 != 10 && v27 != 32 )
  {
    v26 = *(_DWORD *)(v23 + 8);
    v28 = *(unsigned __int16 *)(v25 + 24);
    if ( v28 != 32 && v28 != 10 )
      goto LABEL_10;
    v29 = v25;
    v25 = v24;
    v24 = v29;
  }
  v30 = *(_QWORD *)(v24 + 88);
  _RAX = *(_QWORD *)(v30 + 24);
  if ( *(_DWORD *)(v30 + 32) > 0x40u )
    _RAX = *(_QWORD *)_RAX;
  if ( !_RAX )
    goto LABEL_10;
  if ( (_RAX & (_RAX + 1)) != 0 )
  {
    if ( ((_RAX | (_RAX - 1)) & ((_RAX | (_RAX - 1)) + 1)) != 0 )
      goto LABEL_10;
    __asm { tzcnt   rcx, rax }
    _RAX = ~(_RAX >> _RCX);
    __asm { tzcnt   rdx, rax }
    _RDX = (int)_RDX;
    if ( !_RAX )
      _RDX = 64;
    if ( (unsigned __int64)v22 < (int)_RCX )
      goto LABEL_10;
    v109 = (int)_RCX - (_QWORD)v22 + _RDX;
  }
  else
  {
    _RAX = ~_RAX;
    __asm { tzcnt   rdx, rax }
    v34 = _RAX == 0;
    v35 = 64;
    if ( !v34 )
      v35 = (int)_RDX;
    v109 = v35 - (_QWORD)v22;
  }
  v104 = v26;
  v36 = sub_1D38BB0(*(_QWORD *)(a1 + 272), (__int64)v22, (__int64)&v122, 5, 0, 1, a3, a4, a5, 0);
  v113 = v37;
  v116 = v36;
  v118 = sub_1D38BB0(*(_QWORD *)(a1 + 272), v109, (__int64)&v122, 5, 0, 1, a3, a4, a5, 0);
  v40 = v39;
  v41 = v104;
  v42 = *(_BYTE *)(*(_QWORD *)(v25 + 40) + 16LL * v104);
  if ( *(_WORD *)(a2 + 24) == 123 )
  {
    if ( v42 == 6 )
    {
      v43 = 155;
      if ( (_QWORD *)((char *)v22 + v109) != (_QWORD *)64 )
        v43 = 161;
      goto LABEL_37;
    }
    if ( v42 != 5 )
      goto LABEL_10;
    if ( (_QWORD *)((char *)v22 + v109) != (_QWORD *)32 )
      goto LABEL_36;
    goto LABEL_88;
  }
LABEL_35:
  if ( v42 != 5 )
  {
    v43 = 161;
    if ( v42 == 6 )
      goto LABEL_37;
    goto LABEL_10;
  }
LABEL_36:
  v43 = 158;
LABEL_37:
  v44 = *(_QWORD *)(a2 + 40);
  LODWORD(v125) = v41;
  v45 = *(_QWORD **)(a1 + 272);
  v126 = v116;
  v129 = v40;
  v127 = v113;
  v124 = v25;
  v128 = v118;
  v46 = sub_1D23DE0(v45, v43, (__int64)&v122, v44, *(_DWORD *)(a2 + 60), v38, &v124, 3);
  sub_1D444E0(*(_QWORD *)(a1 + 272), a2, v46);
  v47 = v46;
  v5 = 1;
  sub_1D49010(v47);
  sub_1D2DC70(*(const __m128i **)(a1 + 272), a2, v48, v49, v50, v51);
LABEL_11:
  if ( v122 )
    sub_161E7C0((__int64)&v122, v122);
  return v5;
}
