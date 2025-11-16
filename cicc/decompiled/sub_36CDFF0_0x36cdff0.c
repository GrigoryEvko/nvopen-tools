// Function: sub_36CDFF0
// Address: 0x36cdff0
//
__int64 __fastcall sub_36CDFF0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v7; // r13
  __int64 v8; // r8
  __int64 v10; // r9
  __int64 v11; // rdx
  _QWORD *v12; // rax
  int v13; // edi
  int v14; // esi
  _QWORD *v15; // rcx
  __int64 v16; // r11
  _QWORD *v17; // rbx
  unsigned __int64 v18; // rdx
  int v19; // ecx
  int v20; // esi
  __int64 v21; // rax
  _QWORD *v22; // rcx
  _QWORD *v23; // rbx
  _QWORD *v24; // rax
  __int64 v25; // rbx
  __int64 v26; // rax
  __int64 v27; // rbx
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // r8
  __int64 result; // rax
  __int64 v32; // r8
  int v33; // r10d
  _QWORD *v34; // rcx
  __int64 v35; // rsi
  unsigned __int64 v36; // rdx
  int v37; // eax
  __int64 v38; // r8
  int v39; // r11d
  _QWORD *v40; // rcx
  __int64 v41; // rdx
  unsigned __int64 v42; // rsi
  _QWORD *v43; // rax
  unsigned __int64 v44; // rdx
  __int64 v45; // rax
  __int64 v46; // rax
  __int64 v47; // rdx
  __int64 v48; // rax
  _QWORD *v49; // rcx
  _QWORD *v50; // rdx
  __int64 v51; // rax
  unsigned __int64 v52; // rcx
  __int64 v53; // rax
  unsigned __int64 v54; // rdx
  __int64 v55; // rax
  __int64 v56; // rax
  unsigned int v57; // eax
  __int64 v58; // rax
  __int64 v59; // rax
  int v60; // edx
  unsigned __int64 v61; // rbx
  unsigned __int64 v62; // rdi
  __int64 v63; // r8
  __int64 v64; // rcx
  __int64 v65; // rsi
  _QWORD *v66; // rax
  _QWORD *v67; // rdx
  __int64 v68; // rax
  unsigned __int64 v69; // rax
  __int64 v70; // rdx
  __int64 v71; // rax
  unsigned __int64 v72; // rax
  __int64 v73; // rdx
  __int64 v74; // [rsp+8h] [rbp-208h]
  unsigned int v75; // [rsp+14h] [rbp-1FCh]
  _QWORD *v76; // [rsp+18h] [rbp-1F8h]
  __int64 v77; // [rsp+18h] [rbp-1F8h]
  unsigned __int64 v78; // [rsp+18h] [rbp-1F8h]
  __int64 v79; // [rsp+20h] [rbp-1F0h]
  __int64 v80; // [rsp+20h] [rbp-1F0h]
  int v81; // [rsp+20h] [rbp-1F0h]
  _QWORD *v82; // [rsp+20h] [rbp-1F0h]
  unsigned __int64 v83; // [rsp+20h] [rbp-1F0h]
  unsigned int v84; // [rsp+28h] [rbp-1E8h]
  int v85; // [rsp+28h] [rbp-1E8h]
  unsigned int v86; // [rsp+28h] [rbp-1E8h]
  __int64 v87; // [rsp+28h] [rbp-1E8h]
  unsigned int v88; // [rsp+28h] [rbp-1E8h]
  _QWORD *v89; // [rsp+28h] [rbp-1E8h]
  _QWORD *v90; // [rsp+28h] [rbp-1E8h]
  int v91; // [rsp+28h] [rbp-1E8h]
  __int64 v92; // [rsp+30h] [rbp-1E0h]
  __int64 v93; // [rsp+30h] [rbp-1E0h]
  int v94; // [rsp+30h] [rbp-1E0h]
  unsigned int v95; // [rsp+30h] [rbp-1E0h]
  __int64 v96; // [rsp+38h] [rbp-1D8h]
  __int64 v97; // [rsp+38h] [rbp-1D8h]
  __int64 v98; // [rsp+38h] [rbp-1D8h]
  __int64 v99; // [rsp+38h] [rbp-1D8h]
  unsigned __int64 v100; // [rsp+38h] [rbp-1D8h]
  int v101; // [rsp+38h] [rbp-1D8h]
  unsigned int v102; // [rsp+40h] [rbp-1D0h]
  __int64 v103; // [rsp+40h] [rbp-1D0h]
  __int64 v104; // [rsp+40h] [rbp-1D0h]
  unsigned __int64 v105; // [rsp+40h] [rbp-1D0h]
  __int64 v106; // [rsp+40h] [rbp-1D0h]
  int v107; // [rsp+40h] [rbp-1D0h]
  unsigned __int64 v109; // [rsp+50h] [rbp-1C0h] BYREF
  unsigned int v110; // [rsp+58h] [rbp-1B8h]
  unsigned __int64 v111; // [rsp+60h] [rbp-1B0h] BYREF
  unsigned int v112; // [rsp+68h] [rbp-1A8h]
  _QWORD *v113; // [rsp+70h] [rbp-1A0h] BYREF
  __int64 v114; // [rsp+78h] [rbp-198h]
  _QWORD v115[8]; // [rsp+80h] [rbp-190h] BYREF
  _BYTE *v116; // [rsp+C0h] [rbp-150h] BYREF
  __int64 v117; // [rsp+C8h] [rbp-148h]
  _BYTE v118[128]; // [rsp+D0h] [rbp-140h] BYREF
  _BYTE *v119; // [rsp+150h] [rbp-C0h] BYREF
  __int64 v120; // [rsp+158h] [rbp-B8h]
  _BYTE v121[176]; // [rsp+160h] [rbp-B0h] BYREF

  if ( a3 == -1 || a3 == 0xBFFFFFFFFFFFFFFELL || a4 == -1 || a4 == 0xBFFFFFFFFFFFFFFELL )
    return 1;
  v7 = a1;
  v8 = a3;
  v10 = *(_DWORD *)(a1 + 4) & 0x7FFFFFF;
  v75 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  v11 = 32 * (1LL - (unsigned int)v10);
  v12 = (_QWORD *)(a1 + v11);
  if ( (_DWORD)v10 != v75 )
  {
    v13 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
    if ( (unsigned int)v10 <= v75 )
      v13 = v10;
    v102 = v13;
LABEL_9:
    v14 = 0;
    v116 = v118;
    v15 = v118;
    v74 = v102;
    v16 = 32LL * v102 - 32;
    v117 = 0x1000000000LL;
    v17 = (_QWORD *)(v7 + v16 + v11);
    v18 = v16 >> 5;
    if ( (unsigned __int64)v16 > 0x200 )
    {
      v76 = v12;
      v80 = v8;
      v88 = v10;
      sub_C8D5F0((__int64)&v116, v118, v18, 8u, v8, v10);
      v14 = v117;
      v12 = v76;
      v8 = v80;
      v10 = v88;
      v16 = 32LL * v102 - 32;
      v15 = &v116[8 * (unsigned int)v117];
      v18 = v16 >> 5;
    }
    if ( v17 != v12 )
    {
      do
      {
        if ( v15 )
          *v15 = *v12;
        v12 += 4;
        ++v15;
      }
      while ( v17 != v12 );
      v14 = v117;
    }
    v19 = *(_DWORD *)(a2 + 4);
    LODWORD(v117) = v18 + v14;
    v120 = 0x1000000000LL;
    v20 = 0;
    v21 = 1LL - (v19 & 0x7FFFFFF);
    v119 = v121;
    v22 = v121;
    v21 *= 32;
    v23 = (_QWORD *)(a2 + v21);
    v24 = (_QWORD *)(a2 + v16 + v21);
    if ( (unsigned __int64)v16 > 0x200 )
    {
      v77 = v8;
      v81 = v10;
      v89 = v24;
      v94 = v18;
      sub_C8D5F0((__int64)&v119, v121, v18, 8u, v8, v10);
      v20 = v120;
      v8 = v77;
      LODWORD(v10) = v81;
      v24 = v89;
      LODWORD(v18) = v94;
      v22 = &v119[8 * (unsigned int)v120];
    }
    if ( v24 != v23 )
    {
      do
      {
        if ( v22 )
          *v22 = *v23;
        v23 += 4;
        ++v22;
      }
      while ( v24 != v23 );
      v20 = v120;
    }
    v25 = (unsigned int)v117;
    v79 = v8;
    v84 = v10;
    LODWORD(v120) = v20 + v18;
    v96 = (__int64)v116;
    v26 = sub_BB5290(v7);
    v27 = sub_B4DC50(v26, v96, v25);
    v92 = (__int64)v119;
    v97 = (unsigned int)v120;
    v28 = sub_BB5290(a2);
    v29 = sub_B4DC50(v28, v92, v97);
    v30 = v79;
    if ( v27 == v29 && *(_BYTE *)(v27 + 8) == 15 )
    {
      v57 = v75;
      if ( v84 >= v75 )
        v57 = v84;
      if ( v57 - v102 == 1 )
      {
        if ( v84 > v75 || (v7 = a2, v84 >= v75) )
          v30 = a4;
        v106 = v30;
        v58 = sub_BCCE00(*(_QWORD **)v27, 0x40u);
        v59 = sub_ACD640(v58, 0, 0);
        v60 = *(_DWORD *)(v7 + 4);
        v115[0] = v59;
        v113 = v115;
        v115[1] = *(_QWORD *)(v7 + 32 * (v74 - (v60 & 0x7FFFFFF)));
        v114 = 0x800000002LL;
        v61 = sub_AE54E0(a5, v27, v115, 2);
        v111 = v106 & 0x3FFFFFFFFFFFFFFFLL;
        LOBYTE(v112) = (v106 & 0x4000000000000000LL) != 0;
        if ( sub_CA1930(&v111) <= v61 )
        {
          result = 0;
          if ( v113 != v115 )
          {
            _libc_free((unsigned __int64)v113);
            result = 0;
          }
          v62 = (unsigned __int64)v119;
          if ( v119 == v121 )
            goto LABEL_81;
          goto LABEL_80;
        }
        if ( v113 != v115 )
          _libc_free((unsigned __int64)v113);
      }
    }
    goto LABEL_24;
  }
  v102 = *(_DWORD *)(a1 + 4) & 0x7FFFFFF;
  if ( (unsigned int)v10 <= 2 )
    goto LABEL_9;
  v32 = a1 - 32;
  v33 = 0;
  v117 = 0x1000000000LL;
  v34 = v118;
  v35 = -32 - v11;
  v116 = v118;
  v36 = (-32 - v11) >> 5;
  if ( v35 > 512 )
  {
    v90 = v12;
    v95 = v10;
    v107 = v36;
    sub_C8D5F0((__int64)&v116, v118, v36, 8u, v32, v10);
    v33 = v117;
    v12 = v90;
    v10 = v95;
    v32 = a1 - 32;
    LODWORD(v36) = v107;
    v34 = &v116[8 * (unsigned int)v117];
  }
  if ( (_QWORD *)v32 != v12 )
  {
    do
    {
      if ( v34 )
        *v34 = *v12;
      v12 += 4;
      ++v34;
    }
    while ( (_QWORD *)v32 != v12 );
    v33 = v117;
  }
  v37 = *(_DWORD *)(a2 + 4);
  v38 = a2 - 32;
  v39 = 0;
  LODWORD(v117) = v33 + v36;
  v120 = 0x1000000000LL;
  v40 = v121;
  v119 = v121;
  v41 = 32 * (1LL - (v37 & 0x7FFFFFF));
  v42 = -32 - v41;
  v43 = (_QWORD *)(a2 + v41);
  v44 = (-32 - v41) >> 5;
  if ( v42 > 0x200 )
  {
    v82 = v43;
    v91 = v10;
    v101 = v44;
    sub_C8D5F0((__int64)&v119, v121, v44, 8u, v38, v10);
    v39 = v120;
    v43 = v82;
    LODWORD(v10) = v91;
    v38 = a2 - 32;
    LODWORD(v44) = v101;
    v40 = &v119[8 * (unsigned int)v120];
  }
  if ( (_QWORD *)v38 != v43 )
  {
    do
    {
      if ( v40 )
        *v40 = *v43;
      v43 += 4;
      ++v40;
    }
    while ( (_QWORD *)v38 != v43 );
    v39 = v120;
  }
  LODWORD(v120) = v39 + v44;
  v85 = v10;
  v98 = (__int64)v116;
  v103 = (unsigned int)v117;
  v45 = sub_BB5290(a1);
  v93 = sub_B4DC50(v45, v98, v103);
  v99 = (__int64)v119;
  v104 = (unsigned int)v120;
  v46 = sub_BB5290(a2);
  if ( v93 != sub_B4DC50(v46, v99, v104)
    || *(_BYTE *)(v93 + 8) != 15
    || (v47 = *(_QWORD *)(a1 + 32 * ((unsigned int)(v85 - 1) - (unsigned __int64)(*(_DWORD *)(a1 + 4) & 0x7FFFFFF))),
        *(_BYTE *)v47 != 17)
    || (v48 = *(_QWORD *)(a2 + 32 * ((unsigned int)(v85 - 1) - (unsigned __int64)(*(_DWORD *)(a2 + 4) & 0x7FFFFFF))),
        *(_BYTE *)v48 != 17) )
  {
LABEL_24:
    if ( v119 != v121 )
      _libc_free((unsigned __int64)v119);
    if ( v116 != v118 )
      _libc_free((unsigned __int64)v116);
    return 1;
  }
  v49 = *(_QWORD **)(v47 + 24);
  if ( *(_DWORD *)(v47 + 32) > 0x40u )
    v49 = (_QWORD *)*v49;
  v50 = *(_QWORD **)(v48 + 24);
  if ( *(_DWORD *)(v48 + 32) > 0x40u )
    v50 = (_QWORD *)*v50;
  v86 = (unsigned int)v50;
  v51 = 16LL * (unsigned int)v49 + sub_AE4AC0(a5, v93) + 24;
  v52 = *(_QWORD *)v51;
  LOBYTE(v114) = *(_BYTE *)(v51 + 8);
  v113 = (_QWORD *)v52;
  v105 = sub_CA1930(&v113);
  v53 = 16LL * v86 + sub_AE4AC0(a5, v93) + 24;
  v54 = *(_QWORD *)v53;
  LOBYTE(v53) = *(_BYTE *)(v53 + 8);
  v113 = (_QWORD *)v54;
  LOBYTE(v114) = v53;
  v100 = sub_CA1930(&v113);
  v55 = *(_QWORD *)(*(_QWORD *)(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF)) + 8LL);
  if ( (unsigned int)*(unsigned __int8 *)(v55 + 8) - 17 <= 1 )
    v55 = **(_QWORD **)(v55 + 16);
  v110 = sub_AE2980(a5, *(_DWORD *)(v55 + 8) >> 8)[1];
  if ( v110 > 0x40 )
    sub_C43690((__int64)&v109, 0, 0);
  else
    v109 = 0;
  v56 = *(_QWORD *)(*(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)) + 8LL);
  if ( (unsigned int)*(unsigned __int8 *)(v56 + 8) - 17 <= 1 )
    v56 = **(_QWORD **)(v56 + 16);
  v112 = sub_AE2980(a5, *(_DWORD *)(v56 + 8) >> 8)[1];
  if ( v112 > 0x40 )
    sub_C43690((__int64)&v111, 0, 0);
  else
    v111 = 0;
  v87 = sub_BB5290(a1);
  if ( v87 == sub_BB5290(a2)
    && (unsigned __int8)sub_BB6360(a1, a5, (__int64)&v109, 0, 0)
    && (unsigned __int8)sub_BB6360(a2, a5, (__int64)&v111, 0, 0) )
  {
    if ( v110 > 0x40 )
    {
      v63 = *(_QWORD *)v109;
    }
    else if ( v110 )
    {
      v63 = (__int64)(v109 << (64 - (unsigned __int8)v110)) >> (64 - (unsigned __int8)v110);
    }
    else
    {
      v63 = 0;
    }
    v105 = v63;
    if ( v112 > 0x40 )
    {
      v64 = *(_QWORD *)v111;
    }
    else if ( v112 )
    {
      v64 = (__int64)(v111 << (64 - (unsigned __int8)v112)) >> (64 - (unsigned __int8)v112);
    }
    else
    {
      v64 = 0;
    }
    v100 = v64;
    v65 = *(_QWORD *)v119;
    v66 = *(_QWORD **)(*(_QWORD *)v116 + 24LL);
    if ( *(_DWORD *)(*(_QWORD *)v116 + 32LL) > 0x40u )
      v66 = (_QWORD *)*v66;
    v67 = *(_QWORD **)(v65 + 24);
    if ( *(_DWORD *)(v65 + 32) > 0x40u )
      v67 = (_QWORD *)*v67;
    v78 = v63;
    v83 = v64;
    if ( v67 != v66 )
    {
      v68 = sub_BB5290(a1);
      v69 = sub_BDB740(a5, v68);
      v114 = v70;
      v113 = (_QWORD *)v69;
      v105 = v78 % sub_CA1930(&v113);
      v71 = sub_BB5290(a2);
      v72 = sub_BDB740(a5, v71);
      v114 = v73;
      v113 = (_QWORD *)v72;
      v100 = v83 % sub_CA1930(&v113);
    }
  }
  if ( v105 < v100 )
  {
    v113 = (_QWORD *)(a3 & 0x3FFFFFFFFFFFFFFFLL);
    LOBYTE(v114) = (a3 & 0x4000000000000000LL) != 0;
    if ( v105 + sub_CA1930(&v113) <= v100 )
      goto LABEL_88;
    goto LABEL_64;
  }
  if ( v105 <= v100
    || (v113 = (_QWORD *)(a4 & 0x3FFFFFFFFFFFFFFFLL),
        LOBYTE(v114) = (a4 & 0x4000000000000000LL) != 0,
        v100 + sub_CA1930(&v113) > v105) )
  {
LABEL_64:
    if ( v112 > 0x40 && v111 )
      j_j___libc_free_0_0(v111);
    if ( v110 > 0x40 && v109 )
      j_j___libc_free_0_0(v109);
    goto LABEL_24;
  }
LABEL_88:
  sub_969240((__int64 *)&v111);
  sub_969240((__int64 *)&v109);
  v62 = (unsigned __int64)v119;
  result = 0;
  if ( v119 != v121 )
  {
LABEL_80:
    _libc_free(v62);
    result = 0;
  }
LABEL_81:
  if ( v116 != v118 )
  {
    _libc_free((unsigned __int64)v116);
    return 0;
  }
  return result;
}
