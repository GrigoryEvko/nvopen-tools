// Function: sub_32724F0
// Address: 0x32724f0
//
__int64 __fastcall sub_32724F0(_QWORD *a1, int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, unsigned int a7)
{
  int v7; // r15d
  _QWORD *v8; // r10
  __int64 v12; // r11
  int v14; // eax
  __int64 v15; // rbx
  unsigned int v16; // esi
  __int64 *v17; // r15
  __int64 v18; // rdx
  unsigned __int64 v19; // rax
  unsigned int v20; // ebx
  __int64 v21; // rsi
  int v22; // eax
  __int64 v23; // rax
  int v24; // edx
  unsigned int v25; // r13d
  _QWORD *v27; // rax
  int v28; // eax
  __int64 v29; // rax
  unsigned __int16 v30; // dx
  __int64 v31; // rax
  unsigned __int64 v32; // rax
  __int64 v33; // rdx
  __int64 *v34; // rax
  __int64 v35; // rdx
  __int64 v36; // rcx
  _QWORD *v37; // rdx
  __int64 v38; // rax
  unsigned int v39; // esi
  __int64 v40; // rcx
  __int64 v41; // r15
  __int64 v42; // rax
  _QWORD *v43; // rcx
  _QWORD *v44; // rcx
  __int64 v45; // rax
  __int64 v46; // rdx
  __int64 v47; // rcx
  __int64 v48; // r8
  __int64 v49; // r9
  __int64 v50; // rax
  char v51; // al
  __int64 v52; // rdx
  int v53; // eax
  int v54; // eax
  unsigned __int64 v55; // rax
  __int64 v56; // rdx
  _QWORD *v57; // rax
  __int64 v58; // r12
  _QWORD *v59; // rbx
  __int64 v60; // rdx
  int v61; // eax
  __int64 *v62; // rsi
  unsigned int v63; // eax
  __int64 v64; // rcx
  __int64 v65; // rax
  __int64 v66; // rdx
  __int64 v67; // rcx
  __int64 v68; // r8
  __int64 v69; // r9
  __int64 v70; // rax
  __int64 v71; // r15
  __int64 v72; // r14
  unsigned __int8 (__fastcall *v73)(__int64, __int64, unsigned __int64 *, __int64, _QWORD, _QWORD); // r13
  __int64 v74; // rax
  __int64 v75; // rax
  _QWORD *v76; // r10
  __int64 v77; // rax
  int v78; // ebx
  int v79; // eax
  __int64 v80; // rbx
  _QWORD *v81; // r15
  __int64 v82; // rdx
  int v83; // eax
  __int64 *v84; // rsi
  unsigned int v85; // eax
  __int64 v86; // rcx
  __int64 v87; // rax
  unsigned int v88; // r14d
  __int64 v89; // rdx
  __int64 v90; // rcx
  __int64 v91; // r8
  __int64 v92; // r9
  __int64 v93; // r12
  __int64 v94; // rax
  __int64 v95; // r11
  __int64 v96; // rax
  int v97; // edx
  unsigned __int64 v98; // rdx
  char v99; // al
  unsigned __int64 v100; // rax
  int v101; // eax
  __int64 v102; // [rsp+0h] [rbp-D0h]
  unsigned int v103; // [rsp+0h] [rbp-D0h]
  unsigned int v104; // [rsp+8h] [rbp-C8h]
  __int64 v105; // [rsp+8h] [rbp-C8h]
  __int64 v106; // [rsp+10h] [rbp-C0h]
  __int64 *v107; // [rsp+10h] [rbp-C0h]
  _QWORD *v108; // [rsp+18h] [rbp-B8h]
  __int64 v109; // [rsp+18h] [rbp-B8h]
  __int64 (__fastcall *v110)(__int64, __int64, unsigned __int64 *, __int64, _QWORD, _QWORD); // [rsp+20h] [rbp-B0h]
  _QWORD *v111; // [rsp+28h] [rbp-A8h]
  __int64 v112; // [rsp+28h] [rbp-A8h]
  int v113; // [rsp+28h] [rbp-A8h]
  __int64 v114; // [rsp+28h] [rbp-A8h]
  unsigned __int8 (__fastcall *v115)(__int64, __int64, unsigned __int64 *, __int64, _QWORD, _QWORD); // [rsp+28h] [rbp-A8h]
  _QWORD *v117; // [rsp+30h] [rbp-A0h]
  unsigned int v118; // [rsp+30h] [rbp-A0h]
  _QWORD *v119; // [rsp+30h] [rbp-A0h]
  __int64 (__fastcall *v120)(__int64, __int64, unsigned __int64 *, __int64, _QWORD, _QWORD); // [rsp+30h] [rbp-A0h]
  unsigned __int8 v121; // [rsp+30h] [rbp-A0h]
  _QWORD *v122; // [rsp+30h] [rbp-A0h]
  _QWORD *v123; // [rsp+30h] [rbp-A0h]
  __int64 v124; // [rsp+38h] [rbp-98h]
  __int64 v125; // [rsp+38h] [rbp-98h]
  __int64 v126; // [rsp+38h] [rbp-98h]
  unsigned __int64 v127; // [rsp+40h] [rbp-90h] BYREF
  int v128; // [rsp+48h] [rbp-88h]
  unsigned __int64 v129; // [rsp+50h] [rbp-80h]
  __int64 v130; // [rsp+58h] [rbp-78h]
  __int16 v131; // [rsp+60h] [rbp-70h] BYREF
  __int64 v132; // [rsp+68h] [rbp-68h]
  unsigned __int64 v133; // [rsp+70h] [rbp-60h] BYREF
  __int64 v134; // [rsp+78h] [rbp-58h]
  char v135; // [rsp+80h] [rbp-50h]
  __int64 v136; // [rsp+88h] [rbp-48h]
  __int64 v137; // [rsp+90h] [rbp-40h]

  if ( *(_DWORD *)(a4 + 24) != 56 )
    return 0;
  v7 = *(_DWORD *)(a6 + 24);
  v8 = a1;
  v12 = a5;
  if ( v7 == 373
    || (v7 == 58 || v7 == 190)
    && (v27 = *(_QWORD **)(a6 + 40), *(_DWORD *)(*v27 + 24LL) == 373)
    && ((v28 = *(_DWORD *)(v27[5] + 24LL), v28 == 35) || v28 == 11) )
  {
    v29 = *(_QWORD *)(a6 + 48) + 16LL * a7;
    v30 = *(_WORD *)v29;
    v31 = *(_QWORD *)(v29 + 8);
    LOWORD(v133) = v30;
    v134 = v31;
    if ( v30 )
    {
      if ( v30 == 1 || (unsigned __int16)(v30 - 504) <= 7u )
        BUG();
      v32 = *(_QWORD *)&byte_444C4A0[16 * v30 - 16];
    }
    else
    {
      v32 = sub_3007260((__int64)&v133);
      v12 = a5;
      v8 = a1;
      v129 = v32;
      v130 = v33;
    }
    if ( v32 <= 0x40 )
    {
      v34 = *(__int64 **)(a6 + 40);
      v35 = *v34;
      if ( v7 == 373 )
      {
        v56 = *(_QWORD *)(v35 + 96);
        v57 = *(_QWORD **)(v56 + 24);
        if ( *(_DWORD *)(v56 + 32) > 0x40u )
          v57 = (_QWORD *)*v57;
        v125 = (__int64)v57;
      }
      else
      {
        v36 = *(_QWORD *)(**(_QWORD **)(v35 + 40) + 96LL);
        v37 = *(_QWORD **)(v36 + 24);
        if ( *(_DWORD *)(v36 + 32) > 0x40u )
          v37 = (_QWORD *)*v37;
        v38 = *(_QWORD *)(v34[5] + 96);
        v39 = *(_DWORD *)(v38 + 32);
        v40 = *(_QWORD *)(v38 + 24);
        if ( v7 == 190 )
        {
          if ( v39 > 0x40 )
            v40 = *(_QWORD *)v40;
          v40 = 1LL << v40;
        }
        else if ( v39 > 0x40 )
        {
          v40 = *(_QWORD *)v40;
        }
        v125 = (_QWORD)v37 * v40;
      }
      v41 = *(_QWORD *)(a3 + 56);
      v42 = -v125;
      if ( a2 != 57 )
        v42 = v125;
      v126 = v42;
      while ( v41 )
      {
        v52 = *(_QWORD *)(v41 + 16);
        v53 = *(_DWORD *)(v52 + 24);
        if ( v53 > 365 )
        {
          if ( v53 > 470 )
          {
            if ( v53 == 497 )
            {
              v43 = *(_QWORD **)(v52 + 40);
LABEL_75:
              v44 = v43 + 15;
              goto LABEL_50;
            }
          }
          else if ( v53 > 464 )
          {
            v43 = *(_QWORD **)(v52 + 40);
            goto LABEL_47;
          }
        }
        else
        {
          if ( v53 > 337 )
          {
            v43 = *(_QWORD **)(v52 + 40);
LABEL_78:
            if ( v53 > 363 )
              goto LABEL_75;
            if ( v53 == 339 )
            {
LABEL_59:
              v44 = v43 + 10;
              goto LABEL_50;
            }
LABEL_58:
            if ( (v53 & 0xFFFFFFBF) == 0x12B )
              goto LABEL_59;
            goto LABEL_49;
          }
          if ( v53 <= 294 )
          {
            if ( v53 > 292 )
              goto LABEL_57;
          }
          else if ( (unsigned int)(v53 - 298) <= 1 )
          {
LABEL_57:
            v43 = *(_QWORD **)(v52 + 40);
            goto LABEL_58;
          }
        }
        if ( (*(_BYTE *)(v52 + 32) & 2) == 0 )
          goto LABEL_5;
        v43 = *(_QWORD **)(v52 + 40);
        if ( v53 <= 365 )
          goto LABEL_78;
LABEL_47:
        if ( v53 <= 467 && v53 > 464 )
          goto LABEL_59;
LABEL_49:
        v44 = v43 + 5;
LABEL_50:
        if ( *v44 != a3 )
          goto LABEL_5;
        v133 = 0;
        v134 = 0;
        v136 = 0;
        v135 = 1;
        v137 = v126;
        v45 = *(_QWORD *)(v52 + 104);
        v102 = v12;
        v131 = *(_WORD *)(v52 + 96);
        v132 = v45;
        v111 = v8;
        v104 = sub_2EAC1E0(*(_QWORD *)(v52 + 112));
        v106 = sub_3007410((__int64)&v131, *(__int64 **)(*v111 + 64LL), v46, v47, v48, v49);
        v108 = v111;
        v112 = v111[1];
        v110 = *(__int64 (__fastcall **)(__int64, __int64, unsigned __int64 *, __int64, _QWORD, _QWORD))(*(_QWORD *)v112 + 1288LL);
        v50 = sub_2E79000(*(__int64 **)(*v108 + 40LL));
        v51 = v110(v112, v50, &v133, v106, v104, 0);
        v8 = v108;
        v12 = v102;
        if ( !v51 )
          goto LABEL_5;
        v41 = *(_QWORD *)(v41 + 32);
      }
      return 1;
    }
  }
LABEL_5:
  if ( a2 != 56 )
    return 0;
  v14 = *(_DWORD *)(a6 + 24);
  if ( v14 != 11 && v14 != 35 )
    return 0;
  v15 = *(_QWORD *)(a6 + 96);
  v16 = *(_DWORD *)(v15 + 32);
  v124 = v15;
  v17 = (__int64 *)(v15 + 24);
  v18 = 1LL << ((unsigned __int8)v16 - 1);
  v19 = *(_QWORD *)(v15 + 24);
  if ( v16 > 0x40 )
  {
    v113 = v12;
    v117 = v8;
    if ( (*(_QWORD *)(v19 + 8LL * ((v16 - 1) >> 6)) & v18) != 0 )
    {
      v54 = sub_C44500((__int64)v17);
      v8 = v117;
      LODWORD(v12) = v113;
    }
    else
    {
      v54 = sub_C444A0((__int64)v17);
      LODWORD(v12) = v113;
      v8 = v117;
    }
    v20 = v16 + 1 - v54;
  }
  else if ( (v18 & v19) != 0 )
  {
    if ( !v16 )
      goto LABEL_13;
    v55 = ~(v19 << (64 - (unsigned __int8)v16));
    if ( v55 )
    {
      _BitScanReverse64(&v55, v55);
      v20 = v16 + 1 - (v55 ^ 0x3F);
    }
    else
    {
      v20 = v16 - 63;
    }
  }
  else
  {
    if ( !v19 )
      goto LABEL_13;
    _BitScanReverse64(&v19, v19);
    v20 = 65 - (v19 ^ 0x3F);
  }
  if ( v20 > 0x40 )
    return 0;
LABEL_13:
  v21 = *(_QWORD *)(*(_QWORD *)(a4 + 40) + 40LL);
  v22 = *(_DWORD *)(v21 + 24);
  if ( v22 != 11 && v22 != 35 )
  {
    if ( ((unsigned int)(v22 - 13) <= 1 || (unsigned int)(v22 - 37) <= 1) && v22 == 13 )
    {
      v122 = v8;
      v99 = (*(__int64 (__fastcall **)(_QWORD))(*(_QWORD *)v8[1] + 1952LL))(v8[1]);
      v8 = v122;
      if ( v99 )
        return 0;
    }
    v58 = *(_QWORD *)(a3 + 56);
    if ( !v58 )
      return 1;
    v59 = v8;
    while ( 1 )
    {
      v60 = *(_QWORD *)(v58 + 16);
      v61 = *(_DWORD *)(v60 + 24);
      if ( v61 > 365 )
      {
        if ( v61 > 470 )
        {
          if ( v61 != 497 )
            goto LABEL_93;
        }
        else if ( v61 <= 464 )
        {
          goto LABEL_93;
        }
      }
      else if ( v61 <= 337 )
      {
        if ( v61 > 294 )
        {
          if ( (unsigned int)(v61 - 298) > 1 )
          {
LABEL_93:
            if ( (*(_BYTE *)(v60 + 32) & 2) == 0 )
              return 0;
          }
        }
        else if ( v61 <= 292 )
        {
          goto LABEL_93;
        }
      }
      v133 = 0;
      v134 = 0;
      v136 = 0;
      v137 = 0;
      v135 = 1;
      v62 = *(__int64 **)(v124 + 24);
      v63 = *(_DWORD *)(v124 + 32);
      if ( v63 > 0x40 )
      {
        v64 = *v62;
      }
      else
      {
        v64 = 0;
        if ( v63 )
          v64 = (__int64)((_QWORD)v62 << (64 - (unsigned __int8)v63)) >> (64 - (unsigned __int8)v63);
      }
      v134 = v64;
      v65 = *(_QWORD *)(v60 + 104);
      v131 = *(_WORD *)(v60 + 96);
      v132 = v65;
      v118 = sub_2EAC1E0(*(_QWORD *)(v60 + 112));
      v70 = sub_3007410((__int64)&v131, *(__int64 **)(*v59 + 64LL), v66, v67, v68, v69);
      v71 = v59[1];
      v72 = v70;
      v73 = *(unsigned __int8 (__fastcall **)(__int64, __int64, unsigned __int64 *, __int64, _QWORD, _QWORD))(*(_QWORD *)v71 + 1288LL);
      v74 = sub_2E79000(*(__int64 **)(*v59 + 40LL));
      if ( !v73(v71, v74, &v133, v72, v118, 0) )
        return 0;
      v58 = *(_QWORD *)(v58 + 32);
      if ( !v58 )
        return 1;
    }
  }
  v23 = *(_QWORD *)(a4 + 56);
  if ( v23 )
  {
    v24 = 1;
    do
    {
      if ( (_DWORD)v12 == *(_DWORD *)(v23 + 8) )
      {
        if ( !v24 )
          goto LABEL_109;
        v23 = *(_QWORD *)(v23 + 32);
        if ( !v23 )
          return 0;
        if ( (_DWORD)v12 == *(_DWORD *)(v23 + 8) )
          goto LABEL_109;
        v24 = 0;
      }
      v23 = *(_QWORD *)(v23 + 32);
    }
    while ( v23 );
    v25 = 0;
    if ( v24 != 1 )
      return v25;
  }
LABEL_109:
  v75 = *(_QWORD *)(v21 + 96);
  LODWORD(v134) = *(_DWORD *)(v75 + 32);
  if ( (unsigned int)v134 > 0x40 )
  {
    v123 = v8;
    sub_C43780((__int64)&v133, (const void **)(v75 + 24));
    v8 = v123;
  }
  else
  {
    v133 = *(_QWORD *)(v75 + 24);
  }
  v119 = v8;
  sub_C45EE0((__int64)&v133, v17);
  v76 = v119;
  v103 = v134;
  v128 = v134;
  v77 = 1LL << ((unsigned __int8)v134 - 1);
  v107 = (__int64 *)v133;
  v127 = v133;
  if ( (unsigned int)v134 <= 0x40 )
  {
    if ( (v77 & v133) != 0 )
    {
      if ( !(_DWORD)v134 )
        goto LABEL_157;
      v97 = 64;
      if ( v133 << (64 - (unsigned __int8)v134) != -1 )
      {
        _BitScanReverse64(&v98, ~(v133 << (64 - (unsigned __int8)v134)));
        v97 = v98 ^ 0x3F;
      }
      v25 = 0;
      if ( (unsigned int)(v134 + 1 - v97) > 0x40 )
        return v25;
    }
    else if ( v133 )
    {
      _BitScanReverse64(&v100, v133);
      v25 = 0;
      if ( (unsigned int)v100 == 0x3F )
        return v25;
    }
    if ( (_DWORD)v134 )
    {
      v80 = *(_QWORD *)(a3 + 56);
      v105 = (__int64)(v133 << (64 - (unsigned __int8)v134)) >> (64 - (unsigned __int8)v134);
      if ( !v80 )
        return 0;
      goto LABEL_115;
    }
LABEL_157:
    v105 = 0;
    v80 = *(_QWORD *)(a3 + 56);
    if ( !v80 )
      return 0;
    goto LABEL_115;
  }
  v78 = v134 + 1;
  if ( (*(_QWORD *)(v133 + 8LL * ((unsigned int)(v134 - 1) >> 6)) & v77) != 0 )
  {
    v79 = sub_C44500((__int64)&v127);
    v76 = v119;
    if ( (unsigned int)(v78 - v79) <= 0x40 )
      goto LABEL_114;
LABEL_156:
    v25 = 0;
LABEL_129:
    if ( v107 )
      j_j___libc_free_0_0((unsigned __int64)v107);
    return v25;
  }
  v101 = sub_C444A0((__int64)&v127);
  v76 = v119;
  if ( (unsigned int)(v78 - v101) > 0x40 )
    goto LABEL_156;
LABEL_114:
  v80 = *(_QWORD *)(a3 + 56);
  v105 = *v107;
  if ( !v80 )
    goto LABEL_24;
LABEL_115:
  v81 = v76;
  do
  {
    v82 = *(_QWORD *)(v80 + 16);
    v83 = *(_DWORD *)(v82 + 24);
    if ( v83 > 365 )
    {
      if ( v83 > 470 )
      {
        if ( v83 != 497 )
        {
LABEL_117:
          if ( (*(_BYTE *)(v82 + 32) & 2) == 0 )
            goto LABEL_118;
        }
      }
      else if ( v83 <= 464 )
      {
        goto LABEL_117;
      }
    }
    else if ( v83 <= 337 )
    {
      if ( v83 <= 294 )
      {
        if ( v83 <= 292 )
          goto LABEL_117;
      }
      else if ( (unsigned int)(v83 - 298) > 1 )
      {
        goto LABEL_117;
      }
    }
    v133 = 0;
    v134 = 0;
    v136 = 0;
    v137 = 0;
    v135 = 1;
    v84 = *(__int64 **)(v124 + 24);
    v85 = *(_DWORD *)(v124 + 32);
    if ( v85 > 0x40 )
    {
      v86 = *v84;
    }
    else
    {
      v86 = 0;
      if ( v85 )
        v86 = (__int64)((_QWORD)v84 << (64 - (unsigned __int8)v85)) >> (64 - (unsigned __int8)v85);
    }
    v134 = v86;
    v87 = *(_QWORD *)(v82 + 104);
    v131 = *(_WORD *)(v82 + 96);
    v132 = v87;
    v88 = sub_2EAC1E0(*(_QWORD *)(v82 + 112));
    v93 = sub_3007410((__int64)&v131, *(__int64 **)(*v81 + 64LL), v89, v90, v91, v92);
    v114 = v81[1];
    v120 = *(__int64 (__fastcall **)(__int64, __int64, unsigned __int64 *, __int64, _QWORD, _QWORD))(*(_QWORD *)v114 + 1288LL);
    v94 = sub_2E79000(*(__int64 **)(*v81 + 40LL));
    v121 = v120(v114, v94, &v133, v93, v88, 0);
    if ( v121 )
    {
      v95 = v81[1];
      v134 = v105;
      v109 = v95;
      v115 = *(unsigned __int8 (__fastcall **)(__int64, __int64, unsigned __int64 *, __int64, _QWORD, _QWORD))(*(_QWORD *)v95 + 1288LL);
      v96 = sub_2E79000(*(__int64 **)(*v81 + 40LL));
      if ( !v115(v109, v96, &v133, v93, v88, 0) )
      {
        v25 = v121;
        if ( v103 <= 0x40 )
          return v25;
        goto LABEL_129;
      }
    }
LABEL_118:
    v80 = *(_QWORD *)(v80 + 32);
  }
  while ( v80 );
  if ( v103 <= 0x40 || !v107 )
    return 0;
LABEL_24:
  j_j___libc_free_0_0((unsigned __int64)v107);
  return 0;
}
