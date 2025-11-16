// Function: sub_118E8F0
// Address: 0x118e8f0
//
__int64 __fastcall sub_118E8F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rsi
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // r13
  __int64 v13; // rbx
  __int64 *v14; // rdi
  __int64 v15; // rdx
  unsigned int v16; // eax
  __int64 *v17; // rax
  __int64 v18; // r12
  __int64 v19; // rcx
  unsigned __int64 v20; // rbx
  __int64 v21; // r13
  __int64 v22; // rdx
  __int64 v23; // rax
  __int64 v24; // rbx
  __int64 v25; // rdx
  __int64 v26; // r12
  __int64 v27; // r14
  int v28; // r11d
  __int64 *v29; // rdx
  unsigned int v30; // r8d
  __int64 *v31; // rax
  __int64 v32; // rdi
  unsigned int v33; // esi
  int v34; // r10d
  __int64 *v35; // rdx
  unsigned int v36; // edi
  __int64 *v37; // rax
  __int64 v38; // rcx
  _BYTE *v39; // rsi
  unsigned __int64 v40; // rax
  int v41; // edx
  __int64 v42; // rax
  bool v43; // cf
  __int64 v44; // rdx
  __int64 v45; // rax
  int v46; // ebx
  __int64 v47; // r12
  __int64 v48; // rax
  __int64 v49; // r14
  char v50; // al
  __int64 v51; // r12
  __int64 v52; // rbx
  __int64 v53; // rdx
  unsigned int v54; // esi
  __int64 v55; // rbx
  __int64 v56; // rdx
  __int64 v57; // r13
  int v58; // r11d
  _QWORD *v59; // rdx
  unsigned int v60; // ecx
  _QWORD *v61; // rax
  __int64 v62; // r9
  __int64 v63; // r12
  int v64; // eax
  int v65; // eax
  unsigned int v66; // edx
  __int64 v67; // rax
  __int64 v68; // rdx
  __int64 v69; // rdx
  int v71; // eax
  int v72; // r11d
  unsigned int v73; // r8d
  __int64 v74; // rdi
  int v75; // eax
  __int64 v76; // rcx
  __int64 v77; // rdi
  int v78; // r11d
  __int64 *v79; // r9
  __int64 *v80; // rdi
  __int64 v81; // r14
  int v82; // r10d
  __int64 v83; // rsi
  int v84; // edi
  __int64 *v85; // rsi
  unsigned int v86; // ecx
  __int64 v87; // r8
  unsigned int v88; // ecx
  __int64 v89; // r8
  int v90; // edi
  unsigned int v91; // ecx
  __int64 v92; // r8
  int v93; // edi
  int v94; // edi
  unsigned int v95; // ecx
  __int64 v96; // r8
  int v97; // ecx
  unsigned int v98; // eax
  __int64 v99; // r9
  int v100; // edi
  _QWORD *v101; // rsi
  _QWORD *v102; // rdi
  unsigned int v103; // r12d
  int v104; // eax
  __int64 v105; // rsi
  __int64 v106; // rdx
  int v107; // r12d
  __int64 v108; // rax
  __int64 v109; // rdx
  __int64 *v110; // [rsp+20h] [rbp-180h]
  __int64 *v111; // [rsp+28h] [rbp-178h]
  __int64 v112; // [rsp+38h] [rbp-168h]
  __int64 v113; // [rsp+40h] [rbp-160h]
  __int64 v116; // [rsp+58h] [rbp-148h]
  unsigned __int8 *v117; // [rsp+68h] [rbp-138h]
  __int64 v118; // [rsp+70h] [rbp-130h] BYREF
  __int64 v119; // [rsp+78h] [rbp-128h] BYREF
  __int64 v120[2]; // [rsp+80h] [rbp-120h] BYREF
  __int64 v121[2]; // [rsp+90h] [rbp-110h] BYREF
  __int64 v122; // [rsp+A0h] [rbp-100h] BYREF
  __int64 v123; // [rsp+A8h] [rbp-F8h]
  __int64 v124; // [rsp+B0h] [rbp-F0h]
  unsigned int v125; // [rsp+B8h] [rbp-E8h]
  char v126[32]; // [rsp+C0h] [rbp-E0h] BYREF
  __int16 v127; // [rsp+E0h] [rbp-C0h]
  const char *v128; // [rsp+F0h] [rbp-B0h] BYREF
  __int64 v129; // [rsp+F8h] [rbp-A8h]
  __int64 *v130; // [rsp+100h] [rbp-A0h]
  __int64 *v131; // [rsp+108h] [rbp-98h]
  __int16 v132; // [rsp+110h] [rbp-90h]
  __int64 v133; // [rsp+120h] [rbp-80h] BYREF
  __int64 v134; // [rsp+128h] [rbp-78h]
  __int64 v135; // [rsp+130h] [rbp-70h]
  __int64 v136; // [rsp+138h] [rbp-68h]
  __int64 *v137; // [rsp+140h] [rbp-60h]
  __int64 v138; // [rsp+148h] [rbp-58h]
  _BYTE v139[80]; // [rsp+150h] [rbp-50h] BYREF

  v137 = (__int64 *)v139;
  v138 = 0x400000000LL;
  v128 = *(const char **)(a1 + 40);
  v7 = (__int64)&v128;
  v133 = 0;
  v134 = 0;
  v135 = 0;
  v136 = 0;
  sub_118E2B0((__int64)&v133, (__int64 *)&v128, a3, a4, a5, a6);
  if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
  {
    v12 = *(_QWORD *)(a1 - 8);
    v13 = v12 + 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
  }
  else
  {
    v13 = a1;
    v9 = a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
    v12 = v9;
  }
  for ( ; v13 != v12; v12 += 32 )
  {
    if ( **(_BYTE **)v12 > 0x1Cu )
    {
      v7 = (__int64)&v128;
      v128 = *(const char **)(*(_QWORD *)v12 + 40LL);
      sub_118E2B0((__int64)&v133, (__int64 *)&v128, v8, v9, v10, v11);
    }
  }
  v14 = v137;
  v110 = &v137[(unsigned int)v138];
  if ( v110 == v137 )
  {
    v49 = 0;
    goto LABEL_69;
  }
  v111 = v137;
  while ( 1 )
  {
    v21 = *v111;
    if ( *v111 )
    {
      v15 = (unsigned int)(*(_DWORD *)(v21 + 44) + 1);
      v16 = *(_DWORD *)(v21 + 44) + 1;
    }
    else
    {
      v15 = 0;
      v16 = 0;
    }
    if ( v16 >= *(_DWORD *)(a2 + 32) )
      BUG();
    v17 = *(__int64 **)(*(_QWORD *)(*(_QWORD *)(a2 + 24) + 8 * v15) + 8LL);
    if ( !v17 )
      goto LABEL_20;
    v18 = *v17;
    v19 = *(_QWORD *)(a1 - 96);
    v20 = *(_QWORD *)(*v17 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v20 == *v17 + 48 )
      goto LABEL_238;
    if ( !v20 )
      BUG();
    if ( (unsigned int)*(unsigned __int8 *)(v20 - 24) - 30 > 0xA )
LABEL_238:
      BUG();
    if ( *(_BYTE *)(v20 - 24) == 31
      && (*(_DWORD *)(v20 - 20) & 0x7FFFFFF) == 3
      && v19 == *(_QWORD *)(v20 - 120)
      && (v22 = *(_QWORD *)(v20 - 56)) != 0
      && (v118 = *(_QWORD *)(v20 - 56), (v23 = *(_QWORD *)(v20 - 88)) != 0) )
    {
      v119 = *(_QWORD *)(v20 - 88);
      v113 = *(_QWORD *)(a1 - 64);
      v112 = *(_QWORD *)(a1 - 32);
    }
    else
    {
      v129 = v19;
      v130 = &v118;
      v128 = 0;
      v131 = &v119;
      if ( (unsigned int)*(unsigned __int8 *)(v20 - 24) - 30 > 0xA )
        BUG();
      if ( *(_BYTE *)(v20 - 24) != 31 )
        goto LABEL_20;
      if ( (*(_DWORD *)(v20 - 20) & 0x7FFFFFF) != 3 )
        goto LABEL_20;
      v7 = 30;
      if ( !sub_9987C0((__int64)&v128, 30, *(unsigned __int8 **)(v20 - 120)) )
        goto LABEL_20;
      v108 = *(_QWORD *)(v20 - 56);
      if ( !v108 )
        goto LABEL_20;
      *v130 = v108;
      v109 = *(_QWORD *)(v20 - 88);
      if ( !v109 )
        goto LABEL_20;
      *v131 = v109;
      v22 = v118;
      v113 = *(_QWORD *)(a1 - 32);
      v112 = *(_QWORD *)(a1 - 64);
      v23 = v119;
    }
    if ( v22 != v23 )
      break;
LABEL_20:
    if ( v110 == ++v111 )
    {
      v14 = v137;
      v49 = 0;
      goto LABEL_69;
    }
  }
  v124 = 0;
  v125 = 0;
  v24 = *(_QWORD *)(v21 + 16);
  v120[0] = v18;
  v120[1] = v22;
  v121[0] = v18;
  v121[1] = v23;
  v122 = 0;
  v123 = 0;
  if ( !v24 )
    goto LABEL_45;
  while ( 1 )
  {
    v25 = *(_QWORD *)(v24 + 24);
    if ( (unsigned __int8)(*(_BYTE *)v25 - 30) <= 0xAu )
      break;
    v24 = *(_QWORD *)(v24 + 8);
    if ( !v24 )
      goto LABEL_45;
  }
LABEL_32:
  v26 = *(_QWORD *)(v25 + 40);
  v129 = v21;
  v128 = (const char *)v26;
  if ( !(unsigned __int8)sub_B1A0F0(a2, v120, (__int64 *)&v128) )
  {
    if ( !(unsigned __int8)sub_B1A0F0(a2, v121, (__int64 *)&v128) )
      goto LABEL_110;
    v27 = sub_BD5BF0(v112, v21, v26);
    if ( v125 )
    {
      v72 = 1;
      v29 = 0;
      v73 = (v125 - 1) & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
      v31 = (__int64 *)(v123 + 16LL * v73);
      v74 = *v31;
      if ( v26 == *v31 )
        goto LABEL_35;
      while ( v74 != -4096 )
      {
        if ( v74 == -8192 && !v29 )
          v29 = v31;
        v73 = (v125 - 1) & (v72 + v73);
        v31 = (__int64 *)(v123 + 16LL * v73);
        v74 = *v31;
        if ( v26 == *v31 )
        {
LABEL_35:
          v31[1] = v27;
          v33 = v125;
          if ( v125 )
            goto LABEL_36;
LABEL_98:
          ++v122;
LABEL_99:
          sub_116E750((__int64)&v122, 2 * v33);
          if ( v125 )
          {
            LODWORD(v76) = (v125 - 1) & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
            v71 = v124 + 1;
            v35 = (__int64 *)(v123 + 16LL * (unsigned int)v76);
            v77 = *v35;
            if ( v26 != *v35 )
            {
              v78 = 1;
              v79 = 0;
              while ( v77 != -4096 )
              {
                if ( !v79 && v77 == -8192 )
                  v79 = v35;
                v76 = (v125 - 1) & ((_DWORD)v76 + v78);
                v35 = (__int64 *)(v123 + 16 * v76);
                v77 = *v35;
                if ( v26 == *v35 )
                  goto LABEL_82;
                ++v78;
              }
              if ( v79 )
                v35 = v79;
            }
            goto LABEL_82;
          }
LABEL_237:
          LODWORD(v124) = v124 + 1;
          BUG();
        }
        ++v72;
      }
      if ( !v29 )
        v29 = v31;
      ++v122;
      v75 = v124 + 1;
      if ( 4 * ((int)v124 + 1) < 3 * v125 )
      {
        if ( v125 - HIDWORD(v124) - v75 <= v125 >> 3 )
        {
          sub_116E750((__int64)&v122, v125);
          if ( !v125 )
          {
LABEL_234:
            LODWORD(v124) = v124 + 1;
            BUG();
          }
          v94 = 1;
          v85 = 0;
          v95 = (v125 - 1) & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
          v75 = v124 + 1;
          v29 = (__int64 *)(v123 + 16LL * v95);
          v96 = *v29;
          if ( v26 != *v29 )
          {
            while ( v96 != -4096 )
            {
              if ( !v85 && v96 == -8192 )
                v85 = v29;
              v95 = (v125 - 1) & (v94 + v95);
              v29 = (__int64 *)(v123 + 16LL * v95);
              v96 = *v29;
              if ( v26 == *v29 )
                goto LABEL_95;
              ++v94;
            }
            goto LABEL_136;
          }
        }
        goto LABEL_95;
      }
    }
    else
    {
      ++v122;
    }
    sub_116E750((__int64)&v122, 2 * v125);
    if ( !v125 )
      goto LABEL_234;
    v91 = (v125 - 1) & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
    v75 = v124 + 1;
    v29 = (__int64 *)(v123 + 16LL * v91);
    v92 = *v29;
    if ( v26 != *v29 )
    {
      v93 = 1;
      v85 = 0;
      while ( v92 != -4096 )
      {
        if ( !v85 && v92 == -8192 )
          v85 = v29;
        v91 = (v125 - 1) & (v93 + v91);
        v29 = (__int64 *)(v123 + 16LL * v91);
        v92 = *v29;
        if ( v26 == *v29 )
          goto LABEL_95;
        ++v93;
      }
      goto LABEL_136;
    }
    goto LABEL_95;
  }
  v27 = sub_BD5BF0(v113, v21, v26);
  if ( v125 )
  {
    v28 = 1;
    v29 = 0;
    v30 = (v125 - 1) & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
    v31 = (__int64 *)(v123 + 16LL * v30);
    v32 = *v31;
    if ( v26 == *v31 )
      goto LABEL_35;
    while ( v32 != -4096 )
    {
      if ( v32 != -8192 || v29 )
        v31 = v29;
      v30 = (v125 - 1) & (v28 + v30);
      v32 = *(_QWORD *)(v123 + 16LL * v30);
      if ( v26 == v32 )
      {
        v31 = (__int64 *)(v123 + 16LL * v30);
        goto LABEL_35;
      }
      ++v28;
      v29 = v31;
      v31 = (__int64 *)(v123 + 16LL * v30);
    }
    if ( !v29 )
      v29 = v31;
    ++v122;
    v75 = v124 + 1;
    if ( 4 * ((int)v124 + 1) < 3 * v125 )
    {
      if ( v125 - HIDWORD(v124) - v75 <= v125 >> 3 )
      {
        sub_116E750((__int64)&v122, v125);
        if ( !v125 )
          goto LABEL_232;
        v84 = 1;
        v85 = 0;
        v86 = (v125 - 1) & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
        v75 = v124 + 1;
        v29 = (__int64 *)(v123 + 16LL * v86);
        v87 = *v29;
        if ( v26 != *v29 )
        {
          while ( v87 != -4096 )
          {
            if ( v87 == -8192 && !v85 )
              v85 = v29;
            v86 = (v125 - 1) & (v84 + v86);
            v29 = (__int64 *)(v123 + 16LL * v86);
            v87 = *v29;
            if ( v26 == *v29 )
              goto LABEL_95;
            ++v84;
          }
LABEL_136:
          if ( v85 )
            v29 = v85;
          goto LABEL_95;
        }
      }
      goto LABEL_95;
    }
  }
  else
  {
    ++v122;
  }
  sub_116E750((__int64)&v122, 2 * v125);
  if ( !v125 )
  {
LABEL_232:
    LODWORD(v124) = v124 + 1;
    BUG();
  }
  v88 = (v125 - 1) & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
  v75 = v124 + 1;
  v29 = (__int64 *)(v123 + 16LL * v88);
  v89 = *v29;
  if ( v26 != *v29 )
  {
    v90 = 1;
    v85 = 0;
    while ( v89 != -4096 )
    {
      if ( !v85 && v89 == -8192 )
        v85 = v29;
      v88 = (v125 - 1) & (v90 + v88);
      v29 = (__int64 *)(v123 + 16LL * v88);
      v89 = *v29;
      if ( v26 == *v29 )
        goto LABEL_95;
      ++v90;
    }
    goto LABEL_136;
  }
LABEL_95:
  LODWORD(v124) = v75;
  if ( *v29 != -4096 )
    --HIDWORD(v124);
  *v29 = v26;
  v29[1] = 0;
  v29[1] = v27;
  v33 = v125;
  if ( !v125 )
    goto LABEL_98;
LABEL_36:
  v34 = 1;
  v35 = 0;
  v36 = (v33 - 1) & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
  v37 = (__int64 *)(v123 + 16LL * v36);
  v38 = *v37;
  if ( v26 == *v37 )
  {
LABEL_37:
    v39 = (_BYTE *)v37[1];
  }
  else
  {
    while ( v38 != -4096 )
    {
      if ( !v35 && v38 == -8192 )
        v35 = v37;
      v36 = (v33 - 1) & (v34 + v36);
      v37 = (__int64 *)(v123 + 16LL * v36);
      v38 = *v37;
      if ( v26 == *v37 )
        goto LABEL_37;
      ++v34;
    }
    if ( !v35 )
      v35 = v37;
    ++v122;
    v71 = v124 + 1;
    if ( 4 * ((int)v124 + 1) >= 3 * v33 )
      goto LABEL_99;
    if ( v33 - HIDWORD(v124) - v71 <= v33 >> 3 )
    {
      sub_116E750((__int64)&v122, v33);
      if ( v125 )
      {
        v80 = 0;
        LODWORD(v81) = (v125 - 1) & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
        v82 = 1;
        v71 = v124 + 1;
        v35 = (__int64 *)(v123 + 16LL * (unsigned int)v81);
        v83 = *v35;
        if ( v26 != *v35 )
        {
          while ( v83 != -4096 )
          {
            if ( v83 == -8192 && !v80 )
              v80 = v35;
            v81 = (v125 - 1) & ((_DWORD)v81 + v82);
            v35 = (__int64 *)(v123 + 16 * v81);
            v83 = *v35;
            if ( v26 == *v35 )
              goto LABEL_82;
            ++v82;
          }
          if ( v80 )
            v35 = v80;
        }
        goto LABEL_82;
      }
      goto LABEL_237;
    }
LABEL_82:
    LODWORD(v124) = v71;
    if ( *v35 != -4096 )
      --HIDWORD(v124);
    *v35 = v26;
    v39 = 0;
    v35[1] = 0;
  }
  if ( *v39 <= 0x1Cu )
    goto LABEL_44;
  v40 = *(_QWORD *)(v26 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v40 == v26 + 48 )
  {
    if ( (unsigned __int8)sub_B19DB0(a2, (__int64)v39, 0) )
      goto LABEL_44;
    goto LABEL_110;
  }
  if ( !v40 )
    BUG();
  v41 = *(unsigned __int8 *)(v40 - 24);
  v42 = v40 - 24;
  v43 = (unsigned int)(v41 - 30) < 0xB;
  v44 = 0;
  if ( v43 )
    v44 = v42;
  if ( !(unsigned __int8)sub_B19DB0(a2, (__int64)v39, v44) )
  {
LABEL_110:
    v7 = 16LL * v125;
    sub_C7D6A0(v123, v7, 8);
    goto LABEL_20;
  }
LABEL_44:
  while ( 1 )
  {
    v24 = *(_QWORD *)(v24 + 8);
    if ( !v24 )
      break;
    v25 = *(_QWORD *)(v24 + 24);
    if ( (unsigned __int8)(*(_BYTE *)v25 - 30) <= 0xAu )
      goto LABEL_32;
  }
LABEL_45:
  v45 = v116;
  LOWORD(v45) = 1;
  v116 = v45;
  sub_A88F30(a3, v21, *(_QWORD *)(v21 + 56), 1);
  v127 = 257;
  v132 = 257;
  v46 = v124;
  v47 = *(_QWORD *)(a1 + 8);
  v48 = sub_BD2DA0(80);
  v49 = v48;
  if ( v48 )
  {
    v117 = (unsigned __int8 *)v48;
    sub_B44260(v48, v47, 55, 0x8000000u, 0, 0);
    *(_DWORD *)(v49 + 72) = v46;
    sub_BD6B50((unsigned __int8 *)v49, &v128);
    sub_BD2A10(v49, *(_DWORD *)(v49 + 72), 1);
    v50 = sub_920620((__int64)v117);
  }
  else
  {
    v117 = 0;
    v50 = sub_920620(0);
  }
  if ( v50 )
  {
    v106 = *(_QWORD *)(a3 + 96);
    v107 = *(_DWORD *)(a3 + 104);
    if ( v106 )
      sub_B99FD0(v49, 3u, v106);
    sub_B45150(v49, v107);
  }
  (*(void (__fastcall **)(_QWORD, __int64, char *, _QWORD, _QWORD))(**(_QWORD **)(a3 + 88) + 16LL))(
    *(_QWORD *)(a3 + 88),
    v49,
    v126,
    *(_QWORD *)(a3 + 56),
    *(_QWORD *)(a3 + 64));
  v51 = *(_QWORD *)a3;
  v52 = *(_QWORD *)a3 + 16LL * *(unsigned int *)(a3 + 8);
  if ( *(_QWORD *)a3 != v52 )
  {
    do
    {
      v53 = *(_QWORD *)(v51 + 8);
      v54 = *(_DWORD *)v51;
      v51 += 16;
      sub_B99FD0(v49, v54, v53);
    }
    while ( v52 != v51 );
  }
  v55 = *(_QWORD *)(v21 + 16);
  if ( !v55 )
  {
LABEL_107:
    sub_BD6B90(v117, (unsigned __int8 *)a1);
    v7 = 16LL * v125;
    sub_C7D6A0(v123, v7, 8);
    if ( v49 )
    {
      v14 = v137;
      goto LABEL_69;
    }
    goto LABEL_20;
  }
  while ( 1 )
  {
    v56 = *(_QWORD *)(v55 + 24);
    if ( (unsigned __int8)(*(_BYTE *)v56 - 30) <= 0xAu )
      break;
    v55 = *(_QWORD *)(v55 + 8);
    if ( !v55 )
      goto LABEL_107;
  }
LABEL_54:
  v57 = *(_QWORD *)(v56 + 40);
  if ( !v125 )
  {
    ++v122;
    goto LABEL_171;
  }
  v58 = 1;
  v59 = 0;
  v60 = (v125 - 1) & (((unsigned int)v57 >> 9) ^ ((unsigned int)v57 >> 4));
  v61 = (_QWORD *)(v123 + 16LL * v60);
  v62 = *v61;
  if ( v57 != *v61 )
  {
    while ( v62 != -4096 )
    {
      if ( !v59 && v62 == -8192 )
        v59 = v61;
      v60 = (v125 - 1) & (v58 + v60);
      v61 = (_QWORD *)(v123 + 16LL * v60);
      v62 = *v61;
      if ( v57 == *v61 )
        goto LABEL_56;
      ++v58;
    }
    if ( !v59 )
      v59 = v61;
    ++v122;
    v97 = v124 + 1;
    if ( 4 * ((int)v124 + 1) < 3 * v125 )
    {
      if ( v125 - HIDWORD(v124) - v97 > v125 >> 3 )
      {
LABEL_167:
        LODWORD(v124) = v97;
        if ( *v59 != -4096 )
          --HIDWORD(v124);
        *v59 = v57;
        v63 = 0;
        v59[1] = 0;
        goto LABEL_57;
      }
      sub_116E750((__int64)&v122, v125);
      if ( v125 )
      {
        v102 = 0;
        v103 = (v125 - 1) & (((unsigned int)v57 >> 9) ^ ((unsigned int)v57 >> 4));
        v97 = v124 + 1;
        v104 = 1;
        v59 = (_QWORD *)(v123 + 16LL * v103);
        v105 = *v59;
        if ( *v59 != v57 )
        {
          while ( v105 != -4096 )
          {
            if ( !v102 && v105 == -8192 )
              v102 = v59;
            v103 = (v125 - 1) & (v104 + v103);
            v59 = (_QWORD *)(v123 + 16LL * v103);
            v105 = *v59;
            if ( v57 == *v59 )
              goto LABEL_167;
            ++v104;
          }
          if ( v102 )
            v59 = v102;
        }
        goto LABEL_167;
      }
LABEL_233:
      LODWORD(v124) = v124 + 1;
      BUG();
    }
LABEL_171:
    sub_116E750((__int64)&v122, 2 * v125);
    if ( v125 )
    {
      v97 = v124 + 1;
      v98 = (v125 - 1) & (((unsigned int)v57 >> 9) ^ ((unsigned int)v57 >> 4));
      v59 = (_QWORD *)(v123 + 16LL * v98);
      v99 = *v59;
      if ( v57 != *v59 )
      {
        v100 = 1;
        v101 = 0;
        while ( v99 != -4096 )
        {
          if ( !v101 && v99 == -8192 )
            v101 = v59;
          v98 = (v125 - 1) & (v100 + v98);
          v59 = (_QWORD *)(v123 + 16LL * v98);
          v99 = *v59;
          if ( v57 == *v59 )
            goto LABEL_167;
          ++v100;
        }
        if ( v101 )
          v59 = v101;
      }
      goto LABEL_167;
    }
    goto LABEL_233;
  }
LABEL_56:
  v63 = v61[1];
LABEL_57:
  v64 = *(_DWORD *)(v49 + 4) & 0x7FFFFFF;
  if ( v64 == *(_DWORD *)(v49 + 72) )
  {
    sub_B48D90(v49);
    v64 = *(_DWORD *)(v49 + 4) & 0x7FFFFFF;
  }
  v65 = (v64 + 1) & 0x7FFFFFF;
  v66 = v65 | *(_DWORD *)(v49 + 4) & 0xF8000000;
  v67 = *(_QWORD *)(v49 - 8) + 32LL * (unsigned int)(v65 - 1);
  *(_DWORD *)(v49 + 4) = v66;
  if ( *(_QWORD *)v67 )
  {
    v68 = *(_QWORD *)(v67 + 8);
    **(_QWORD **)(v67 + 16) = v68;
    if ( v68 )
      *(_QWORD *)(v68 + 16) = *(_QWORD *)(v67 + 16);
  }
  *(_QWORD *)v67 = v63;
  if ( v63 )
  {
    v69 = *(_QWORD *)(v63 + 16);
    *(_QWORD *)(v67 + 8) = v69;
    if ( v69 )
      *(_QWORD *)(v69 + 16) = v67 + 8;
    *(_QWORD *)(v67 + 16) = v63 + 16;
    *(_QWORD *)(v63 + 16) = v67;
  }
  *(_QWORD *)(*(_QWORD *)(v49 - 8) + 32LL * *(unsigned int *)(v49 + 72)
                                   + 8LL * ((*(_DWORD *)(v49 + 4) & 0x7FFFFFFu) - 1)) = v57;
  while ( 1 )
  {
    v55 = *(_QWORD *)(v55 + 8);
    if ( !v55 )
      break;
    v56 = *(_QWORD *)(v55 + 24);
    if ( (unsigned __int8)(*(_BYTE *)v56 - 30) <= 0xAu )
      goto LABEL_54;
  }
  sub_BD6B90(v117, (unsigned __int8 *)a1);
  v7 = 16LL * v125;
  sub_C7D6A0(v123, v7, 8);
  v14 = v137;
LABEL_69:
  if ( v14 != (__int64 *)v139 )
    _libc_free(v14, v7);
  sub_C7D6A0(v134, 8LL * (unsigned int)v136, 8);
  return v49;
}
