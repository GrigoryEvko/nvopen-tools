// Function: sub_276AF50
// Address: 0x276af50
//
__int64 __fastcall sub_276AF50(__int64 *a1)
{
  _QWORD *v1; // rdx
  __int64 v2; // rax
  __int64 v3; // rcx
  __int64 v4; // rax
  __int64 v5; // r13
  unsigned __int64 *v6; // r12
  __int64 *v7; // rbx
  __int64 v8; // r13
  __int64 v9; // r12
  int v10; // ebx
  __int64 v11; // r12
  __int64 *v12; // r12
  __int64 v13; // r13
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  unsigned int v19; // r14d
  unsigned int v20; // eax
  _QWORD *v21; // rbx
  _QWORD *v22; // r13
  __int64 v23; // r15
  unsigned __int64 v24; // r12
  unsigned __int64 v25; // rdi
  __int64 v26; // rdi
  __int64 v27; // rsi
  unsigned __int64 *v28; // rdx
  _QWORD *v29; // rdi
  int v30; // r11d
  unsigned int v31; // ecx
  _QWORD *v32; // rax
  unsigned __int64 *v33; // r8
  unsigned __int64 *v34; // r12
  int v35; // ecx
  unsigned __int64 *v36; // rdx
  int v37; // r11d
  __int64 v38; // rcx
  unsigned __int64 *v39; // rax
  unsigned __int64 v40; // rdi
  unsigned __int64 *v41; // r12
  unsigned int v42; // eax
  unsigned __int64 *v43; // r9
  int v44; // r10d
  int v45; // r10d
  unsigned int v46; // eax
  __int64 *v48; // r12
  __int64 v49; // rax
  int v50; // ebx
  __int64 v51; // rcx
  unsigned __int64 v52; // rax
  __int64 v53; // r13
  __int64 v54; // rax
  __int8 *v55; // rsi
  size_t v56; // rdx
  __int64 v57; // rdx
  __int64 v58; // rcx
  __int64 v59; // r8
  __int64 v60; // r9
  __int64 *v61; // r13
  __int64 v62; // r12
  __int64 v63; // rax
  __int64 v64; // rdx
  __int64 v65; // rcx
  __int64 v66; // r8
  __int64 v67; // r9
  __int64 *v68; // r13
  int v69; // ecx
  unsigned int v70; // ebx
  __int64 v71; // r12
  __int64 v72; // rbx
  __int64 v73; // rdx
  __int64 v74; // rcx
  __int64 v75; // r8
  __int64 v76; // r9
  __int64 v77; // rax
  __int64 v78; // rax
  __int64 v79; // rax
  __int64 v80; // rax
  __int64 v81; // rax
  __int64 v82; // rax
  __int64 v83; // [rsp+8h] [rbp-538h]
  __int64 v84; // [rsp+10h] [rbp-530h]
  __int64 v85; // [rsp+18h] [rbp-528h]
  __int64 *v86; // [rsp+20h] [rbp-520h]
  int v88; // [rsp+38h] [rbp-508h]
  __int64 **v89; // [rsp+38h] [rbp-508h]
  __int64 v90; // [rsp+40h] [rbp-500h] BYREF
  unsigned __int64 *v91; // [rsp+48h] [rbp-4F8h] BYREF
  unsigned __int64 v92; // [rsp+50h] [rbp-4F0h] BYREF
  unsigned int v93; // [rsp+58h] [rbp-4E8h]
  __int64 v94; // [rsp+60h] [rbp-4E0h] BYREF
  _QWORD *v95; // [rsp+68h] [rbp-4D8h]
  __int64 v96; // [rsp+70h] [rbp-4D0h]
  unsigned int v97; // [rsp+78h] [rbp-4C8h]
  __int64 v98[4]; // [rsp+80h] [rbp-4C0h] BYREF
  unsigned __int64 v99[6]; // [rsp+A0h] [rbp-4A0h] BYREF
  __int64 v100[4]; // [rsp+D0h] [rbp-470h] BYREF
  unsigned __int64 v101; // [rsp+F0h] [rbp-450h] BYREF
  __int64 *v102; // [rsp+100h] [rbp-440h]
  __int16 v103; // [rsp+120h] [rbp-420h] BYREF
  char v104; // [rsp+122h] [rbp-41Eh]
  int v105; // [rsp+124h] [rbp-41Ch]
  char v106; // [rsp+128h] [rbp-418h]
  __int64 v107; // [rsp+130h] [rbp-410h]
  int v108; // [rsp+138h] [rbp-408h]
  __int64 v109; // [rsp+140h] [rbp-400h]
  int v110; // [rsp+148h] [rbp-3F8h]
  int v111; // [rsp+150h] [rbp-3F0h]
  __int64 v112; // [rsp+158h] [rbp-3E8h]
  __int64 v113; // [rsp+160h] [rbp-3E0h]
  __int64 v114; // [rsp+168h] [rbp-3D8h]
  unsigned int v115; // [rsp+170h] [rbp-3D0h]
  __int64 v116; // [rsp+178h] [rbp-3C8h]
  __int64 v117; // [rsp+180h] [rbp-3C0h]
  __int64 v118; // [rsp+188h] [rbp-3B8h]
  __int64 v119; // [rsp+190h] [rbp-3B0h]
  __int64 v120; // [rsp+198h] [rbp-3A8h]
  __int64 v121; // [rsp+1A0h] [rbp-3A0h]
  _QWORD v122[10]; // [rsp+1B0h] [rbp-390h] BYREF
  _BYTE v123[344]; // [rsp+200h] [rbp-340h] BYREF
  __int64 v124; // [rsp+358h] [rbp-1E8h]
  unsigned __int64 *v125; // [rsp+360h] [rbp-1E0h] BYREF
  unsigned __int64 v126; // [rsp+368h] [rbp-1D8h] BYREF
  unsigned int v127; // [rsp+370h] [rbp-1D0h]
  _BYTE v128[344]; // [rsp+3B0h] [rbp-190h] BYREF
  __int64 v129; // [rsp+508h] [rbp-38h]

  v1 = (_QWORD *)*a1;
  v106 = 0;
  v107 = 0;
  v108 = 0;
  v109 = 0;
  v110 = 0;
  v111 = 0;
  v112 = 0;
  v113 = 0;
  v114 = 0;
  v115 = 0;
  v116 = 0;
  v117 = 0;
  v118 = 0;
  v119 = 0;
  v120 = 0;
  v121 = 0;
  v2 = v1[1];
  v105 = 0;
  v83 = v2;
  v88 = *(_DWORD *)(v2 + 4);
  v103 = 0;
  v104 = 0;
  if ( (v88 & 0x7FFFFFFu) <= 3 )
  {
    v26 = 0;
    v27 = 0;
    v19 = 0;
    goto LABEL_88;
  }
  v3 = v1[5];
  v4 = v1[4];
  v94 = 0;
  v95 = 0;
  v96 = 0;
  v97 = 0;
  v84 = v3;
  if ( v4 == v3 )
  {
LABEL_20:
    LODWORD(v92) = 0;
    sub_DF9560(a1[3], v83, &v92);
    if ( (_DWORD)v92 )
    {
      v10 = v108;
      v11 = v107 / (unsigned int)v92;
    }
    else
    {
      v49 = *(unsigned int *)(v83 + 4);
      LODWORD(v126) = 32;
      v125 = (unsigned __int64 *)((unsigned __int64)(v49 << 37) >> 38);
      sub_C46E90((__int64)&v125);
      v50 = v126;
      if ( (unsigned int)v126 > 0x40 )
      {
        v70 = v50 - sub_C444A0((__int64)&v125);
        if ( v125 )
          j_j___libc_free_0_0((unsigned __int64)v125);
        v51 = v70;
      }
      else
      {
        v51 = 0;
        if ( v125 )
        {
          _BitScanReverse64(&v52, (unsigned __int64)v125);
          v51 = 64 - ((unsigned int)v52 ^ 0x3F);
        }
      }
      v10 = v108;
      v11 = v107 / v51;
    }
    if ( v10 )
    {
      if ( v10 <= 0 )
        goto LABEL_24;
    }
    else if ( (unsigned int)qword_4FFAC48 >= v11 )
    {
LABEL_24:
      v12 = (__int64 *)a1[4];
      v13 = *v12;
      v14 = sub_B2BE50(*v12);
      if ( sub_B6EA50(v14)
        || (v77 = sub_B2BE50(v13),
            v78 = sub_B6F970(v77),
            (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v78 + 48LL))(v78)) )
      {
        sub_B174A0((__int64)&v125, (__int64)"dfa-jump-threading", (__int64)"JumpThreaded", 12, v83);
        sub_B18290((__int64)&v125, "Switch statement jump-threaded.", 0x1Fu);
        sub_23FE290((__int64)v122, (__int64)&v125, v15, v16, v17, v18);
        v124 = v129;
        v122[0] = &unk_49D9D78;
        v125 = (unsigned __int64 *)&unk_49D9D40;
        sub_23FD590((__int64)v128);
        sub_1049740(v12, (__int64)v122);
        v122[0] = &unk_49D9D40;
        sub_23FD590((__int64)v123);
      }
      v19 = 1;
      goto LABEL_27;
    }
    v68 = (__int64 *)a1[4];
    if ( (unsigned __int8)sub_27657B0(*v68) )
    {
      sub_B176B0((__int64)&v125, (__int64)"dfa-jump-threading", (__int64)"NotProfitable", 13, v83);
      sub_B18290((__int64)&v125, "Duplication cost exceeds the cost threshold (cost=", 0x32u);
      sub_B16D50((__int64)v100, "Cost", 4, v11, (unsigned int)v10);
      v71 = sub_2445430((__int64)&v125, (__int64)v100);
      sub_B18290(v71, ", threshold=", 0xCu);
      sub_B169E0(v98, "Threshold", 9, qword_4FFAC48);
      v72 = sub_2445430(v71, (__int64)v98);
      sub_B18290(v72, ").", 2u);
      sub_23FE290((__int64)v122, v72, v73, v74, v75, v76);
      v124 = *(_QWORD *)(v72 + 424);
      v122[0] = &unk_49D9DB0;
      sub_2240A30(v99);
      sub_2240A30((unsigned __int64 *)v98);
      sub_2240A30(&v101);
      sub_2240A30((unsigned __int64 *)v100);
      v125 = (unsigned __int64 *)&unk_49D9D40;
      sub_23FD590((__int64)v128);
      sub_1049740(v68, (__int64)v122);
      v122[0] = &unk_49D9D40;
      sub_23FD590((__int64)v123);
    }
    v19 = 0;
    goto LABEL_27;
  }
  v5 = v4;
  v86 = a1 + 5;
  while ( 1 )
  {
    sub_276A0C0(v100, (_QWORD *)v5);
    v93 = *(_DWORD *)(v5 + 88);
    if ( v93 > 0x40 )
      sub_C43780((__int64)&v92, (const void **)(v5 + 80));
    else
      v92 = *(_QWORD *)(v5 + 80);
    v90 = *(_QWORD *)(v5 + 96);
    v91 = *(unsigned __int64 **)(*a1 + 16);
    v6 = (unsigned __int64 *)sub_2766DC0((__int64)v91, (__int64)&v92, (__int64)&v94);
    if ( v6 )
      goto LABEL_7;
    sub_30ABD80(&v103, v91, a1[3], v86, 0, 0);
    if ( !v97 )
    {
      ++v94;
      v125 = 0;
      goto LABEL_122;
    }
    v36 = v91;
    v37 = 1;
    LODWORD(v38) = (v97 - 1) & (((unsigned int)v91 >> 9) ^ ((unsigned int)v91 >> 4));
    v39 = &v95[4 * (unsigned int)v38];
    v40 = *v39;
    if ( v91 != (unsigned __int64 *)*v39 )
    {
      while ( v40 != -4096 )
      {
        if ( v40 == -8192 && !v6 )
          v6 = v39;
        v38 = (v97 - 1) & ((_DWORD)v38 + v37);
        v39 = &v95[4 * v38];
        v40 = *v39;
        if ( v91 == (unsigned __int64 *)*v39 )
          goto LABEL_67;
        ++v37;
      }
      if ( !v6 )
        v6 = v39;
      ++v94;
      v69 = v96 + 1;
      v125 = v6;
      if ( 4 * ((int)v96 + 1) < 3 * v97 )
      {
        if ( v97 - HIDWORD(v96) - v69 > v97 >> 3 )
        {
LABEL_118:
          LODWORD(v96) = v69;
          if ( *v6 != -4096 )
            --HIDWORD(v96);
          *v6 = (unsigned __int64)v36;
          v41 = v6 + 1;
          v36 = v91;
          *v41 = 0;
          v41[1] = 0;
          v41[2] = 0;
          goto LABEL_68;
        }
        sub_2765FD0((__int64)&v94, v97);
LABEL_123:
        sub_2765330((__int64)&v94, (__int64 *)&v91, &v125);
        v36 = v91;
        v6 = v125;
        v69 = v96 + 1;
        goto LABEL_118;
      }
LABEL_122:
      sub_2765FD0((__int64)&v94, 2 * v97);
      goto LABEL_123;
    }
LABEL_67:
    v41 = v39 + 1;
LABEL_68:
    v125 = v36;
    v127 = v93;
    if ( v93 > 0x40 )
      sub_C43780((__int64)&v126, (const void **)&v92);
    else
      v126 = v92;
    sub_2765DC0(v41, (__int64)&v125);
    if ( v127 > 0x40 && v126 )
      j_j___libc_free_0_0(v126);
LABEL_7:
    if ( *(_QWORD *)v100[2] == v90 )
      goto LABEL_16;
    sub_2767580(v98, v100, &v90);
    v85 = v5;
    v7 = (__int64 *)v98[0];
    v8 = v98[2];
    v89 = (__int64 **)(v98[3] + 8);
LABEL_9:
    if ( v102 != v7 )
    {
      while ( 1 )
      {
        v91 = (unsigned __int64 *)*v7;
        v9 = sub_2766DC0((__int64)v91, (__int64)&v92, (__int64)&v94);
        if ( !v9 )
          break;
LABEL_11:
        if ( (__int64 *)v8 != ++v7 )
          goto LABEL_9;
        v7 = *v89++;
        v8 = (__int64)(v7 + 64);
        if ( v102 == v7 )
          goto LABEL_13;
      }
      sub_30ABD80(&v103, v91, a1[3], v86, 0, 0);
      if ( v97 )
      {
        v28 = v91;
        v29 = 0;
        v30 = 1;
        v31 = (v97 - 1) & (((unsigned int)v91 >> 9) ^ ((unsigned int)v91 >> 4));
        v32 = &v95[4 * v31];
        v33 = (unsigned __int64 *)*v32;
        if ( v91 == (unsigned __int64 *)*v32 )
        {
LABEL_44:
          v34 = v32 + 1;
LABEL_45:
          v125 = v28;
          v127 = v93;
          if ( v93 > 0x40 )
            sub_C43780((__int64)&v126, (const void **)&v92);
          else
            v126 = v92;
          sub_2765DC0(v34, (__int64)&v125);
          if ( v127 > 0x40 && v126 )
            j_j___libc_free_0_0(v126);
          goto LABEL_11;
        }
        while ( v33 != (unsigned __int64 *)-4096LL )
        {
          if ( !v29 && v33 == (unsigned __int64 *)-8192LL )
            v29 = v32;
          v31 = (v97 - 1) & (v30 + v31);
          v32 = &v95[4 * v31];
          v33 = (unsigned __int64 *)*v32;
          if ( v91 == (unsigned __int64 *)*v32 )
            goto LABEL_44;
          ++v30;
        }
        if ( !v29 )
          v29 = v32;
        ++v94;
        v35 = v96 + 1;
        if ( 4 * ((int)v96 + 1) < 3 * v97 )
        {
          if ( v97 - HIDWORD(v96) - v35 <= v97 >> 3 )
          {
            sub_2765FD0((__int64)&v94, v97);
            if ( !v97 )
            {
LABEL_146:
              LODWORD(v96) = v96 + 1;
              BUG();
            }
            v45 = 1;
            v35 = v96 + 1;
            v46 = (v97 - 1) & (((unsigned int)v91 >> 9) ^ ((unsigned int)v91 >> 4));
            v29 = &v95[4 * v46];
            v28 = (unsigned __int64 *)*v29;
            if ( v91 != (unsigned __int64 *)*v29 )
            {
              while ( v28 != (unsigned __int64 *)-4096LL )
              {
                if ( v28 == (unsigned __int64 *)-8192LL && !v9 )
                  v9 = (__int64)v29;
                v46 = (v97 - 1) & (v45 + v46);
                v29 = &v95[4 * v46];
                v28 = (unsigned __int64 *)*v29;
                if ( v91 == (unsigned __int64 *)*v29 )
                  goto LABEL_60;
                ++v45;
              }
              v28 = v91;
              if ( v9 )
                v29 = (_QWORD *)v9;
            }
          }
          goto LABEL_60;
        }
      }
      else
      {
        ++v94;
      }
      sub_2765FD0((__int64)&v94, 2 * v97);
      if ( !v97 )
        goto LABEL_146;
      v28 = v91;
      v42 = (v97 - 1) & (((unsigned int)v91 >> 9) ^ ((unsigned int)v91 >> 4));
      v35 = v96 + 1;
      v29 = &v95[4 * v42];
      v43 = (unsigned __int64 *)*v29;
      if ( (unsigned __int64 *)*v29 != v91 )
      {
        v44 = 1;
        while ( v43 != (unsigned __int64 *)-4096LL )
        {
          if ( v43 == (unsigned __int64 *)-8192LL && !v9 )
            v9 = (__int64)v29;
          v42 = (v97 - 1) & (v44 + v42);
          v29 = &v95[4 * v42];
          v43 = (unsigned __int64 *)*v29;
          if ( v91 == (unsigned __int64 *)*v29 )
            goto LABEL_60;
          ++v44;
        }
        if ( v9 )
          v29 = (_QWORD *)v9;
      }
LABEL_60:
      LODWORD(v96) = v35;
      if ( *v29 != -4096 )
        --HIDWORD(v96);
      *v29 = v28;
      v34 = v29 + 1;
      v28 = v91;
      v29[1] = 0;
      v29[2] = 0;
      v29[3] = 0;
      goto LABEL_45;
    }
LABEL_13:
    v5 = v85;
    if ( v104 )
    {
      v48 = (__int64 *)a1[4];
      v53 = *v48;
      v54 = sub_B2BE50(*v48);
      if ( !sub_B6EA50(v54) )
      {
        v79 = sub_B2BE50(v53);
        v80 = sub_B6F970(v79);
        if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v80 + 48LL))(v80) )
          goto LABEL_91;
      }
      sub_B176B0((__int64)&v125, (__int64)"dfa-jump-threading", (__int64)"NonDuplicatableInst", 19, v83);
      v55 = "Contains non-duplicatable instructions.";
      v56 = 39;
LABEL_101:
      sub_B18290((__int64)&v125, v55, v56);
      sub_23FE290((__int64)v122, (__int64)&v125, v57, v58, v59, v60);
      v124 = v129;
      v122[0] = &unk_49D9DB0;
      v125 = (unsigned __int64 *)&unk_49D9D40;
      sub_23FD590((__int64)v128);
      sub_1049740(v48, (__int64)v122);
      v122[0] = &unk_49D9D40;
      sub_23FD590((__int64)v123);
      goto LABEL_91;
    }
    if ( v105 )
      break;
    if ( v108 )
    {
      v48 = (__int64 *)a1[4];
      if ( !(unsigned __int8)sub_27657B0(*v48) )
        goto LABEL_91;
      sub_B176B0((__int64)&v125, (__int64)"dfa-jump-threading", (__int64)"ConvergentInst", 14, v83);
      v55 = "Contains instructions with invalid cost.";
      v56 = 40;
      goto LABEL_101;
    }
LABEL_16:
    if ( v93 > 0x40 && v92 )
      j_j___libc_free_0_0(v92);
    v5 += 112;
    sub_2767770((unsigned __int64 *)v100);
    if ( v84 == v5 )
      goto LABEL_20;
  }
  v61 = (__int64 *)a1[4];
  v62 = *v61;
  v63 = sub_B2BE50(*v61);
  if ( sub_B6EA50(v63)
    || (v81 = sub_B2BE50(v62),
        v82 = sub_B6F970(v81),
        (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v82 + 48LL))(v82)) )
  {
    sub_B176B0((__int64)&v125, (__int64)"dfa-jump-threading", (__int64)"ConvergentInst", 14, v83);
    sub_B18290((__int64)&v125, "Contains convergent instructions.", 0x21u);
    sub_23FE290((__int64)v122, (__int64)&v125, v64, v65, v66, v67);
    v124 = v129;
    v122[0] = &unk_49D9DB0;
    v125 = (unsigned __int64 *)&unk_49D9D40;
    sub_23FD590((__int64)v128);
    sub_1049740(v61, (__int64)v122);
    v122[0] = &unk_49D9D40;
    sub_23FD590((__int64)v123);
  }
LABEL_91:
  if ( v93 > 0x40 && v92 )
    j_j___libc_free_0_0(v92);
  v19 = 0;
  sub_2767770((unsigned __int64 *)v100);
LABEL_27:
  v20 = v97;
  if ( v97 )
  {
    v21 = v95;
    v22 = &v95[4 * v97];
    do
    {
      if ( *v21 != -4096 && *v21 != -8192 )
      {
        v23 = v21[2];
        v24 = v21[1];
        if ( v23 != v24 )
        {
          do
          {
            if ( *(_DWORD *)(v24 + 16) > 0x40u )
            {
              v25 = *(_QWORD *)(v24 + 8);
              if ( v25 )
                j_j___libc_free_0_0(v25);
            }
            v24 += 24LL;
          }
          while ( v23 != v24 );
          v24 = v21[1];
        }
        if ( v24 )
          j_j___libc_free_0(v24);
      }
      v21 += 4;
    }
    while ( v22 != v21 );
    v20 = v97;
  }
  sub_C7D6A0((__int64)v95, 32LL * v20, 8);
  v26 = v113;
  v27 = 24LL * v115;
LABEL_88:
  sub_C7D6A0(v26, v27, 8);
  return v19;
}
