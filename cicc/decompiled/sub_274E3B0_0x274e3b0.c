// Function: sub_274E3B0
// Address: 0x274e3b0
//
__int64 __fastcall sub_274E3B0(__int64 a1, __int64 *a2, __int64 a3)
{
  __int64 v3; // rbx
  __int64 *v4; // rax
  __int64 v5; // rax
  unsigned __int64 v6; // rax
  __int64 v7; // r13
  unsigned int v8; // r15d
  __int64 v9; // r14
  unsigned int i; // r12d
  int v11; // r10d
  _QWORD *v12; // rdx
  unsigned int v13; // ecx
  unsigned int v14; // edi
  __int64 v15; // rax
  __int64 v16; // rsi
  __int64 v17; // rax
  __int64 v18; // rbx
  unsigned int v19; // esi
  unsigned int v20; // ecx
  int v21; // eax
  __int64 v22; // r8
  _DWORD *v23; // rdx
  unsigned int v24; // r13d
  __int64 v25; // r12
  __int64 v26; // r15
  bool v27; // al
  __int64 v28; // r14
  __int64 v29; // rax
  __int64 v30; // rdx
  unsigned int v31; // esi
  __int64 v32; // rdi
  int v33; // eax
  bool v34; // cl
  _QWORD *v35; // r9
  int v36; // r10d
  unsigned int v37; // ecx
  __int64 v38; // rdi
  __int64 v39; // rdx
  __int64 v40; // rcx
  __int64 v41; // rcx
  __int64 v42; // rdi
  __int64 v43; // rsi
  __int64 v44; // rdx
  __int64 v45; // rcx
  __int64 v46; // r8
  __int64 v47; // r9
  __int64 v48; // rdx
  __int64 v49; // rcx
  __int64 v50; // r8
  __int64 v51; // r9
  _QWORD *v52; // rbx
  _QWORD *v53; // r14
  void (__fastcall *v54)(_QWORD *, _QWORD *, __int64); // rax
  __int64 v55; // rax
  __int64 v57; // rax
  __int64 v58; // rbx
  __int64 v59; // rax
  unsigned __int8 v60; // cl
  __int64 v61; // rdx
  __int64 v62; // r9
  int v63; // r10d
  unsigned int v64; // r14d
  __int64 *v65; // rdx
  __int64 v66; // r8
  __int64 *v67; // rax
  __int64 v68; // rdi
  int v70; // eax
  __int64 v71; // rax
  __int64 *v72; // rdx
  __int64 v73; // r13
  __int64 v74; // rax
  __int64 v75; // r14
  __int64 v76; // r12
  __int64 v77; // rax
  __int64 v78; // rbx
  __int64 v79; // r12
  _QWORD *v80; // rdi
  __int64 v81; // rax
  __int64 v82; // rdx
  __int64 v83; // rdx
  __int64 v84; // rcx
  __int64 v85; // r8
  __int64 v86; // r9
  unsigned int v87; // esi
  __int64 v88; // rdi
  int v89; // r10d
  __int64 *v90; // r8
  __int64 v91; // rsi
  int v92; // r10d
  __int64 v93; // rdi
  int v94; // r11d
  int v95; // r9d
  unsigned int v96; // r11d
  _QWORD *v97; // r10
  __int64 v98; // [rsp+10h] [rbp-400h]
  unsigned __int8 v99; // [rsp+1Bh] [rbp-3F5h]
  unsigned int v100; // [rsp+1Ch] [rbp-3F4h]
  __int64 v101; // [rsp+20h] [rbp-3F0h]
  __int64 v102; // [rsp+28h] [rbp-3E8h]
  __int64 v103; // [rsp+28h] [rbp-3E8h]
  unsigned __int8 v104; // [rsp+28h] [rbp-3E8h]
  __int64 v106; // [rsp+38h] [rbp-3D8h]
  unsigned int v107; // [rsp+38h] [rbp-3D8h]
  int v109; // [rsp+48h] [rbp-3C8h]
  __int64 v110; // [rsp+48h] [rbp-3C8h]
  __int64 v111; // [rsp+58h] [rbp-3B8h] BYREF
  __int64 v112; // [rsp+60h] [rbp-3B0h] BYREF
  __int64 v113; // [rsp+68h] [rbp-3A8h]
  __int64 v114; // [rsp+70h] [rbp-3A0h]
  unsigned int v115; // [rsp+78h] [rbp-398h]
  unsigned __int64 v116; // [rsp+80h] [rbp-390h] BYREF
  unsigned int v117; // [rsp+88h] [rbp-388h]
  unsigned __int64 v118; // [rsp+90h] [rbp-380h]
  unsigned int v119; // [rsp+98h] [rbp-378h]
  const char *v120; // [rsp+A0h] [rbp-370h] BYREF
  unsigned __int64 v121; // [rsp+A8h] [rbp-368h]
  char v122; // [rsp+C0h] [rbp-350h]
  char v123; // [rsp+C1h] [rbp-34Fh]
  __int64 v124; // [rsp+D0h] [rbp-340h] BYREF
  char *v125; // [rsp+D8h] [rbp-338h]
  char v126; // [rsp+E8h] [rbp-328h] BYREF
  char v127; // [rsp+108h] [rbp-308h]
  char v128; // [rsp+110h] [rbp-300h]
  unsigned __int64 v129[2]; // [rsp+120h] [rbp-2F0h] BYREF
  _BYTE v130[512]; // [rsp+130h] [rbp-2E0h] BYREF
  __int64 v131; // [rsp+330h] [rbp-E0h]
  __int64 v132; // [rsp+338h] [rbp-D8h]
  __int64 v133; // [rsp+340h] [rbp-D0h]
  __int64 v134; // [rsp+348h] [rbp-C8h]
  char v135; // [rsp+350h] [rbp-C0h]
  __int64 v136; // [rsp+358h] [rbp-B8h]
  char *v137; // [rsp+360h] [rbp-B0h]
  __int64 v138; // [rsp+368h] [rbp-A8h]
  int v139; // [rsp+370h] [rbp-A0h]
  char v140; // [rsp+374h] [rbp-9Ch]
  char v141; // [rsp+378h] [rbp-98h] BYREF
  __int16 v142; // [rsp+3B8h] [rbp-58h]
  _QWORD *v143; // [rsp+3C0h] [rbp-50h]
  _QWORD *v144; // [rsp+3C8h] [rbp-48h]
  __int64 v145; // [rsp+3D0h] [rbp-40h]

  v129[0] = (unsigned __int64)v130;
  v3 = *(_QWORD *)(a1 + 40);
  v129[1] = 0x1000000000LL;
  v137 = &v141;
  v142 = 0;
  v4 = *(__int64 **)(a1 - 8);
  v133 = a3;
  v131 = 0;
  v132 = 0;
  v134 = 0;
  v135 = 1;
  v136 = 0;
  v138 = 8;
  v139 = 0;
  v140 = 1;
  v143 = 0;
  v144 = 0;
  v145 = 0;
  v102 = *v4;
  v98 = v3;
  v112 = 0;
  v113 = 0;
  v5 = *(_QWORD *)(v3 + 48);
  v114 = 0;
  v6 = v5 & 0xFFFFFFFFFFFFFFF8LL;
  v115 = 0;
  if ( v6 != v3 + 48 )
  {
    if ( !v6 )
      BUG();
    v7 = v6 - 24;
    if ( (unsigned int)*(unsigned __int8 *)(v6 - 24) - 30 <= 0xA )
    {
      v109 = sub_B46E30(v7);
      if ( v109 )
      {
        v8 = 0;
        v9 = 0;
        for ( i = 0; ; i = v115 )
        {
          v17 = sub_B46EC0(v7, v8);
          v18 = v17;
          if ( !i )
            break;
          v11 = 1;
          v12 = 0;
          v13 = ((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4);
          v14 = (i - 1) & v13;
          v15 = v9 + 16LL * v14;
          v16 = *(_QWORD *)v15;
          if ( v18 == *(_QWORD *)v15 )
          {
LABEL_7:
            ++v8;
            ++*(_DWORD *)(v15 + 8);
            if ( v109 == v8 )
              goto LABEL_16;
            goto LABEL_8;
          }
          while ( v16 != -4096 )
          {
            if ( v16 == -8192 && !v12 )
              v12 = (_QWORD *)v15;
            v14 = (i - 1) & (v11 + v14);
            v15 = v9 + 16LL * v14;
            v16 = *(_QWORD *)v15;
            if ( v18 == *(_QWORD *)v15 )
              goto LABEL_7;
            ++v11;
          }
          if ( !v12 )
            v12 = (_QWORD *)v15;
          ++v112;
          v21 = v114 + 1;
          if ( 4 * ((int)v114 + 1) >= 3 * i )
            goto LABEL_11;
          if ( i - (v21 + HIDWORD(v114)) <= i >> 3 )
          {
            v107 = v13;
            sub_274DF80((__int64)&v112, i);
            if ( !v115 )
              goto LABEL_159;
            v35 = 0;
            v36 = 1;
            v37 = (v115 - 1) & v107;
            v21 = v114 + 1;
            v12 = (_QWORD *)(v113 + 16LL * v37);
            v38 = *v12;
            if ( v18 != *v12 )
            {
              while ( v38 != -4096 )
              {
                if ( !v35 && v38 == -8192 )
                  v35 = v12;
                v37 = (v115 - 1) & (v36 + v37);
                v12 = (_QWORD *)(v113 + 16LL * v37);
                v38 = *v12;
                if ( v18 == *v12 )
                  goto LABEL_13;
                ++v36;
              }
              goto LABEL_42;
            }
          }
LABEL_13:
          LODWORD(v114) = v21;
          if ( *v12 != -4096 )
            --HIDWORD(v114);
          *v12 = v18;
          v23 = v12 + 1;
          ++v8;
          *v23 = 0;
          *v23 = 1;
          if ( v109 == v8 )
            goto LABEL_16;
LABEL_8:
          v9 = v113;
        }
        ++v112;
LABEL_11:
        sub_274DF80((__int64)&v112, 2 * i);
        if ( !v115 )
          goto LABEL_159;
        v19 = v115 - 1;
        v20 = (v115 - 1) & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
        v21 = v114 + 1;
        v12 = (_QWORD *)(v113 + 16LL * v20);
        v22 = *v12;
        if ( v18 != *v12 )
        {
          v94 = 1;
          v35 = 0;
          while ( v22 != -4096 )
          {
            if ( v22 != -8192 || v35 )
              v12 = v35;
            v95 = v94 + 1;
            v96 = v20 + v94;
            v20 = v19 & v96;
            v97 = (_QWORD *)(v113 + 16LL * (v19 & v96));
            v22 = *v97;
            if ( v18 == *v97 )
            {
              v12 = (_QWORD *)(v113 + 16LL * (v19 & v96));
              goto LABEL_13;
            }
            v94 = v95;
            v35 = v12;
            v12 = v97;
          }
LABEL_42:
          if ( v35 )
            v12 = v35;
          goto LABEL_13;
        }
        goto LABEL_13;
      }
    }
  }
LABEL_16:
  v127 = 0;
  v24 = 0;
  v128 = 0;
  v124 = a1;
  sub_B540B0(&v124);
  v99 = 0;
  v106 = ((*(_DWORD *)(v124 + 4) & 0x7FFFFFFu) >> 1) - 1;
  if ( (*(_DWORD *)(v124 + 4) & 0x7FFFFFFu) >> 1 == 1 )
  {
    v111 = *(_QWORD *)(*(_QWORD *)(v124 - 8) + 32LL);
  }
  else
  {
    v25 = 0;
    v110 = v124;
    v26 = v102;
    while ( 1 )
    {
      v28 = *(_QWORD *)(*(_QWORD *)(v110 - 8) + 32LL * (unsigned int)(2 * (v25 + 1)));
      v29 = sub_22CF7C0(a2, 0x20u, v26, v28, a1, 1);
      v30 = v29;
      if ( !v29 || *(_BYTE *)v29 != 17 )
        goto LABEL_20;
      v31 = *(_DWORD *)(v29 + 32);
      v32 = v29 + 24;
      if ( v31 <= 0x40 )
      {
        v34 = *(_QWORD *)(v29 + 24) == 0;
      }
      else
      {
        v100 = *(_DWORD *)(v29 + 32);
        v103 = v29 + 24;
        v101 = v29;
        v33 = sub_C444A0(v32);
        v31 = v100;
        v32 = v103;
        v30 = v101;
        v34 = v100 == v33;
      }
      if ( !v34 )
        break;
      v57 = 32;
      if ( (_DWORD)v25 != -2 )
        v57 = 32LL * (unsigned int)(2 * v25 + 3);
      v104 = v34;
      v58 = *(_QWORD *)(*(_QWORD *)(v110 - 8) + v57);
      sub_AA5980(v58, v98, 0);
      v59 = sub_B541A0((__int64)&v124, v110, v25);
      v60 = v104;
      v25 = v61;
      v110 = v59;
      v106 = ((*(_DWORD *)(v124 + 4) & 0x7FFFFFFu) >> 1) - 1;
      v26 = **(_QWORD **)(v124 - 8);
      if ( !v115 )
      {
        ++v112;
        goto LABEL_121;
      }
      v62 = v113;
      v63 = 1;
      v64 = ((unsigned int)v58 >> 9) ^ ((unsigned int)v58 >> 4);
      v65 = 0;
      v66 = (v115 - 1) & v64;
      v67 = (__int64 *)(v113 + 16 * v66);
      v68 = *v67;
      if ( *v67 == v58 )
        goto LABEL_80;
      while ( 1 )
      {
        if ( v68 == -4096 )
        {
          if ( !v65 )
            v65 = v67;
          ++v112;
          v70 = v114 + 1;
          if ( 4 * ((int)v114 + 1) >= 3 * v115 )
          {
LABEL_121:
            sub_274DF80((__int64)&v112, 2 * v115);
            if ( v115 )
            {
              v60 = v104;
              v87 = (v115 - 1) & (((unsigned int)v58 >> 9) ^ ((unsigned int)v58 >> 4));
              v70 = v114 + 1;
              v65 = (__int64 *)(v113 + 16LL * v87);
              v88 = *v65;
              if ( v58 == *v65 )
                goto LABEL_88;
              v89 = 1;
              v90 = 0;
              while ( v88 != -4096 )
              {
                if ( !v90 && v88 == -8192 )
                  v90 = v65;
                v87 = (v115 - 1) & (v89 + v87);
                v65 = (__int64 *)(v113 + 16LL * v87);
                v88 = *v65;
                if ( v58 == *v65 )
                  goto LABEL_88;
                ++v89;
              }
LABEL_125:
              if ( v90 )
                v65 = v90;
              goto LABEL_88;
            }
          }
          else
          {
            if ( v115 - HIDWORD(v114) - v70 > v115 >> 3 )
            {
LABEL_88:
              LODWORD(v114) = v70;
              if ( *v65 != -4096 )
                --HIDWORD(v114);
              *v65 = v58;
              *((_DWORD *)v65 + 2) = -1;
              goto LABEL_91;
            }
            sub_274DF80((__int64)&v112, v115);
            if ( v115 )
            {
              v90 = 0;
              LODWORD(v91) = (v115 - 1) & v64;
              v92 = 1;
              v60 = v104;
              v70 = v114 + 1;
              v65 = (__int64 *)(v113 + 16LL * (unsigned int)v91);
              v93 = *v65;
              if ( *v65 == v58 )
                goto LABEL_88;
              while ( v93 != -4096 )
              {
                if ( !v90 && v93 == -8192 )
                  v90 = v65;
                v91 = ((_DWORD)v91 + v92) & (v115 - 1);
                v65 = (__int64 *)(v113 + 16 * v91);
                v93 = *v65;
                if ( v58 == *v65 )
                  goto LABEL_88;
                ++v92;
              }
              goto LABEL_125;
            }
          }
LABEL_159:
          LODWORD(v114) = v114 + 1;
          BUG();
        }
        if ( v65 || v68 != -8192 )
          v67 = v65;
        v66 = (v115 - 1) & (v63 + (_DWORD)v66);
        v68 = *(_QWORD *)(v113 + 16LL * (unsigned int)v66);
        if ( v58 == v68 )
          break;
        ++v63;
        v65 = v67;
        v67 = (__int64 *)(v113 + 16LL * (unsigned int)v66);
      }
      v67 = (__int64 *)(v113 + 16LL * (unsigned int)v66);
LABEL_80:
      if ( (*((_DWORD *)v67 + 2))-- == 1 )
      {
        v120 = (const char *)v98;
        v121 = v58 | 4;
        sub_FFDB80((__int64)v129, (unsigned __int64 *)&v120, 1, v104, v66, v62);
        v99 = v104;
        goto LABEL_21;
      }
LABEL_91:
      v99 = v60;
LABEL_21:
      if ( v106 == v25 )
        goto LABEL_52;
    }
    if ( v31 > 0x40 )
      v27 = v31 - 1 == (unsigned int)sub_C444A0(v32);
    else
      v27 = *(_QWORD *)(v30 + 24) == 1;
    if ( !v27 )
    {
LABEL_20:
      ++v24;
      ++v25;
      goto LABEL_21;
    }
    v39 = *(_QWORD *)(v124 - 8);
    if ( *(_QWORD *)v39 )
    {
      v40 = *(_QWORD *)(v39 + 8);
      **(_QWORD **)(v39 + 16) = v40;
      if ( v40 )
        *(_QWORD *)(v40 + 16) = *(_QWORD *)(v39 + 16);
    }
    *(_QWORD *)v39 = v28;
    v99 = v27;
    if ( v28 )
    {
      v41 = *(_QWORD *)(v28 + 16);
      *(_QWORD *)(v39 + 8) = v41;
      if ( v41 )
        *(_QWORD *)(v41 + 16) = v39 + 8;
      *(_QWORD *)(v39 + 16) = v28 + 16;
      v99 = v27;
      *(_QWORD *)(v28 + 16) = v39;
    }
LABEL_52:
    v42 = *(_QWORD *)(*(_QWORD *)(v124 - 8) + 32LL);
    v111 = v42;
    if ( v24 > 1 )
    {
      v71 = sub_AA5030(v42, 1);
      if ( !v71 )
        BUG();
      if ( *(_BYTE *)(v71 - 24) != 36 )
      {
        if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
          v72 = *(__int64 **)(a1 - 8);
        else
          v72 = (__int64 *)(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF));
        sub_22CEA30((__int64)&v116, a2, v72, 0);
        if ( !(unsigned __int8)sub_AB0550((__int64)&v116, v24) )
        {
          v75 = *(_QWORD *)(v98 + 72);
          v123 = 1;
          v120 = "default.unreachable";
          v122 = 3;
          v76 = sub_AA48A0(v98);
          v77 = sub_22077B0(0x50u);
          v78 = v77;
          if ( v77 )
            sub_AA4D50(v77, v76, (__int64)&v120, v75, v111);
          v79 = sub_AA48A0(v98);
          sub_B43C20((__int64)&v120, v78);
          v80 = sub_BD2C40(72, unk_3F148B8);
          if ( v80 )
            sub_B4C8A0((__int64)v80, v79, (__int64)v120, v121);
          sub_AA5980(v111, v98, 0);
          v81 = *(_QWORD *)(v124 - 8);
          if ( *(_QWORD *)(v81 + 32) )
          {
            v82 = *(_QWORD *)(v81 + 40);
            **(_QWORD **)(v81 + 48) = v82;
            if ( v82 )
              *(_QWORD *)(v82 + 16) = *(_QWORD *)(v81 + 48);
          }
          *(_QWORD *)(v81 + 32) = v78;
          if ( v78 )
          {
            v83 = *(_QWORD *)(v78 + 16);
            *(_QWORD *)(v81 + 40) = v83;
            if ( v83 )
              *(_QWORD *)(v83 + 16) = v81 + 40;
            *(_QWORD *)(v81 + 48) = v78 + 16;
            *(_QWORD *)(v78 + 16) = v81 + 32;
          }
          if ( *(_DWORD *)sub_274E160((__int64)&v112, &v111) == 1 )
          {
            v120 = (const char *)v98;
            v121 = v111 | 4;
            sub_FFB3D0((__int64)v129, (unsigned __int64 *)&v120, 1, v84, v85, v86);
          }
          v121 = v78 & 0xFFFFFFFFFFFFFFFBLL;
          v120 = (const char *)v98;
          sub_FFB3D0((__int64)v129, (unsigned __int64 *)&v120, 1, v84, v85, v86);
          v99 = 1;
        }
        if ( v119 > 0x40 && v118 )
          j_j___libc_free_0_0(v118);
        if ( v117 > 0x40 && v116 )
          j_j___libc_free_0_0(v116);
      }
    }
  }
  if ( v128 )
  {
    v73 = v124;
    v74 = sub_B53F50((__int64)&v124);
    sub_B99FD0(v73, 2u, v74);
  }
  if ( v127 && v125 != &v126 )
    _libc_free((unsigned __int64)v125);
  if ( v99 )
    sub_F5CD10(v98, 0, 0, (__int64)v129);
  v43 = 16LL * v115;
  sub_C7D6A0(v113, v43, 8);
  sub_FFCE90((__int64)v129, v43, v44, v45, v46, v47);
  sub_FFD870((__int64)v129, v43, v48, v49, v50, v51);
  sub_FFBC40((__int64)v129, v43);
  v52 = v144;
  v53 = v143;
  if ( v144 != v143 )
  {
    do
    {
      v54 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))v53[7];
      *v53 = &unk_49E5048;
      if ( v54 )
        v54(v53 + 5, v53 + 5, 3);
      *v53 = &unk_49DB368;
      v55 = v53[3];
      if ( v55 != 0 && v55 != -4096 && v55 != -8192 )
        sub_BD60C0(v53 + 1);
      v53 += 9;
    }
    while ( v52 != v53 );
    v53 = v143;
  }
  if ( v53 )
    j_j___libc_free_0((unsigned __int64)v53);
  if ( !v140 )
    _libc_free((unsigned __int64)v137);
  if ( (_BYTE *)v129[0] != v130 )
    _libc_free(v129[0]);
  return v99;
}
