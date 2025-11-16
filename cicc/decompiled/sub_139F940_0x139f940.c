// Function: sub_139F940
// Address: 0x139f940
//
char __fastcall sub_139F940(__int64 a1, __int64 a2, __int64 a3, int a4, __int64 a5, _QWORD *a6, __int64 a7, __int64 a8)
{
  __int64 v10; // r8
  unsigned int v11; // eax
  __int64 v12; // rcx
  __int64 v13; // rax
  __int64 v14; // rdx
  unsigned int v16; // r13d
  int v17; // eax
  int v18; // ebx
  unsigned int v19; // r15d
  unsigned __int64 v20; // rdx
  unsigned __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // r14
  unsigned int v24; // r13d
  unsigned int v25; // ebx
  unsigned __int64 v26; // r13
  unsigned int v27; // eax
  __int64 v28; // rsi
  __int64 v29; // rdx
  __int64 v30; // rbx
  unsigned int v31; // r14d
  unsigned int v32; // r13d
  unsigned int v33; // ecx
  unsigned int v34; // edx
  unsigned __int64 v35; // rax
  __int64 v36; // rdx
  __int64 v37; // rbx
  unsigned int v38; // r14d
  unsigned int v39; // ecx
  unsigned int v40; // edx
  unsigned __int64 v41; // rax
  __int64 v42; // rdx
  __int64 v43; // rsi
  unsigned __int64 v44; // r14
  unsigned __int64 v45; // r14
  bool v46; // bl
  unsigned __int64 v47; // rax
  unsigned __int64 v48; // rdi
  char v49; // al
  __int64 *v50; // rdx
  __int64 v51; // rdx
  unsigned __int64 v52; // rsi
  unsigned __int64 v53; // rdx
  unsigned __int64 v54; // rdx
  char v55; // al
  __int64 *v56; // rdx
  __int64 v57; // rdx
  unsigned __int64 v58; // rsi
  unsigned __int64 v59; // rdx
  unsigned __int64 v60; // rdx
  bool v61; // cc
  unsigned int v62; // r8d
  unsigned int v63; // r13d
  char v64; // cl
  unsigned __int64 v65; // r13
  unsigned __int64 v66; // r13
  bool v67; // bl
  unsigned int v68; // edi
  __int64 v69; // rsi
  __int64 v70; // rdx
  unsigned __int64 v71; // rbx
  unsigned __int64 v72; // rdx
  unsigned __int64 v73; // rsi
  unsigned __int64 v74; // rdi
  __int64 v75; // rax
  unsigned __int64 v76; // rsi
  unsigned __int64 v77; // rdi
  __int64 v78; // rax
  unsigned __int64 v79; // rsi
  unsigned int v80; // eax
  unsigned int v81; // r8d
  unsigned int v84; // r13d
  unsigned __int64 v85; // rdx
  unsigned int v86; // ebx
  __int64 v87; // rsi
  __int64 v88; // rdx
  unsigned __int64 v89; // rsi
  unsigned __int64 v90; // rdx
  __int64 v91; // rdx
  __int64 v92; // rsi
  unsigned int v93; // eax
  unsigned int v94; // r8d
  unsigned __int64 v95; // rdx
  int v96; // ecx
  int v97; // eax
  unsigned int v98; // eax
  unsigned int v99; // ebx
  __int64 v100; // rsi
  int v101; // eax
  int v102; // eax
  int v103; // r9d
  unsigned int v104; // eax
  unsigned int v105; // ebx
  int v106; // eax
  int v107; // eax
  __int64 v108; // rax
  __int64 v109; // rax
  unsigned int v110; // ebx
  unsigned int v111; // edi
  __int64 v112; // rsi
  __int64 v113; // rdx
  int v114; // eax
  unsigned __int64 v116; // [rsp+0h] [rbp-C0h]
  unsigned __int64 v117; // [rsp+0h] [rbp-C0h]
  unsigned __int64 v118; // [rsp+0h] [rbp-C0h]
  int v119; // [rsp+Ch] [rbp-B4h]
  int v120; // [rsp+Ch] [rbp-B4h]
  int v121; // [rsp+Ch] [rbp-B4h]
  int v122; // [rsp+Ch] [rbp-B4h]
  int v123; // [rsp+Ch] [rbp-B4h]
  int v124; // [rsp+Ch] [rbp-B4h]
  unsigned int v125; // [rsp+Ch] [rbp-B4h]
  unsigned int v126; // [rsp+Ch] [rbp-B4h]
  unsigned int v127; // [rsp+Ch] [rbp-B4h]
  unsigned int v128; // [rsp+Ch] [rbp-B4h]
  int v129; // [rsp+Ch] [rbp-B4h]
  int v130; // [rsp+Ch] [rbp-B4h]
  int v131; // [rsp+Ch] [rbp-B4h]
  int v132; // [rsp+Ch] [rbp-B4h]
  int v133; // [rsp+Ch] [rbp-B4h]
  int v134; // [rsp+Ch] [rbp-B4h]
  int v135; // [rsp+Ch] [rbp-B4h]
  int v136; // [rsp+Ch] [rbp-B4h]
  int v137; // [rsp+Ch] [rbp-B4h]
  __int64 v138; // [rsp+10h] [rbp-B0h] BYREF
  __int64 v139; // [rsp+18h] [rbp-A8h] BYREF
  unsigned __int64 v140; // [rsp+20h] [rbp-A0h] BYREF
  unsigned int v141; // [rsp+28h] [rbp-98h]
  unsigned __int64 v142; // [rsp+30h] [rbp-90h] BYREF
  unsigned int v143; // [rsp+38h] [rbp-88h]
  unsigned __int64 v144; // [rsp+40h] [rbp-80h] BYREF
  unsigned int v145; // [rsp+48h] [rbp-78h]
  unsigned __int64 v146; // [rsp+50h] [rbp-70h] BYREF
  unsigned int v147; // [rsp+58h] [rbp-68h]
  _QWORD v148[12]; // [rsp+60h] [rbp-60h] BYREF

  v148[2] = a1;
  v10 = *((unsigned int *)a6 + 2);
  v138 = a3;
  v148[0] = &v138;
  v148[3] = &v139;
  v148[4] = a8;
  v11 = *(unsigned __int8 *)(a2 + 16);
  v139 = a2;
  v12 = v11;
  LODWORD(v13) = v11 - 29;
  v148[1] = a7;
  v14 = a2;
  switch ( (int)v13 )
  {
    case 0:
    case 49:
      if ( (_BYTE)v12 == 78 )
      {
        v13 = *(_QWORD *)(a2 - 24);
        if ( !*(_BYTE *)(v13 + 16) && (*(_BYTE *)(v13 + 33) & 0x20) != 0 )
        {
          LODWORD(v13) = *(_DWORD *)(v13 + 36);
          if ( (_DWORD)v13 == 31 )
          {
            if ( !a4 )
            {
              v128 = v10;
              sub_139F520((__int64)v148, v10, v138, 0);
              v93 = *(_DWORD *)(a7 + 24);
              v94 = v128;
              if ( v93 > 0x40 )
              {
                v97 = sub_16A57B0(a7 + 16);
                v94 = v128;
              }
              else
              {
                v95 = *(_QWORD *)(a7 + 16);
                v96 = 64;
                if ( v95 )
                {
                  _BitScanReverse64(&v95, v95);
                  v96 = v95 ^ 0x3F;
                }
                v97 = v93 + v96 - 64;
              }
              v98 = v97 + 1;
              v147 = v94;
              if ( v98 > v94 )
                v98 = v94;
              v99 = v98;
              if ( v94 > 0x40 )
              {
                sub_16A4EF0(&v146, 0, 0);
                v94 = v147;
              }
              else
              {
                v146 = 0;
              }
              v100 = v94 - v99;
              if ( (_DWORD)v100 != v94 )
              {
                if ( (unsigned int)v100 > 0x3F || v94 > 0x40 )
                  sub_16A5260(&v146, v100, v94);
                else
                  v146 |= 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v99) << ((unsigned __int8)v94
                                                                                - (unsigned __int8)v99);
              }
              goto LABEL_3;
            }
          }
          else if ( (unsigned int)v13 > 0x1F )
          {
            if ( (_DWORD)v13 == 33 && !a4 )
            {
              v127 = v10;
              sub_139F520((__int64)v148, v10, v138, 0);
              v80 = *(_DWORD *)(a7 + 24);
              v81 = v127;
              if ( v80 > 0x40 )
              {
                v114 = sub_16A58A0(a7 + 16);
                v81 = v127;
                LODWORD(_R13) = v114;
              }
              else
              {
                _RDX = *(_QWORD *)(a7 + 16);
                __asm { tzcnt   r13, rdx }
                if ( !_RDX )
                  LODWORD(_R13) = 64;
                if ( v80 < (unsigned int)_R13 )
                  LODWORD(_R13) = *(_DWORD *)(a7 + 24);
              }
              v84 = _R13 + 1;
              v147 = v81;
              if ( v84 > v81 )
                v84 = v81;
              if ( v81 > 0x40 )
                sub_16A4EF0(&v146, 0, 0);
              else
                v146 = 0;
              if ( v84 )
              {
                if ( v84 > 0x40 )
                {
                  sub_16A5260(&v146, 0, v84);
                }
                else
                {
                  v85 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v84);
                  if ( v147 > 0x40 )
                    *(_QWORD *)v146 |= v85;
                  else
                    v146 |= v85;
                }
              }
              goto LABEL_3;
            }
          }
          else
          {
            if ( (_DWORD)v13 == 5 )
            {
              sub_16A8270(&v146, a5, a2, v12, v10);
              if ( *((_DWORD *)a6 + 2) <= 0x40u )
                goto LABEL_6;
              goto LABEL_4;
            }
            if ( (_DWORD)v13 == 6 )
            {
              sub_16A85B0(&v146, a5, a2, v12, v10);
              if ( *((_DWORD *)a6 + 2) <= 0x40u )
                goto LABEL_6;
              goto LABEL_4;
            }
          }
        }
      }
      return v13;
    case 6:
    case 8:
    case 10:
      v16 = *(_DWORD *)(a5 + 8);
      if ( v16 <= 0x40 )
      {
        v18 = *(_DWORD *)(a5 + 8);
        if ( *(_QWORD *)a5 )
        {
          _BitScanReverse64(&v71, *(_QWORD *)a5);
          v18 = v16 - 64 + (v71 ^ 0x3F);
        }
      }
      else
      {
        v119 = v10;
        v17 = sub_16A57B0(a5);
        LODWORD(v10) = v119;
        v18 = v17;
      }
      v147 = v10;
      v19 = v16 - v18;
      if ( (unsigned int)v10 <= 0x40 )
      {
        v146 = 0;
        v13 = 0;
        if ( !v19 )
          goto LABEL_19;
        if ( v19 <= 0x40 )
        {
          v20 = 0;
          v21 = 0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v18 - (unsigned __int8)v16 + 64);
LABEL_14:
          v146 = v20 | v21;
          goto LABEL_15;
        }
LABEL_225:
        sub_16A5260(&v146, 0, v19);
        goto LABEL_15;
      }
      sub_16A4EF0(&v146, 0, 0);
      if ( !v19 )
        goto LABEL_15;
      if ( v19 > 0x40 )
        goto LABEL_225;
      v20 = v146;
      v21 = 0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v18 - (unsigned __int8)v16 + 64);
      if ( v147 <= 0x40 )
        goto LABEL_14;
      *(_QWORD *)v146 |= v21;
LABEL_15:
      if ( *((_DWORD *)a6 + 2) > 0x40u && *a6 )
        j_j___libc_free_0_0(*a6);
      v13 = v146;
      LODWORD(v10) = v147;
LABEL_19:
      *a6 = v13;
      *((_DWORD *)a6 + 2) = v10;
      return v13;
    case 18:
      if ( a4 )
        return v13;
      if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
      {
        v22 = *(_QWORD *)(a2 - 8);
      }
      else
      {
        v13 = 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
        v22 = a2 - v13;
      }
      v23 = *(_QWORD *)(v22 + 24);
      if ( *(_BYTE *)(v23 + 16) != 13 )
        return v13;
      v24 = *(_DWORD *)(v23 + 32);
      v25 = v10 - 1;
      if ( v24 > 0x40 )
      {
        v130 = v10;
        v117 = (unsigned int)(v10 - 1);
        v102 = sub_16A57B0(v23 + 24);
        LODWORD(v10) = v130;
        v103 = v102;
        v104 = v24;
        LODWORD(v26) = v25;
        if ( v104 - v103 <= 0x40 )
        {
          v26 = **(_QWORD **)(v23 + 24);
          if ( v117 < v26 )
            LODWORD(v26) = v25;
        }
      }
      else
      {
        v26 = *(_QWORD *)(v23 + 24);
        if ( (unsigned int)(v10 - 1) < v26 )
          LODWORD(v26) = v10 - 1;
      }
      v27 = *(_DWORD *)(a5 + 8);
      v147 = v27;
      if ( v27 > 0x40 )
      {
        v134 = v10;
        sub_16A4FD0(&v146, a5);
        v27 = v147;
        LODWORD(v10) = v134;
        if ( v147 > 0x40 )
        {
          sub_16A8110(&v146, (unsigned int)v26);
          LODWORD(v10) = v134;
          goto LABEL_44;
        }
      }
      else
      {
        v146 = *(_QWORD *)a5;
      }
      if ( v27 == (_DWORD)v26 )
        v146 = 0;
      else
        v146 >>= v26;
LABEL_44:
      if ( *((_DWORD *)a6 + 2) > 0x40u && *a6 )
      {
        v120 = v10;
        j_j___libc_free_0_0(*a6);
        LODWORD(v10) = v120;
      }
      *a6 = v146;
      *((_DWORD *)a6 + 2) = v147;
      LOBYTE(v13) = *(_BYTE *)(v139 + 17) >> 1;
      if ( (v13 & 2) != 0 )
      {
        v147 = v10;
        if ( (unsigned int)v10 > 0x40 )
        {
          sub_16A4EF0(&v146, 0, 0);
          LODWORD(v10) = v147;
          v25 = v147 - 1;
        }
        else
        {
          v146 = 0;
        }
        v110 = v25 - v26;
        if ( v110 != (_DWORD)v10 )
        {
          if ( v110 > 0x3F || (unsigned int)v10 > 0x40 )
            sub_16A5260(&v146, v110, (unsigned int)v10);
          else
            v146 |= 0xFFFFFFFFFFFFFFFFLL >> (63 - (unsigned __int8)v26) << v110;
        }
        goto LABEL_102;
      }
      if ( (*(_BYTE *)(v139 + 17) & 2) != 0 )
      {
        v147 = v10;
        if ( (unsigned int)v10 > 0x40 )
        {
          sub_16A4EF0(&v146, 0, 0);
          LODWORD(v10) = v147;
        }
        else
        {
          v146 = 0;
        }
        v28 = (unsigned int)(v10 - v26);
        if ( (_DWORD)v28 != (_DWORD)v10 )
        {
          if ( (unsigned int)v28 > 0x3F || (unsigned int)v10 > 0x40 )
            sub_16A5260(&v146, v28, (unsigned int)v10);
          else
            v146 |= 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v26) << ((unsigned __int8)v10 - (unsigned __int8)v26);
        }
        goto LABEL_102;
      }
      return v13;
    case 19:
      if ( a4 )
        return v13;
      if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
      {
        v29 = *(_QWORD *)(a2 - 8);
      }
      else
      {
        v13 = 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
        v29 = a2 - v13;
      }
      v30 = *(_QWORD *)(v29 + 24);
      if ( *(_BYTE *)(v30 + 16) != 13 )
        return v13;
      v31 = *(_DWORD *)(v30 + 32);
      v32 = v10 - 1;
      if ( v31 > 0x40 )
      {
        v133 = v10;
        v118 = (unsigned int)(v10 - 1);
        v107 = sub_16A57B0(v30 + 24);
        LODWORD(v10) = v133;
        if ( v31 - v107 <= 0x40 && v118 >= **(_QWORD **)(v30 + 24) )
          v32 = **(_QWORD **)(v30 + 24);
      }
      else if ( (unsigned __int64)(unsigned int)(v10 - 1) >= *(_QWORD *)(v30 + 24) )
      {
        v32 = *(_QWORD *)(v30 + 24);
      }
      v33 = *(_DWORD *)(a5 + 8);
      v147 = v33;
      if ( v33 > 0x40 )
      {
        v135 = v10;
        sub_16A4FD0(&v146, a5);
        v33 = v147;
        LODWORD(v10) = v135;
        if ( v147 > 0x40 )
        {
          sub_16A7DC0(&v146, v32);
          v34 = *((_DWORD *)a6 + 2);
          LODWORD(v10) = v135;
          goto LABEL_67;
        }
        v34 = *((_DWORD *)a6 + 2);
      }
      else
      {
        v34 = v10;
        v146 = *(_QWORD *)a5;
      }
      v35 = 0;
      if ( v33 != v32 )
        v35 = (v146 << v32) & (0xFFFFFFFFFFFFFFFFLL >> -(char)v33);
      v146 = v35;
LABEL_67:
      if ( v34 > 0x40 && *a6 )
      {
        v121 = v10;
        j_j___libc_free_0_0(*a6);
        LODWORD(v10) = v121;
      }
      *a6 = v146;
      *((_DWORD *)a6 + 2) = v147;
      LOBYTE(v13) = v139;
      if ( (*(_BYTE *)(v139 + 17) & 2) != 0 )
        goto LABEL_96;
      return v13;
    case 20:
      if ( a4 )
        return v13;
      if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
      {
        v36 = *(_QWORD *)(a2 - 8);
      }
      else
      {
        v13 = 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
        v36 = a2 - v13;
      }
      v37 = *(_QWORD *)(v36 + 24);
      if ( *(_BYTE *)(v37 + 16) != 13 )
        return v13;
      v38 = *(_DWORD *)(v37 + 32);
      v32 = v10 - 1;
      if ( v38 > 0x40 )
      {
        v129 = v10;
        v116 = (unsigned int)(v10 - 1);
        v101 = sub_16A57B0(v37 + 24);
        LODWORD(v10) = v129;
        if ( v38 - v101 <= 0x40 && v116 >= **(_QWORD **)(v37 + 24) )
          v32 = **(_QWORD **)(v37 + 24);
      }
      else if ( (unsigned __int64)(unsigned int)(v10 - 1) >= *(_QWORD *)(v37 + 24) )
      {
        v32 = *(_QWORD *)(v37 + 24);
      }
      v39 = *(_DWORD *)(a5 + 8);
      v147 = v39;
      if ( v39 > 0x40 )
      {
        v136 = v10;
        sub_16A4FD0(&v146, a5);
        v39 = v147;
        LODWORD(v10) = v136;
        if ( v147 > 0x40 )
        {
          sub_16A7DC0(&v146, v32);
          v40 = *((_DWORD *)a6 + 2);
          LODWORD(v10) = v136;
          goto LABEL_84;
        }
        v40 = *((_DWORD *)a6 + 2);
      }
      else
      {
        v40 = v10;
        v146 = *(_QWORD *)a5;
      }
      v41 = 0;
      if ( v32 != v39 )
        v41 = (v146 << v32) & (0xFFFFFFFFFFFFFFFFLL >> -(char)v39);
      v146 = v41;
LABEL_84:
      if ( v40 > 0x40 && *a6 )
      {
        v122 = v10;
        j_j___libc_free_0_0(*a6);
        LODWORD(v10) = v122;
      }
      v145 = v10;
      *a6 = v146;
      *((_DWORD *)a6 + 2) = v147;
      if ( (unsigned int)v10 > 0x40 )
      {
        v131 = v10;
        sub_16A4EF0(&v144, 0, 0);
        v42 = v145;
        LODWORD(v10) = v131;
        v43 = v145 - v32;
        if ( v145 == (_DWORD)v43 )
          goto LABEL_262;
      }
      else
      {
        v144 = 0;
        v42 = (unsigned int)v10;
        v43 = (unsigned int)v10 - v32;
        if ( (_DWORD)v10 == (_DWORD)v43 )
        {
          v44 = 0;
          goto LABEL_92;
        }
      }
      if ( (unsigned int)v43 <= 0x3F && (unsigned int)v42 <= 0x40 )
      {
        v44 = v144 | (0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v32) << v43);
LABEL_92:
        v45 = *(_QWORD *)a5 & v44;
LABEL_93:
        v46 = v45 == 0;
        goto LABEL_94;
      }
      v137 = v10;
      sub_16A5260(&v144, v43, v42);
      LODWORD(v43) = v145;
      LODWORD(v10) = v137;
LABEL_262:
      v44 = v144;
      if ( (unsigned int)v43 <= 0x40 )
        goto LABEL_92;
      v132 = v10;
      sub_16A8890(&v144, a5);
      v105 = v145;
      v45 = v144;
      v145 = 0;
      LODWORD(v10) = v132;
      v147 = v105;
      v146 = v144;
      if ( v105 <= 0x40 )
        goto LABEL_93;
      v106 = sub_16A57B0(&v146);
      LODWORD(v10) = v132;
      v46 = v105 == v106;
      if ( v45 )
      {
        j_j___libc_free_0_0(v45);
        LODWORD(v10) = v132;
        if ( v145 > 0x40 )
        {
          if ( v144 )
          {
            j_j___libc_free_0_0(v144);
            LODWORD(v10) = v132;
          }
        }
      }
LABEL_94:
      if ( !v46 )
      {
        v111 = *((_DWORD *)a6 + 2);
        v112 = *a6;
        v113 = 1LL << ((unsigned __int8)v111 - 1);
        if ( v111 > 0x40 )
          *(_QWORD *)(v112 + 8LL * ((v111 - 1) >> 6)) |= v113;
        else
          *a6 = v113 | v112;
      }
      LOBYTE(v13) = v139;
      if ( (*(_BYTE *)(v139 + 17) & 2) != 0 )
      {
LABEL_96:
        v147 = v10;
        if ( (unsigned int)v10 > 0x40 )
          sub_16A4EF0(&v146, 0, 0);
        else
          v146 = 0;
        if ( v32 )
        {
          if ( v32 > 0x40 )
          {
            sub_16A5260(&v146, 0, v32);
          }
          else
          {
            v47 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v32);
            if ( v147 > 0x40 )
              *(_QWORD *)v146 |= v47;
            else
              v146 |= v47;
          }
        }
LABEL_102:
        if ( *((_DWORD *)a6 + 2) > 0x40u )
        {
          LOBYTE(v13) = sub_16A89F0(a6, &v146);
        }
        else
        {
          LOBYTE(v13) = v146;
          *a6 |= v146;
        }
        if ( v147 > 0x40 )
        {
          v48 = v146;
          if ( v146 )
            goto LABEL_106;
        }
      }
      return v13;
    case 21:
      if ( (unsigned int)v10 <= 0x40 && *(_DWORD *)(a5 + 8) <= 0x40u )
      {
        v77 = *(_QWORD *)a5;
        *a6 = *(_QWORD *)a5;
        v78 = *(unsigned int *)(a5 + 8);
        *((_DWORD *)a6 + 2) = v78;
        v79 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v78;
        if ( (unsigned int)v78 > 0x40 )
        {
          v108 = (unsigned int)((unsigned __int64)(v78 + 63) >> 6) - 1;
          *(_QWORD *)(v77 + 8 * v108) &= v79;
          v14 = v139;
        }
        else
        {
          *a6 = v77 & v79;
        }
      }
      else
      {
        v124 = v10;
        sub_16A51C0(a6, a5);
        v14 = v139;
        LODWORD(v10) = v124;
      }
      v55 = *(_BYTE *)(v14 + 23) & 0x40;
      if ( a4 )
      {
        if ( v55 )
          v56 = *(__int64 **)(v14 - 8);
        else
          v56 = (__int64 *)(v14 - 24LL * (*(_DWORD *)(v14 + 20) & 0xFFFFFFF));
        v57 = *v56;
        if ( *(_BYTE *)(v57 + 16) <= 0x17u )
          sub_139F520((__int64)v148, v10, v57, v138);
        LODWORD(v13) = *(_DWORD *)(a8 + 8);
        v141 = v13;
        if ( (unsigned int)v13 <= 0x40 )
        {
          v58 = *(_QWORD *)a8;
LABEL_129:
          v141 = 0;
          v59 = ~v58 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v13);
          v140 = v59;
LABEL_130:
          v143 = 0;
          v54 = *(_QWORD *)a7 & v59;
          v142 = v54;
          goto LABEL_131;
        }
        sub_16A4FD0(&v140, a8);
        LODWORD(v13) = v141;
        if ( v141 <= 0x40 )
        {
          v58 = v140;
          goto LABEL_129;
        }
        sub_16A8F40(&v140);
        LODWORD(v13) = v141;
        v59 = v140;
        v87 = a7;
        v141 = 0;
        v143 = v13;
        v142 = v140;
        if ( (unsigned int)v13 <= 0x40 )
          goto LABEL_130;
        goto LABEL_203;
      }
      if ( v55 )
        v88 = *(_QWORD *)(v14 - 8);
      else
        v88 = v14 - 24LL * (*(_DWORD *)(v14 + 20) & 0xFFFFFFF);
      sub_139F520((__int64)v148, v10, v138, *(_QWORD *)(v88 + 24));
      LODWORD(v13) = *(_DWORD *)(a8 + 8);
      v145 = v13;
      if ( (unsigned int)v13 <= 0x40 )
      {
        v89 = *(_QWORD *)a8;
        goto LABEL_212;
      }
      v92 = a8;
      goto LABEL_232;
    case 22:
      if ( (unsigned int)v10 <= 0x40 && *(_DWORD *)(a5 + 8) <= 0x40u )
      {
        v74 = *(_QWORD *)a5;
        *a6 = *(_QWORD *)a5;
        v75 = *(unsigned int *)(a5 + 8);
        *((_DWORD *)a6 + 2) = v75;
        v76 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v75;
        if ( (unsigned int)v75 > 0x40 )
        {
          v109 = (unsigned int)((unsigned __int64)(v75 + 63) >> 6) - 1;
          *(_QWORD *)(v74 + 8 * v109) &= v76;
          v14 = v139;
        }
        else
        {
          *a6 = v74 & v76;
        }
      }
      else
      {
        v123 = v10;
        sub_16A51C0(a6, a5);
        v14 = v139;
        LODWORD(v10) = v123;
      }
      v49 = *(_BYTE *)(v14 + 23) & 0x40;
      if ( !a4 )
      {
        if ( v49 )
          v91 = *(_QWORD *)(v14 - 8);
        else
          v91 = v14 - 24LL * (*(_DWORD *)(v14 + 20) & 0xFFFFFFF);
        sub_139F520((__int64)v148, v10, v138, *(_QWORD *)(v91 + 24));
        LODWORD(v13) = *(_DWORD *)(a8 + 24);
        v145 = v13;
        if ( (unsigned int)v13 > 0x40 )
        {
          v92 = a8 + 16;
LABEL_232:
          sub_16A4FD0(&v144, v92);
          LODWORD(v13) = v145;
          if ( v145 > 0x40 )
          {
            sub_16A8F40(&v144);
            LODWORD(v13) = v145;
            v90 = v144;
LABEL_213:
            v61 = *((_DWORD *)a6 + 2) <= 0x40u;
            v147 = v13;
            v146 = v90;
            v145 = 0;
            if ( v61 )
            {
              *a6 &= v90;
            }
            else
            {
              sub_16A8890(a6, &v146);
              LODWORD(v13) = v147;
            }
            if ( (unsigned int)v13 > 0x40 && v146 )
              LOBYTE(v13) = j_j___libc_free_0_0(v146);
            if ( v145 <= 0x40 )
              return v13;
            v48 = v144;
            if ( !v144 )
              return v13;
LABEL_106:
            LOBYTE(v13) = j_j___libc_free_0_0(v48);
            return v13;
          }
          v89 = v144;
        }
        else
        {
          v89 = *(_QWORD *)(a8 + 16);
        }
LABEL_212:
        v90 = ~v89 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v13);
        v144 = v90;
        goto LABEL_213;
      }
      if ( v49 )
        v50 = *(__int64 **)(v14 - 8);
      else
        v50 = (__int64 *)(v14 - 24LL * (*(_DWORD *)(v14 + 20) & 0xFFFFFFF));
      v51 = *v50;
      if ( *(_BYTE *)(v51 + 16) <= 0x17u )
        sub_139F520((__int64)v148, v10, v51, v138);
      LODWORD(v13) = *(_DWORD *)(a8 + 24);
      v141 = v13;
      if ( (unsigned int)v13 <= 0x40 )
      {
        v52 = *(_QWORD *)(a8 + 16);
LABEL_117:
        v141 = 0;
        v53 = ~v52 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v13);
        v140 = v53;
LABEL_118:
        v143 = 0;
        v54 = *(_QWORD *)(a7 + 16) & v53;
        v142 = v54;
        goto LABEL_131;
      }
      sub_16A4FD0(&v140, a8 + 16);
      LODWORD(v13) = v141;
      if ( v141 <= 0x40 )
      {
        v52 = v140;
        goto LABEL_117;
      }
      sub_16A8F40(&v140);
      LODWORD(v13) = v141;
      v87 = a7 + 16;
      v53 = v140;
      v141 = 0;
      v143 = v13;
      v142 = v140;
      if ( (unsigned int)v13 <= 0x40 )
        goto LABEL_118;
LABEL_203:
      sub_16A8890(&v142, v87);
      LODWORD(v13) = v143;
      v54 = v142;
      v143 = 0;
      v145 = v13;
      v144 = v142;
      if ( (unsigned int)v13 <= 0x40 )
      {
LABEL_131:
        v60 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v13) & ~v54;
        v144 = v60;
      }
      else
      {
        sub_16A8F40(&v144);
        LODWORD(v13) = v145;
        v60 = v144;
      }
      v61 = *((_DWORD *)a6 + 2) <= 0x40u;
      v147 = v13;
      v146 = v60;
      v145 = 0;
      if ( v61 )
      {
        *a6 &= v60;
      }
      else
      {
        sub_16A8890(a6, &v146);
        LODWORD(v13) = v147;
      }
      if ( (unsigned int)v13 > 0x40 && v146 )
        LOBYTE(v13) = j_j___libc_free_0_0(v146);
      if ( v145 > 0x40 && v144 )
        LOBYTE(v13) = j_j___libc_free_0_0(v144);
      if ( v143 > 0x40 && v142 )
        LOBYTE(v13) = j_j___libc_free_0_0(v142);
      if ( v141 > 0x40 )
      {
        v48 = v140;
        if ( v140 )
          goto LABEL_106;
      }
      return v13;
    case 23:
    case 48:
      goto LABEL_21;
    case 31:
      sub_16A5C50(&v146, a5, (unsigned int)v10);
      if ( *((_DWORD *)a6 + 2) > 0x40u )
        goto LABEL_4;
      goto LABEL_6;
    case 32:
      sub_16A5A50(&v146, a5);
LABEL_3:
      if ( *((_DWORD *)a6 + 2) > 0x40u )
      {
LABEL_4:
        if ( *a6 )
          j_j___libc_free_0_0(*a6);
      }
LABEL_6:
      *a6 = v146;
      LOBYTE(v13) = v147;
      *((_DWORD *)a6 + 2) = v147;
      return v13;
    case 33:
      v125 = v10;
      sub_16A5A50(&v146, a5);
      v62 = v125;
      if ( *((_DWORD *)a6 + 2) > 0x40u && *a6 )
      {
        j_j___libc_free_0_0(*a6);
        v62 = v125;
      }
      *a6 = v146;
      LOBYTE(v13) = v147;
      *((_DWORD *)a6 + 2) = v147;
      v63 = *(_DWORD *)(a5 + 8);
      v145 = v63;
      if ( v63 <= 0x40 )
      {
        v144 = 0;
        v64 = v62 - v63;
        if ( v62 == v63 )
        {
          v65 = 0;
          goto LABEL_157;
        }
      }
      else
      {
        v126 = v62;
        sub_16A4EF0(&v144, 0, 0);
        LOBYTE(v13) = v145;
        v64 = v126 - v63;
        v62 = v145 + v126 - v63;
        if ( v145 == v62 )
          goto LABEL_195;
        v63 = v145;
      }
      if ( v62 <= 0x3F && v63 <= 0x40 )
      {
        v65 = v144 | (0xFFFFFFFFFFFFFFFFLL >> (v64 + 64) << v62);
        goto LABEL_157;
      }
      LOBYTE(v13) = sub_16A5260(&v144, v62, v63);
      v62 = v145;
LABEL_195:
      v65 = v144;
      if ( v62 > 0x40 )
      {
        LOBYTE(v13) = sub_16A8890(&v144, a5);
        v86 = v145;
        v66 = v144;
        v145 = 0;
        v147 = v86;
        v146 = v144;
        if ( v86 > 0x40 )
        {
          LODWORD(v13) = sub_16A57B0(&v146);
          v67 = v86 == (_DWORD)v13;
          if ( v66 )
          {
            LOBYTE(v13) = j_j___libc_free_0_0(v66);
            if ( v145 > 0x40 )
            {
              if ( v144 )
                LOBYTE(v13) = j_j___libc_free_0_0(v144);
            }
          }
LABEL_159:
          if ( !v67 )
          {
            v68 = *((_DWORD *)a6 + 2);
            v69 = *a6;
            v70 = 1LL << ((unsigned __int8)v68 - 1);
            if ( v68 > 0x40 )
            {
              v13 = (v68 - 1) >> 6;
              *(_QWORD *)(v69 + 8 * v13) |= v70;
            }
            else
            {
              *a6 = v70 | v69;
              LOBYTE(v13) = v70 | v69;
            }
          }
          return v13;
        }
LABEL_158:
        v67 = v66 == 0;
        goto LABEL_159;
      }
LABEL_157:
      v66 = *(_QWORD *)a5 & v65;
      goto LABEL_158;
    case 50:
      if ( a4 )
      {
LABEL_21:
        if ( (unsigned int)v10 <= 0x40 && *(_DWORD *)(a5 + 8) <= 0x40u )
        {
          v72 = *(_QWORD *)a5;
          *a6 = *(_QWORD *)a5;
          v13 = *(unsigned int *)(a5 + 8);
          *((_DWORD *)a6 + 2) = v13;
          v73 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v13;
          if ( (unsigned int)v13 > 0x40 )
          {
            v13 = (unsigned int)((unsigned __int64)(v13 + 63) >> 6) - 1;
            *(_QWORD *)(v72 + 8 * v13) &= v73;
          }
          else
          {
            *a6 = v73 & v72;
          }
        }
        else
        {
          LOBYTE(v13) = sub_16A51C0(a6, a5);
        }
      }
      return v13;
    default:
      return v13;
  }
}
