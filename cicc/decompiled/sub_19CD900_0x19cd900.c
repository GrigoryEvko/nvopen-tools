// Function: sub_19CD900
// Address: 0x19cd900
//
__int64 __fastcall sub_19CD900(
        __int64 a1,
        __m128 a2,
        double a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        __m128 a9)
{
  __int64 v9; // rbx
  __int64 v10; // r9
  unsigned __int64 v11; // r13
  double v12; // xmm4_8
  double v13; // xmm5_8
  char v14; // al
  __int64 v15; // r14
  unsigned __int8 v16; // al
  __int64 v17; // r15
  _QWORD *v18; // r14
  __int64 v19; // rax
  _QWORD *v20; // r13
  char v21; // al
  unsigned __int64 v22; // rbx
  __int64 v23; // r12
  unsigned __int8 v24; // al
  __int64 v25; // rdi
  int v26; // r8d
  __int64 v27; // rax
  __int64 v28; // rax
  _QWORD *v29; // rax
  __int64 v30; // rdx
  __int64 v31; // rdx
  unsigned __int64 v32; // rax
  __int64 v33; // rax
  __int64 v34; // rax
  __int64 v35; // r12
  int v37; // eax
  __int64 v38; // rdx
  __int64 v39; // rax
  __int64 v40; // rdx
  unsigned __int8 v41; // al
  __int64 v42; // r12
  __int64 v43; // r8
  __int64 v44; // rax
  __int64 *v45; // rax
  __int64 v46; // r12
  __int64 v47; // rdx
  __int64 v48; // rdx
  __int64 v49; // rsi
  __int64 v50; // r15
  unsigned int v51; // r10d
  __int64 v52; // r11
  __int64 v53; // r14
  unsigned int v54; // ecx
  __int64 v55; // rdx
  __int64 v56; // rdi
  __int64 v57; // rdi
  __int64 v58; // rdi
  __int64 v59; // rdi
  int v60; // eax
  __int64 v61; // rdx
  unsigned int *v62; // rcx
  unsigned int *v63; // rdx
  __int64 v64; // rax
  _QWORD *v65; // r13
  __int64 v66; // rdx
  unsigned __int64 v67; // rcx
  __int64 v68; // rdx
  int v69; // eax
  __int64 v70; // rax
  __int64 v71; // rdx
  unsigned int v72; // eax
  __int64 v73; // r12
  __int64 v74; // r13
  __int64 v75; // rdx
  __int64 v76; // rax
  unsigned int *v77; // rbx
  unsigned int *v78; // r15
  __int64 v79; // rsi
  __int64 v80; // rax
  int v81; // edx
  bool v82; // bl
  __int64 v83; // rdx
  unsigned __int64 v84; // rax
  __int64 v85; // rbx
  __int64 v86; // rdi
  unsigned __int64 v87; // rax
  __int64 v88; // rdx
  __int64 v89; // rax
  __int64 v90; // rdx
  __int64 v91; // rax
  __int64 v92; // rdx
  unsigned __int64 v93; // rax
  __int64 v94; // rax
  __int64 v95; // rcx
  __int64 v96; // rdx
  unsigned int v97; // edx
  unsigned int v98; // esi
  __int64 v99; // rax
  int v100; // r15d
  __int64 v101; // rax
  __int64 v102; // rdi
  _QWORD *v103; // rdx
  __int64 v104; // rax
  __int64 v105; // rax
  __int64 v106; // r12
  _QWORD *v107; // rax
  __int64 v108; // rdx
  __int64 v109; // rdx
  unsigned __int64 v110; // rcx
  __int64 v111; // rdx
  __int64 v112; // rax
  __int64 v113; // r14
  __int64 v114; // rdx
  unsigned __int64 v115; // rcx
  __int64 v116; // rdx
  __int64 v117; // rax
  unsigned __int64 v118; // r13
  __int64 v119; // rcx
  __int64 v120; // rax
  __int64 v121; // rcx
  __int64 v122; // rcx
  unsigned __int64 v123; // [rsp+8h] [rbp-108h]
  _QWORD *v124; // [rsp+10h] [rbp-100h]
  __int64 v125; // [rsp+20h] [rbp-F0h]
  __int64 v126; // [rsp+30h] [rbp-E0h]
  __int64 v127; // [rsp+30h] [rbp-E0h]
  __int64 v128; // [rsp+38h] [rbp-D8h]
  unsigned int v129; // [rsp+38h] [rbp-D8h]
  unsigned __int8 v130; // [rsp+40h] [rbp-D0h]
  __int64 v131; // [rsp+40h] [rbp-D0h]
  __int64 v132; // [rsp+40h] [rbp-D0h]
  __int64 v133; // [rsp+48h] [rbp-C8h]
  int v134; // [rsp+50h] [rbp-C0h]
  _QWORD *v135; // [rsp+58h] [rbp-B8h]
  __int64 v136; // [rsp+58h] [rbp-B8h]
  __int64 v137; // [rsp+58h] [rbp-B8h]
  __int64 v138; // [rsp+58h] [rbp-B8h]
  int v139; // [rsp+60h] [rbp-B0h]
  _QWORD *v140; // [rsp+60h] [rbp-B0h]
  _QWORD *v141; // [rsp+60h] [rbp-B0h]
  __int64 v142; // [rsp+68h] [rbp-A8h]
  unsigned __int64 v143; // [rsp+70h] [rbp-A0h] BYREF
  unsigned int v144; // [rsp+78h] [rbp-98h]
  unsigned __int64 v145; // [rsp+80h] [rbp-90h] BYREF
  unsigned int v146; // [rsp+88h] [rbp-88h]
  unsigned int *v147; // [rsp+90h] [rbp-80h] BYREF
  __int64 v148; // [rsp+98h] [rbp-78h]
  _BYTE v149[112]; // [rsp+A0h] [rbp-70h] BYREF

  v133 = a1 + 72;
  v130 = 0;
  v142 = *(_QWORD *)(a1 + 80);
  if ( v142 != a1 + 72 )
  {
    while ( 1 )
    {
      v9 = v142 - 24;
      if ( !v142 )
        v9 = 0;
      v11 = sub_157EBA0(v9);
      v14 = *(_BYTE *)(v11 + 16);
      if ( v14 != 26 )
        break;
      if ( (*(_DWORD *)(v11 + 20) & 0xFFFFFFF) != 1 )
      {
        v15 = *(_QWORD *)(v11 - 72);
        v16 = *(_BYTE *)(v15 + 16);
        if ( v16 > 0x17u )
        {
          if ( v16 == 75 )
          {
            v100 = *(_WORD *)(v15 + 18) & 0x7FFF;
            if ( (unsigned int)(v100 - 32) > 1 )
              goto LABEL_9;
            v101 = *(_QWORD *)(v15 - 24);
            if ( *(_BYTE *)(v101 + 16) != 13 )
              goto LABEL_9;
            v102 = *(_QWORD *)(v15 - 48);
            if ( *(_BYTE *)(v102 + 16) != 78 || *(_DWORD *)(v101 + 32) > 0x40u )
              goto LABEL_9;
            v103 = *(_QWORD **)(v101 + 24);
          }
          else
          {
            if ( v16 != 78 )
              goto LABEL_9;
            v102 = *(_QWORD *)(v11 - 72);
            v100 = 33;
            v15 = 0;
            v103 = 0;
          }
          v104 = *(_QWORD *)(v102 - 24);
          v141 = v103;
          if ( !*(_BYTE *)(v104 + 16) && *(_DWORD *)(v104 + 36) == 56 )
          {
            v105 = *(_DWORD *)(v102 + 20) & 0xFFFFFFF;
            v137 = *(_QWORD *)(v102 + 24 * (1 - v105));
            if ( *(_BYTE *)(v137 + 16) == 13 )
            {
              v106 = *(_QWORD *)(v102 - 24 * v105);
              v147 = (unsigned int *)sub_16498A0(v102);
              v107 = *(_QWORD **)(v137 + 24);
              if ( *(_DWORD *)(v137 + 32) > 0x40u )
                v107 = (_QWORD *)*v107;
              if ( (v141 == v107) == (v100 == 32) )
                v108 = sub_161BE60(&v147, dword_4FB3AA0, dword_4FB39C0);
              else
                v108 = sub_161BE60(&v147, dword_4FB39C0, dword_4FB3AA0);
              sub_1625C10(v11, 2, v108);
              if ( v15 )
              {
                if ( *(_QWORD *)(v15 - 48) )
                {
                  v109 = *(_QWORD *)(v15 - 40);
                  v110 = *(_QWORD *)(v15 - 32) & 0xFFFFFFFFFFFFFFFCLL;
                  *(_QWORD *)v110 = v109;
                  if ( v109 )
                    *(_QWORD *)(v109 + 16) = v110 | *(_QWORD *)(v109 + 16) & 3LL;
                }
                *(_QWORD *)(v15 - 48) = v106;
                if ( v106 )
                {
                  v111 = *(_QWORD *)(v106 + 8);
                  *(_QWORD *)(v15 - 40) = v111;
                  if ( v111 )
                    *(_QWORD *)(v111 + 16) = (v15 - 40) | *(_QWORD *)(v111 + 16) & 3LL;
                  v112 = *(_QWORD *)(v15 - 32);
                  v113 = v15 - 48;
                  *(_QWORD *)(v113 + 16) = (v106 + 8) | v112 & 3;
                  *(_QWORD *)(v106 + 8) = v113;
                }
              }
              else
              {
                if ( *(_QWORD *)(v11 - 72) )
                {
                  v114 = *(_QWORD *)(v11 - 64);
                  v115 = *(_QWORD *)(v11 - 56) & 0xFFFFFFFFFFFFFFFCLL;
                  *(_QWORD *)v115 = v114;
                  if ( v114 )
                    *(_QWORD *)(v114 + 16) = v115 | *(_QWORD *)(v114 + 16) & 3LL;
                }
                *(_QWORD *)(v11 - 72) = v106;
                if ( v106 )
                {
                  v116 = *(_QWORD *)(v106 + 8);
                  *(_QWORD *)(v11 - 64) = v116;
                  if ( v116 )
                    *(_QWORD *)(v116 + 16) = (v11 - 64) | *(_QWORD *)(v116 + 16) & 3LL;
                  v117 = *(_QWORD *)(v11 - 56);
                  v118 = v11 - 72;
                  *(_QWORD *)(v118 + 16) = (v106 + 8) | v117 & 3;
                  *(_QWORD *)(v106 + 8) = v118;
                }
              }
            }
          }
        }
      }
LABEL_9:
      v17 = v9 + 40;
      v18 = (_QWORD *)(*(_QWORD *)(v9 + 40) & 0xFFFFFFFFFFFFFFF8LL);
      if ( (_QWORD *)(v9 + 40) != v18 )
      {
        while ( 1 )
        {
          v20 = v18;
          v21 = *((_BYTE *)v18 - 8);
          v22 = *v18 & 0xFFFFFFFFFFFFFFF8LL;
          v18 = (_QWORD *)v22;
          if ( v21 == 78 )
            break;
          if ( v21 != 79 )
            goto LABEL_13;
          v23 = *(v20 - 12);
          v24 = *(_BYTE *)(v23 + 16);
          if ( v24 <= 0x17u )
            goto LABEL_13;
          if ( v24 == 75 )
          {
            v69 = *(unsigned __int16 *)(v23 + 18);
            BYTE1(v69) &= ~0x80u;
            v26 = v69;
            if ( (unsigned int)(v69 - 32) > 1 )
              goto LABEL_13;
            v70 = *(_QWORD *)(v23 - 24);
            if ( *(_BYTE *)(v70 + 16) != 13 )
              goto LABEL_13;
            v25 = *(_QWORD *)(v23 - 48);
            if ( *(_BYTE *)(v25 + 16) != 78 || *(_DWORD *)(v70 + 32) > 0x40u )
              goto LABEL_13;
            v10 = *(_QWORD *)(v70 + 24);
          }
          else
          {
            if ( v24 != 78 )
              goto LABEL_13;
            v25 = *(v20 - 12);
            v26 = 33;
            v23 = 0;
            v10 = 0;
          }
          v27 = *(_QWORD *)(v25 - 24);
          v135 = (_QWORD *)v10;
          v139 = v26;
          if ( *(_BYTE *)(v27 + 16) )
            goto LABEL_13;
          if ( *(_DWORD *)(v27 + 36) != 56 )
            goto LABEL_13;
          v28 = *(_DWORD *)(v25 + 20) & 0xFFFFFFF;
          v128 = *(_QWORD *)(v25 + 24 * (1 - v28));
          if ( *(_BYTE *)(v128 + 16) != 13 )
            goto LABEL_13;
          v126 = *(_QWORD *)(v25 - 24 * v28);
          v147 = (unsigned int *)sub_16498A0(v25);
          v29 = *(_QWORD **)(v128 + 24);
          if ( *(_DWORD *)(v128 + 32) > 0x40u )
            v29 = (_QWORD *)*v29;
          if ( (v135 == v29) == (v139 == 32) )
            v30 = sub_161BE60(&v147, dword_4FB3AA0, dword_4FB39C0);
          else
            v30 = sub_161BE60(&v147, dword_4FB39C0, dword_4FB3AA0);
          sub_1625C10((__int64)(v20 - 3), 2, v30);
          if ( !v23 )
          {
            if ( *(v20 - 12) )
            {
              v92 = *(v20 - 11);
              v93 = *(v20 - 10) & 0xFFFFFFFFFFFFFFFCLL;
              *(_QWORD *)v93 = v92;
              if ( v92 )
                *(_QWORD *)(v92 + 16) = *(_QWORD *)(v92 + 16) & 3LL | v93;
            }
            *(v20 - 12) = v126;
            if ( v126 )
            {
              v94 = *(_QWORD *)(v126 + 8);
              *(v20 - 11) = v94;
              if ( v94 )
                *(_QWORD *)(v94 + 16) = (unsigned __int64)(v20 - 11) | *(_QWORD *)(v94 + 16) & 3LL;
              LODWORD(v10) = (_DWORD)v20 - 96;
              *(v20 - 10) = (v126 + 8) | *(v20 - 10) & 3LL;
              *(_QWORD *)(v126 + 8) = v20 - 12;
            }
            goto LABEL_13;
          }
          if ( *(_QWORD *)(v23 - 48) )
          {
            v31 = *(_QWORD *)(v23 - 40);
            v32 = *(_QWORD *)(v23 - 32) & 0xFFFFFFFFFFFFFFFCLL;
            *(_QWORD *)v32 = v31;
            if ( v31 )
              *(_QWORD *)(v31 + 16) = *(_QWORD *)(v31 + 16) & 3LL | v32;
          }
          *(_QWORD *)(v23 - 48) = v126;
          if ( v126 )
          {
            v33 = *(_QWORD *)(v126 + 8);
            *(_QWORD *)(v23 - 40) = v33;
            if ( v33 )
              *(_QWORD *)(v33 + 16) = (v23 - 40) | *(_QWORD *)(v33 + 16) & 3LL;
            v34 = *(_QWORD *)(v23 - 32);
            v35 = v23 - 48;
            *(_QWORD *)(v35 + 16) = (v126 + 8) | v34 & 3;
            *(_QWORD *)(v126 + 8) = v35;
            if ( v17 == v22 )
              goto LABEL_35;
          }
          else
          {
LABEL_13:
            if ( v17 == v22 )
              goto LABEL_35;
          }
        }
        v19 = *(v20 - 6);
        if ( *(_BYTE *)(v19 + 16) || *(_DWORD *)(v19 + 36) != 56 )
          goto LABEL_13;
        v37 = *((_DWORD *)v20 - 1);
        v140 = v20 - 3;
        v38 = v37 & 0xFFFFFFF;
        v136 = v20[3 * (1 - v38) - 3];
        if ( *(_BYTE *)(v136 + 16) != 13 )
        {
LABEL_52:
          sub_164D160((__int64)v140, v20[-3 * (v37 & 0xFFFFFFF) - 3], a2, a3, a4, a5, v12, v13, a8, a9);
          sub_15F20C0(v140);
          v130 = 1;
          goto LABEL_13;
        }
        v39 = v20[-3 * v38 - 3];
        v147 = (unsigned int *)v149;
        v148 = 0x400000000LL;
        v40 = v39;
        v41 = *(_BYTE *)(v39 + 16);
        if ( v41 <= 0x17u )
        {
LABEL_51:
          v37 = *((_DWORD *)v20 - 1);
          goto LABEL_52;
        }
        v42 = v40;
        while ( v41 != 77 )
        {
          if ( v41 == 61 || v41 == 62 )
          {
            v43 = *(_QWORD *)(v42 - 24);
            v44 = (unsigned int)v148;
            if ( (unsigned int)v148 < HIDWORD(v148) )
              goto LABEL_48;
LABEL_46:
            v131 = v43;
            sub_16CD150((__int64)&v147, v149, 0, 8, v43, v10);
            v44 = (unsigned int)v148;
            v43 = v131;
            goto LABEL_48;
          }
          if ( v41 != 52 || *(_BYTE *)(*(_QWORD *)(v42 - 24) + 16LL) != 13 )
            goto LABEL_49;
          v43 = *(_QWORD *)(v42 - 48);
          v44 = (unsigned int)v148;
          if ( (unsigned int)v148 >= HIDWORD(v148) )
            goto LABEL_46;
LABEL_48:
          *(_QWORD *)&v147[2 * v44] = v42;
          v42 = v43;
          LODWORD(v148) = v148 + 1;
          v41 = *(_BYTE *)(v43 + 16);
          if ( v41 <= 0x17u )
            goto LABEL_49;
        }
        v71 = v42;
        v72 = *(_DWORD *)(v42 + 20) & 0xFFFFFFF;
        if ( !v72 )
        {
LABEL_49:
          if ( v147 != (unsigned int *)v149 )
            _libc_free((unsigned __int64)v147);
          goto LABEL_51;
        }
        v125 = v17;
        v73 = 0;
        v123 = v22;
        v132 = 8LL * v72;
        v124 = v20;
        v74 = v71;
        while ( 2 )
        {
          if ( (*(_BYTE *)(v74 + 23) & 0x40) != 0 )
            v75 = *(_QWORD *)(v74 - 8);
          else
            v75 = v74 - 24LL * (*(_DWORD *)(v74 + 20) & 0xFFFFFFF);
          v76 = *(_QWORD *)(v75 + 3 * v73);
          if ( *(_BYTE *)(v76 + 16) != 13 )
            goto LABEL_134;
          v144 = *(_DWORD *)(v76 + 32);
          if ( v144 > 0x40 )
            sub_16A4FD0((__int64)&v143, (const void **)(v76 + 24));
          else
            v143 = *(_QWORD *)(v76 + 24);
          v77 = v147;
          v78 = &v147[2 * (unsigned int)v148];
          if ( v147 == v78 )
          {
LABEL_116:
            if ( *(_DWORD *)(v136 + 32) <= 0x40u )
              v82 = *(_QWORD *)(v136 + 24) == v143;
            else
              v82 = sub_16A5220(v136 + 24, (const void **)&v143);
            if ( v144 > 0x40 && v143 )
              j_j___libc_free_0_0(v143);
            if ( v82 )
              goto LABEL_134;
            v83 = (*(_BYTE *)(v74 + 23) & 0x40) != 0
                ? *(_QWORD *)(v74 - 8)
                : v74 - 24LL * (*(_DWORD *)(v74 + 20) & 0xFFFFFFF);
            v127 = *(_QWORD *)(v73 + v83 + 24LL * *(unsigned int *)(v74 + 56) + 8);
            v84 = sub_157EBA0(v127);
            v85 = v84;
            if ( *(_BYTE *)(v84 + 16) != 26 || (*(_DWORD *)(v84 + 20) & 0xFFFFFFF) != 3 )
            {
              v86 = sub_157F0B0(v127);
              if ( !v86 )
                goto LABEL_134;
              v87 = sub_157EBA0(v86);
              v85 = v87;
              if ( *(_BYTE *)(v87 + 16) != 26 || (*(_DWORD *)(v87 + 20) & 0xFFFFFFF) == 1 )
                goto LABEL_134;
            }
            v145 = sub_16498A0(v74);
            if ( (*(_BYTE *)(v74 + 23) & 0x40) != 0 )
              v88 = *(_QWORD *)(v74 - 8);
            else
              v88 = v74 - 24LL * (*(_DWORD *)(v74 + 20) & 0xFFFFFFF);
            v89 = v88 + 24LL * *(unsigned int *)(v74 + 56);
            v90 = *(_QWORD *)(v85 - 48);
            v91 = *(_QWORD *)(v73 + v89 + 8);
            if ( v91 != v90 )
            {
              if ( v91 != *(_QWORD *)(v85 + 40) )
              {
                if ( v91 != *(_QWORD *)(v85 - 24) )
                {
LABEL_134:
                  v73 += 8;
                  if ( v132 == v73 )
                    goto LABEL_135;
                  continue;
                }
LABEL_156:
                v97 = dword_4FB3AA0;
                v98 = dword_4FB39C0;
LABEL_157:
                v73 += 8;
                v99 = sub_161BE60(&v145, v98, v97);
                sub_1625C10(v85, 2, v99);
                if ( v132 == v73 )
                {
LABEL_135:
                  v17 = v125;
                  v20 = v124;
                  v22 = v123;
                  goto LABEL_49;
                }
                continue;
              }
              v95 = *(_QWORD *)(v74 + 40);
              if ( v90 != v95 )
              {
                v96 = *(_QWORD *)(v85 - 24);
                if ( v91 != v96 && v95 != v96 )
                  goto LABEL_134;
                goto LABEL_156;
              }
            }
            v97 = dword_4FB39C0;
            v98 = dword_4FB3AA0;
            goto LABEL_157;
          }
          break;
        }
        while ( 2 )
        {
          while ( 2 )
          {
            v80 = *((_QWORD *)v78 - 1);
            v81 = *(unsigned __int8 *)(v80 + 16);
            if ( v81 == 61 )
            {
              sub_16A5C50((__int64)&v145, (const void **)&v143, *(_DWORD *)(*(_QWORD *)v80 + 8LL) >> 8);
              if ( v144 > 0x40 )
                goto LABEL_113;
            }
            else
            {
              if ( v81 != 62 )
              {
                if ( (*(_BYTE *)(v80 + 23) & 0x40) != 0 )
                {
                  v79 = *(_QWORD *)(*(_QWORD *)(v80 - 8) + 24LL);
                  if ( v144 <= 0x40 )
                    goto LABEL_108;
LABEL_139:
                  sub_16A8F00((__int64 *)&v143, (__int64 *)(v79 + 24));
                }
                else
                {
                  v79 = *(_QWORD *)(v80 - 24LL * (*(_DWORD *)(v80 + 20) & 0xFFFFFFF) + 24);
                  if ( v144 > 0x40 )
                    goto LABEL_139;
LABEL_108:
                  v143 ^= *(_QWORD *)(v79 + 24);
                }
                v78 -= 2;
                if ( v77 == v78 )
                  goto LABEL_116;
                continue;
              }
              sub_16A5B10((__int64)&v145, &v143, *(_DWORD *)(*(_QWORD *)v80 + 8LL) >> 8);
              if ( v144 > 0x40 )
              {
LABEL_113:
                if ( v143 )
                  j_j___libc_free_0_0(v143);
              }
            }
            break;
          }
          v78 -= 2;
          v143 = v145;
          v144 = v146;
          if ( v77 == v78 )
            goto LABEL_116;
          continue;
        }
      }
LABEL_35:
      v142 = *(_QWORD *)(v142 + 8);
      if ( v133 == v142 )
        return v130;
    }
    if ( v14 != 27 )
      goto LABEL_9;
    v45 = (*(_BYTE *)(v11 + 23) & 0x40) != 0
        ? *(__int64 **)(v11 - 8)
        : (__int64 *)(v11 - 24LL * (*(_DWORD *)(v11 + 20) & 0xFFFFFFF));
    v46 = *v45;
    if ( *(_BYTE *)(*v45 + 16) != 78 )
      goto LABEL_9;
    v47 = *(_QWORD *)(v46 - 24);
    if ( *(_BYTE *)(v47 + 16) )
      goto LABEL_9;
    if ( *(_DWORD *)(v47 + 36) != 56 )
      goto LABEL_9;
    v48 = *(_DWORD *)(v46 + 20) & 0xFFFFFFF;
    v49 = *(_QWORD *)(v46 + 24 * (1 - v48));
    if ( *(_BYTE *)(v49 + 16) != 13 )
      goto LABEL_9;
    v50 = *(_QWORD *)(v46 - 24 * v48);
    v51 = (*(_DWORD *)(v11 + 20) & 0xFFFFFFFu) >> 1;
    v52 = v51 - 1;
    v53 = v52 >> 2;
    if ( v52 >> 2 )
    {
      v10 = 4 * v53;
      v54 = 2;
      v55 = 0;
      while ( 1 )
      {
        v53 = v55 + 1;
        v59 = v45[3 * v54];
        if ( v59 )
        {
          if ( v49 == v59 )
            goto LABEL_71;
        }
        v56 = v45[3 * v54 + 6];
        if ( v56 && v49 == v56 )
          goto LABEL_72;
        v53 = v55 + 3;
        v57 = v45[3 * v54 + 12];
        if ( v57 && v49 == v57 )
        {
          v53 = v55 + 2;
          goto LABEL_72;
        }
        v55 += 4;
        v58 = v45[3 * (unsigned int)(2 * v55)];
        if ( v58 && v49 == v58 )
          goto LABEL_72;
        v54 += 8;
        if ( v55 == v10 )
        {
          v53 = v55;
          v119 = v52 - v55;
          goto LABEL_191;
        }
      }
    }
    v119 = v51 - 1;
LABEL_191:
    switch ( v119 )
    {
      case 2LL:
        v55 = v53;
        break;
      case 3LL:
        v55 = v53 + 1;
        v121 = v45[3 * (unsigned int)(2 * (v53 + 1))];
        if ( v121 && v49 == v121 )
        {
LABEL_72:
          if ( v52 != v53 )
          {
LABEL_73:
            v60 = dword_4FB39C0;
            v61 = v51;
            v147 = (unsigned int *)v149;
            v62 = (unsigned int *)v149;
            v148 = 0x1000000000LL;
            if ( v51 > 0x10 )
            {
              v129 = v51;
              v134 = dword_4FB39C0;
              v138 = v51;
              sub_16CD150((__int64)&v147, v149, v51, 4, (int)v149, v10);
              v62 = v147;
              v51 = v129;
              v60 = v134;
              v61 = v138;
            }
            v63 = &v62[v61];
            LODWORD(v148) = v51;
            if ( v63 != v62 )
            {
              do
                *v62++ = v60;
              while ( v63 != v62 );
              v62 = v147;
            }
            if ( v53 == 4294967294LL )
              *v62 = dword_4FB3AA0;
            else
              v62[(unsigned int)(v53 + 1)] = dword_4FB3AA0;
            v145 = sub_16498A0(v46);
            v64 = sub_161BD30(&v145, v147, (unsigned int)v148);
            sub_1625C10(v11, 2, v64);
            if ( (*(_BYTE *)(v11 + 23) & 0x40) != 0 )
              v65 = *(_QWORD **)(v11 - 8);
            else
              v65 = (_QWORD *)(v11 - 24LL * (*(_DWORD *)(v11 + 20) & 0xFFFFFFF));
            if ( *v65 )
            {
              v66 = v65[1];
              v67 = v65[2] & 0xFFFFFFFFFFFFFFFCLL;
              *(_QWORD *)v67 = v66;
              if ( v66 )
                *(_QWORD *)(v66 + 16) = v67 | *(_QWORD *)(v66 + 16) & 3LL;
            }
            *v65 = v50;
            if ( v50 )
            {
              v68 = *(_QWORD *)(v50 + 8);
              v65[1] = v68;
              if ( v68 )
                *(_QWORD *)(v68 + 16) = (unsigned __int64)(v65 + 1) | *(_QWORD *)(v68 + 16) & 3LL;
              v65[2] = (v50 + 8) | v65[2] & 3LL;
              *(_QWORD *)(v50 + 8) = v65;
            }
            if ( v147 != (unsigned int *)v149 )
              _libc_free((unsigned __int64)v147);
            goto LABEL_9;
          }
LABEL_196:
          v53 = 4294967294LL;
          goto LABEL_73;
        }
        break;
      case 1LL:
        goto LABEL_194;
      default:
        goto LABEL_196;
    }
    v53 = v55 + 1;
    v122 = v45[3 * (unsigned int)(2 * (v55 + 1))];
    if ( v122 && v49 == v122 )
    {
LABEL_71:
      v53 = v55;
      goto LABEL_72;
    }
LABEL_194:
    v120 = v45[3 * (unsigned int)(2 * v53 + 2)];
    if ( !v120 || v49 != v120 )
      goto LABEL_196;
    goto LABEL_72;
  }
  return v130;
}
