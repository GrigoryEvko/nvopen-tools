// Function: sub_1721670
// Address: 0x1721670
//
unsigned __int8 *__fastcall sub_1721670(
        __int64 a1,
        __int64 a2,
        double a3,
        double a4,
        double a5,
        __int64 a6,
        __int64 a7)
{
  __int64 v9; // r14
  __int64 v10; // r12
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rsi
  __int64 v14; // rcx
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r14
  __int64 v19; // r12
  char v20; // al
  char v21; // al
  __int64 v22; // r12
  __int64 v24; // rax
  const void **v25; // rdi
  unsigned __int8 v26; // al
  __int64 v27; // rax
  _BYTE *v28; // rdi
  unsigned __int8 v29; // al
  __int64 v30; // rdi
  unsigned int v31; // edx
  unsigned int v33; // eax
  char v34; // dl
  __int64 v35; // rdx
  __int64 v36; // rsi
  __int64 v37; // rcx
  __int64 v38; // rdx
  const void **v39; // rdi
  __int64 v40; // rax
  _BYTE *v41; // rsi
  __int64 v42; // rax
  char v43; // dl
  __int64 v44; // rdx
  __int64 v45; // rdx
  const void **v46; // rdi
  __int64 v47; // rdx
  __int64 v48; // rdx
  __int64 v49; // rax
  __int64 v50; // rax
  __int64 v51; // rax
  __int64 v52; // rax
  unsigned __int8 v53; // al
  unsigned int v54; // edx
  _QWORD *v55; // r8
  bool v56; // al
  bool v57; // bl
  unsigned int v58; // eax
  __int64 v59; // rsi
  unsigned __int64 v60; // rdx
  __int64 v61; // rbx
  __int64 v62; // rax
  __int64 v63; // rcx
  __int64 v64; // rdx
  __int64 v65; // rax
  __int64 v66; // rax
  unsigned __int8 v67; // al
  char v68; // al
  const void **v69; // r12
  unsigned int v70; // eax
  __int64 v71; // rsi
  unsigned __int64 v72; // rdx
  __int64 v73; // rbx
  __int64 v74; // rax
  __int64 v75; // rcx
  __int64 v76; // rdx
  __int64 v77; // rax
  __int64 v78; // rax
  __int64 v79; // rdi
  __int64 v80; // rsi
  __int64 v81; // rax
  __int64 v82; // rdx
  bool v83; // zf
  __int64 v84; // rsi
  __int64 v85; // rsi
  __int64 v86; // rdx
  unsigned __int8 *v87; // rsi
  __int64 v88; // rax
  __int64 v89; // rdx
  __int64 v90; // rax
  __int64 v91; // rax
  __int64 v92; // rax
  unsigned int v93; // edi
  unsigned __int64 v94; // rsi
  _QWORD *v95; // rax
  __int64 v96; // rdx
  unsigned __int8 v97; // al
  __int64 v98; // rax
  _QWORD *v99; // rax
  unsigned int v100; // esi
  int v101; // eax
  _QWORD *v102; // rax
  _QWORD *v103; // [rsp+0h] [rbp-D0h]
  __int64 v104; // [rsp+0h] [rbp-D0h]
  int v105; // [rsp+0h] [rbp-D0h]
  unsigned int v106; // [rsp+8h] [rbp-C8h]
  __int64 v107; // [rsp+8h] [rbp-C8h]
  __int64 *v108; // [rsp+8h] [rbp-C8h]
  __int64 v109; // [rsp+8h] [rbp-C8h]
  __int64 v110; // [rsp+10h] [rbp-C0h] BYREF
  __int64 v111; // [rsp+18h] [rbp-B8h] BYREF
  __int64 *v112; // [rsp+20h] [rbp-B0h] BYREF
  const void **v113; // [rsp+28h] [rbp-A8h] BYREF
  const void **v114; // [rsp+30h] [rbp-A0h] BYREF
  unsigned __int8 *v115; // [rsp+38h] [rbp-98h] BYREF
  unsigned __int64 v116; // [rsp+40h] [rbp-90h] BYREF
  unsigned int v117; // [rsp+48h] [rbp-88h]
  __int64 **v118; // [rsp+50h] [rbp-80h] BYREF
  const void ***v119; // [rsp+58h] [rbp-78h] BYREF
  _QWORD *v120; // [rsp+60h] [rbp-70h] BYREF
  unsigned int v121; // [rsp+68h] [rbp-68h]
  __int16 v122; // [rsp+70h] [rbp-60h]
  __int64 **v123; // [rsp+80h] [rbp-50h] BYREF
  const void ***v124; // [rsp+88h] [rbp-48h]
  __int16 v125; // [rsp+90h] [rbp-40h]

  v9 = *(_QWORD *)(a1 - 48);
  v10 = *(_QWORD *)(a1 - 24);
  v11 = *(_QWORD *)(v9 + 8);
  if ( !v11 || *(_QWORD *)(v11 + 8) )
  {
    v12 = *(_QWORD *)(v10 + 8);
    if ( !v12 || *(_QWORD *)(v12 + 8) )
      return 0;
  }
  v13 = *(_QWORD *)(a1 - 24);
  v110 = 0;
  v123 = (__int64 **)&v110;
  v111 = 0;
  v112 = 0;
  v113 = 0;
  v114 = 0;
  if ( sub_171CD50(&v123, v13, (__int64)&v110, a7) )
  {
    v24 = v9;
    v9 = v10;
    v10 = v24;
  }
  _RSI = (const void **)v9;
  v123 = (__int64 **)&v110;
  if ( !sub_171CD50(&v123, v9, (__int64)&v110, v14) )
  {
LABEL_8:
    v18 = *(_QWORD *)(a1 - 24);
    v19 = *(_QWORD *)(a1 - 48);
    v20 = *(_BYTE *)(v18 + 16);
    if ( v20 == 52 )
    {
      if ( !*(_QWORD *)(v18 - 48) )
        goto LABEL_11;
      v111 = *(_QWORD *)(v18 - 48);
      v25 = *(const void ***)(v18 - 24);
      v26 = *((_BYTE *)v25 + 16);
      if ( v26 == 13 )
      {
        v27 = v19;
        v19 = v18;
        v113 = v25 + 3;
        v18 = v27;
        goto LABEL_20;
      }
      v47 = (__int64)*v25;
      if ( *((_BYTE *)*v25 + 8) != 16 || v26 > 0x10u )
        goto LABEL_11;
    }
    else
    {
      if ( v20 != 5 )
        goto LABEL_11;
      if ( *(_WORD *)(v18 + 18) != 28 )
        goto LABEL_11;
      v49 = *(_DWORD *)(v18 + 20) & 0xFFFFFFF;
      v17 = 4 * v49;
      if ( !*(_QWORD *)(v18 - 24 * v49) )
        goto LABEL_11;
      v111 = *(_QWORD *)(v18 - 24LL * (*(_DWORD *)(v18 + 20) & 0xFFFFFFF));
      v47 = 1 - v49;
      v25 = *(const void ***)(v18 + 24 * (1 - v49));
      if ( *((_BYTE *)v25 + 16) == 13 )
      {
        v50 = v18;
        v18 = v19;
        v113 = v25 + 3;
        v19 = v50;
        goto LABEL_14;
      }
      if ( *((_BYTE *)*v25 + 8) != 16 )
        goto LABEL_11;
    }
    v88 = sub_15A1020(v25, (__int64)_RSI, v47, v17);
    if ( v88 && *(_BYTE *)(v88 + 16) == 13 )
    {
      v89 = v19;
      v19 = v18;
      v113 = (const void **)(v88 + 24);
      v21 = *(_BYTE *)(v18 + 16);
      v18 = v89;
LABEL_12:
      if ( v21 != 52 )
      {
        if ( v21 != 5 )
          return 0;
LABEL_14:
        if ( *(_WORD *)(v19 + 18) != 28 )
          return 0;
        v51 = *(_DWORD *)(v19 + 20) & 0xFFFFFFF;
        v17 = 4 * v51;
        if ( !*(_QWORD *)(v19 - 24 * v51) )
          return 0;
        v111 = *(_QWORD *)(v19 - 24LL * (*(_DWORD *)(v19 + 20) & 0xFFFFFFF));
        v48 = 1 - v51;
        v28 = *(_BYTE **)(v19 + 24 * (1 - v51));
        if ( v28[16] != 13 )
        {
          if ( *(_BYTE *)(*(_QWORD *)v28 + 8LL) != 16 )
            return 0;
          goto LABEL_60;
        }
        goto LABEL_22;
      }
LABEL_20:
      if ( !*(_QWORD *)(v19 - 48) )
        return 0;
      v111 = *(_QWORD *)(v19 - 48);
      v28 = *(_BYTE **)(v19 - 24);
      v29 = v28[16];
      if ( v29 != 13 )
      {
        v48 = *(_QWORD *)v28;
        if ( *(_BYTE *)(*(_QWORD *)v28 + 8LL) != 16 || v29 > 0x10u )
          return 0;
LABEL_60:
        v52 = sub_15A1020(v28, (__int64)_RSI, v48, v17);
        if ( !v52 || *(_BYTE *)(v52 + 16) != 13 )
          return 0;
        v30 = v52 + 24;
        v113 = (const void **)(v52 + 24);
LABEL_23:
        v31 = *(_DWORD *)(v30 + 8);
        if ( v31 > 0x40 )
        {
          v33 = sub_16A58A0(v30);
        }
        else
        {
          _RCX = *(_QWORD *)v30;
          v33 = 64;
          __asm { tzcnt   rsi, rcx }
          if ( *(_QWORD *)v30 )
            v33 = (unsigned int)_RSI;
          if ( v33 > v31 )
            v33 = *(_DWORD *)(v30 + 8);
        }
        if ( v33 )
          return 0;
        v34 = *(_BYTE *)(v111 + 16);
        if ( v34 == 50 )
        {
          if ( !*(_QWORD *)(v111 - 48) )
            return 0;
          v112 = *(__int64 **)(v111 - 48);
          v39 = *(const void ***)(v111 - 24);
          v53 = *((_BYTE *)v39 + 16);
          if ( v53 != 13 )
          {
            if ( *((_BYTE *)*v39 + 8) != 16 )
              return 0;
            if ( v53 > 0x10u )
              return 0;
            v91 = sub_15A1020(v39, (__int64)_RSI, (__int64)*v39, _RCX);
            if ( !v91 || *(_BYTE *)(v91 + 16) != 13 )
              return 0;
            v41 = (_BYTE *)(v91 + 24);
            v114 = (const void **)(v91 + 24);
            goto LABEL_67;
          }
        }
        else
        {
          if ( v34 != 5 )
            return 0;
          if ( *(_WORD *)(v111 + 18) != 26 )
            return 0;
          v35 = *(_DWORD *)(v111 + 20) & 0xFFFFFFF;
          v36 = 4 * v35;
          if ( !*(_QWORD *)(v111 - 24 * v35) )
            return 0;
          v112 = *(__int64 **)(v111 - 24LL * (*(_DWORD *)(v111 + 20) & 0xFFFFFFF));
          v37 = 1 - v35;
          v38 = 3 * (1 - v35);
          v39 = *(const void ***)(v111 + 8 * v38);
          if ( *((_BYTE *)v39 + 16) != 13 )
          {
            if ( *((_BYTE *)*v39 + 8) != 16 )
              return 0;
            v40 = sub_15A1020(v39, v36, v38, v37);
            if ( !v40 || *(_BYTE *)(v40 + 16) != 13 )
              return 0;
            v41 = (_BYTE *)(v40 + 24);
            v114 = (const void **)(v40 + 24);
            goto LABEL_67;
          }
        }
        v41 = v39 + 3;
        v114 = v39 + 3;
LABEL_67:
        v121 = *((_DWORD *)v41 + 2);
        if ( v121 > 0x40 )
          sub_16A4FD0((__int64)&v120, (const void **)v41);
        else
          v120 = *(_QWORD **)v41;
        sub_16A7490((__int64)&v120, 1);
        v54 = v121;
        v55 = v120;
        v121 = 0;
        LODWORD(v124) = v54;
        v123 = (__int64 **)v120;
        if ( *((_DWORD *)v113 + 2) <= 0x40u )
        {
          v57 = *v113 == v120;
        }
        else
        {
          v103 = v120;
          v106 = v54;
          v56 = sub_16A5220((__int64)v113, (const void **)&v123);
          v55 = v103;
          v54 = v106;
          v57 = v56;
        }
        if ( v54 > 0x40 )
        {
          if ( v55 )
          {
            j_j___libc_free_0_0(v55);
            if ( v121 > 0x40 )
            {
              if ( v120 )
                j_j___libc_free_0_0(v120);
            }
          }
        }
        if ( v57 )
        {
          v122 = 257;
          v58 = *((_DWORD *)v114 + 2);
          v117 = v58;
          if ( v58 > 0x40 )
          {
            sub_16A4FD0((__int64)&v116, v114);
            v58 = v117;
            if ( v117 > 0x40 )
            {
              sub_16A8F40((__int64 *)&v116);
              v58 = v117;
              v60 = v116;
LABEL_76:
              v61 = (__int64)v112;
              v118 = (__int64 **)v60;
              LODWORD(v119) = v58;
              v117 = 0;
              v62 = sub_15A1070(*v112, (__int64)&v118);
              v64 = v62;
              if ( *(_BYTE *)(v62 + 16) <= 0x10u )
              {
                v107 = v62;
                if ( sub_1593BB0(v62, (__int64)&v118, v62, v63) )
                {
LABEL_81:
                  if ( (unsigned int)v119 > 0x40 && v118 )
                    j_j___libc_free_0_0(v118);
                  if ( v117 > 0x40 && v116 )
                    j_j___libc_free_0_0(v116);
                  goto LABEL_87;
                }
                v64 = v107;
                if ( *(_BYTE *)(v61 + 16) <= 0x10u )
                {
                  v61 = sub_15A2D10((__int64 *)v61, v107, a3, a4, a5);
                  v65 = sub_14DBA30(v61, *(_QWORD *)(a2 + 96), 0);
                  if ( v65 )
                    v61 = v65;
                  goto LABEL_81;
                }
              }
              v125 = 257;
              v78 = sub_15FB440(27, (__int64 *)v61, v64, (__int64)&v123, 0);
              v79 = *(_QWORD *)(a2 + 8);
              v61 = v78;
              if ( v79 )
              {
                v108 = *(__int64 **)(a2 + 16);
                sub_157E9D0(v79 + 40, v78);
                v80 = *v108;
                v81 = *(_QWORD *)(v61 + 24) & 7LL;
                *(_QWORD *)(v61 + 32) = v108;
                v80 &= 0xFFFFFFFFFFFFFFF8LL;
                *(_QWORD *)(v61 + 24) = v80 | v81;
                *(_QWORD *)(v80 + 8) = v61 + 24;
                *v108 = *v108 & 7 | (v61 + 24);
              }
              sub_164B780(v61, (__int64 *)&v120);
              v83 = *(_QWORD *)(a2 + 80) == 0;
              v115 = (unsigned __int8 *)v61;
              if ( v83 )
                sub_4263D6(v61, &v120, v82);
              (*(void (__fastcall **)(__int64, unsigned __int8 **))(a2 + 88))(a2 + 64, &v115);
              v84 = *(_QWORD *)a2;
              if ( *(_QWORD *)a2 )
              {
                v115 = *(unsigned __int8 **)a2;
                sub_1623A60((__int64)&v115, v84, 2);
                v85 = *(_QWORD *)(v61 + 48);
                v86 = v61 + 48;
                if ( v85 )
                {
                  sub_161E7C0(v61 + 48, v85);
                  v86 = v61 + 48;
                }
                v87 = v115;
                *(_QWORD *)(v61 + 48) = v115;
                if ( v87 )
                  sub_1623210((__int64)&v115, v87, v86);
              }
              goto LABEL_81;
            }
            v59 = v116;
          }
          else
          {
            v59 = (__int64)*v114;
          }
          v60 = ~v59 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v58);
          v116 = v60;
          goto LABEL_76;
        }
        return 0;
      }
LABEL_22:
      v30 = (__int64)(v28 + 24);
      v113 = (const void **)v30;
      goto LABEL_23;
    }
LABEL_11:
    v21 = *(_BYTE *)(v19 + 16);
    goto LABEL_12;
  }
  _RSI = (const void **)v10;
  v123 = (__int64 **)&v111;
  v124 = &v113;
  if ( (unsigned __int8)sub_171CFA0(&v123, v10, v16, v17) )
  {
    v18 = v110;
    v42 = v10;
    v110 = v10;
  }
  else
  {
    v42 = v110;
    v18 = v10;
  }
  v43 = *(_BYTE *)(v42 + 16);
  if ( v43 == 52 )
  {
    _RSI = *(const void ***)(v42 - 48);
    if ( !_RSI )
      goto LABEL_8;
    v111 = *(_QWORD *)(v42 - 48);
    v46 = *(const void ***)(v42 - 24);
    v67 = *((_BYTE *)v46 + 16);
    if ( v67 != 13 )
    {
      v45 = (__int64)*v46;
      if ( *((_BYTE *)*v46 + 8) != 16 || v67 > 0x10u )
        goto LABEL_8;
      goto LABEL_131;
    }
LABEL_95:
    v113 = v46 + 3;
    goto LABEL_96;
  }
  if ( v43 != 5 )
    goto LABEL_8;
  if ( *(_WORD *)(v42 + 18) != 28 )
    goto LABEL_8;
  v44 = *(_DWORD *)(v42 + 20) & 0xFFFFFFF;
  v17 = -3 * v44;
  _RSI = *(const void ***)(v42 - 24 * v44);
  if ( !_RSI )
    goto LABEL_8;
  v111 = *(_QWORD *)(v42 - 24LL * (*(_DWORD *)(v42 + 20) & 0xFFFFFFF));
  v17 = 1 - v44;
  v45 = 3 * (1 - v44);
  v46 = *(const void ***)(v42 + 8 * v45);
  if ( *((_BYTE *)v46 + 16) == 13 )
    goto LABEL_95;
  if ( *((_BYTE *)*v46 + 8) != 16 )
    goto LABEL_8;
LABEL_131:
  v90 = sub_15A1020(v46, (__int64)_RSI, v45, v17);
  if ( !v90 || *(_BYTE *)(v90 + 16) != 13 )
    goto LABEL_8;
  _RSI = (const void **)v111;
  v113 = (const void **)(v90 + 24);
LABEL_96:
  v118 = &v112;
  v119 = &v114;
  v68 = *((_BYTE *)_RSI + 16);
  if ( v68 == 51 )
  {
    if ( !*(_RSI - 6) )
      goto LABEL_99;
    v112 = (__int64 *)*(_RSI - 6);
    if ( (unsigned __int8)sub_13D2630(&v119, *(_RSI - 3)) )
      goto LABEL_146;
LABEL_143:
    _RSI = (const void **)v111;
    goto LABEL_99;
  }
  if ( v68 != 5 || *((_WORD *)_RSI + 9) != 27 || (v92 = *((_DWORD *)_RSI + 5) & 0xFFFFFFF, !_RSI[-3 * v92]) )
  {
LABEL_99:
    v123 = &v112;
    v124 = &v114;
    if ( !(unsigned __int8)sub_13D5F90(&v123, (__int64)_RSI) )
      goto LABEL_8;
    v69 = v113;
    _RSI = v114;
    if ( *((_DWORD *)v113 + 2) <= 0x40u )
    {
      if ( *v113 != *v114 )
        goto LABEL_8;
    }
    else if ( !sub_16A5220((__int64)v113, v114) )
    {
      goto LABEL_8;
    }
    v122 = 257;
    v70 = *((_DWORD *)v69 + 2);
    v117 = v70;
    if ( v70 > 0x40 )
    {
      sub_16A4FD0((__int64)&v116, v69);
      v70 = v117;
      if ( v117 > 0x40 )
      {
        sub_16A8F40((__int64 *)&v116);
        v70 = v117;
        v72 = v116;
LABEL_105:
        v73 = (__int64)v112;
        v118 = (__int64 **)v72;
        LODWORD(v119) = v70;
        v117 = 0;
        v74 = sub_15A1070(*v112, (__int64)&v118);
        v76 = v74;
        if ( *(_BYTE *)(v74 + 16) <= 0x10u )
        {
          v104 = v74;
          if ( sub_1593BB0(v74, (__int64)&v118, v74, v75) )
          {
LABEL_110:
            sub_135E100((__int64 *)&v118);
            sub_135E100((__int64 *)&v116);
            v123 = (__int64 **)"sub";
            v125 = 259;
            return sub_171D0D0(a2, v18, v73, (__int64 *)&v123, 0, 0, a3, a4, a5);
          }
          v76 = v104;
          if ( *(_BYTE *)(v73 + 16) <= 0x10u )
          {
            v73 = sub_15A2D10((__int64 *)v73, v104, a3, a4, a5);
            v77 = sub_14DBA30(v73, *(_QWORD *)(a2 + 96), 0);
            if ( v77 )
              v73 = v77;
            goto LABEL_110;
          }
        }
        v125 = 257;
        v102 = (_QWORD *)sub_15FB440(27, (__int64 *)v73, v76, (__int64)&v123, 0);
        v73 = (__int64)sub_171D920(a2, v102, (__int64 *)&v120);
        goto LABEL_110;
      }
      v71 = v116;
    }
    else
    {
      v71 = (__int64)*v69;
    }
    v72 = ~v71 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v70);
    v116 = v72;
    goto LABEL_105;
  }
  v112 = (__int64 *)_RSI[-3 * (*((_DWORD *)_RSI + 5) & 0xFFFFFFF)];
  if ( !(unsigned __int8)sub_13D7780(&v119, _RSI[3 * (1 - v92)]) )
    goto LABEL_143;
LABEL_146:
  v93 = *((_DWORD *)v113 + 2);
  v121 = v93;
  if ( v93 <= 0x40 )
  {
    v94 = (unsigned __int64)*v113;
LABEL_148:
    v95 = (_QWORD *)(~v94 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v93));
    v120 = v95;
    goto LABEL_149;
  }
  sub_16A4FD0((__int64)&v120, v113);
  v93 = v121;
  if ( v121 <= 0x40 )
  {
    v94 = (unsigned __int64)v120;
    goto LABEL_148;
  }
  sub_16A8F40((__int64 *)&v120);
  v93 = v121;
  v95 = v120;
LABEL_149:
  LODWORD(v124) = v93;
  v123 = (__int64 **)v95;
  v121 = 0;
  if ( *((_DWORD *)v114 + 2) <= 0x40u )
  {
    if ( *v114 != v95 )
      goto LABEL_151;
  }
  else if ( !sub_16A5220((__int64)v114, (const void **)&v123) )
  {
LABEL_151:
    sub_135E100((__int64 *)&v123);
    sub_135E100((__int64 *)&v120);
    _RSI = (const void **)v111;
    goto LABEL_99;
  }
  sub_135E100((__int64 *)&v123);
  sub_135E100((__int64 *)&v120);
  v61 = (__int64)v112;
  v122 = 257;
  v96 = sub_15A1070(*v112, (__int64)v113);
  v97 = *(_BYTE *)(v96 + 16);
  if ( v97 <= 0x10u )
  {
    if ( v97 == 13 )
    {
      v100 = *(_DWORD *)(v96 + 32);
      if ( v100 <= 0x40 )
      {
        if ( *(_QWORD *)(v96 + 24) == 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v100) )
          goto LABEL_87;
      }
      else
      {
        v105 = *(_DWORD *)(v96 + 32);
        v109 = v96;
        v101 = sub_16A58F0(v96 + 24);
        v96 = v109;
        if ( v105 == v101 )
          goto LABEL_87;
      }
    }
    if ( *(_BYTE *)(v61 + 16) <= 0x10u )
    {
      v61 = sub_15A2CF0((__int64 *)v61, v96, a3, a4, a5);
      v98 = sub_14DBA30(v61, *(_QWORD *)(a2 + 96), 0);
      if ( v98 )
        v61 = v98;
      goto LABEL_87;
    }
  }
  v125 = 257;
  v99 = (_QWORD *)sub_15FB440(26, (__int64 *)v61, v96, (__int64)&v123, 0);
  v61 = (__int64)sub_171D920(a2, v99, (__int64 *)&v120);
LABEL_87:
  v123 = (__int64 **)"sub";
  v125 = 259;
  if ( *(_BYTE *)(v18 + 16) > 0x10u || *(_BYTE *)(v61 + 16) > 0x10u )
    return sub_170A2B0(a2, 13, (__int64 *)v18, v61, (__int64 *)&v123, 0, 0);
  v22 = sub_15A2B60((__int64 *)v18, v61, 0, 0, a3, a4, a5);
  v66 = sub_14DBA30(v22, *(_QWORD *)(a2 + 96), 0);
  if ( v66 )
    return (unsigned __int8 *)v66;
  return (unsigned __int8 *)v22;
}
