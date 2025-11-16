// Function: sub_1792E60
// Address: 0x1792e60
//
_QWORD *__fastcall sub_1792E60(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        _BYTE *a4,
        __int64 a5,
        double a6,
        double a7,
        double a8)
{
  __int64 v8; // rax
  __int64 v9; // r15
  __int64 v10; // rax
  _QWORD *v11; // rbx
  int v12; // eax
  __int64 v14; // rcx
  bool v17; // al
  __int64 v18; // rcx
  __int64 v19; // rdx
  unsigned __int8 v20; // al
  int v21; // eax
  bool v22; // al
  __int64 v23; // rax
  char v24; // al
  __int64 v25; // rax
  __int64 v26; // r14
  __int64 v27; // rdx
  bool v28; // al
  __int64 v29; // rax
  char v30; // cl
  __int64 v31; // rax
  char v32; // dl
  __int64 v33; // rdi
  __int64 v34; // rbx
  __int64 v35; // rax
  __int64 v36; // r9
  unsigned __int8 *v37; // rax
  unsigned __int8 *v38; // rax
  __int64 v39; // r14
  __int64 v40; // rdx
  __int64 v41; // rcx
  __int64 v42; // rax
  __int64 v43; // r14
  __int64 v44; // rax
  _QWORD *v45; // rax
  int v47; // eax
  bool v48; // al
  _BYTE *v49; // rdi
  __int64 v50; // rax
  __int64 v51; // rcx
  int v52; // eax
  char v53; // al
  __int64 v54; // rax
  __int64 v55; // rax
  unsigned int v56; // r14d
  int v57; // eax
  __int64 v58; // rax
  unsigned __int8 v59; // al
  __int64 v60; // rax
  __int64 v61; // rdx
  __int64 v62; // r9
  __int64 v63; // rax
  __int64 v64; // rax
  __int64 v65; // rsi
  __int64 v66; // rdi
  __int64 v67; // rdx
  bool v68; // zf
  __int64 v69; // rsi
  __int64 v70; // rsi
  __int64 v71; // rdx
  unsigned __int8 *v72; // rsi
  unsigned int v73; // ecx
  __int64 v74; // rax
  unsigned int v75; // ecx
  int v76; // eax
  bool v77; // al
  __int64 v78; // rax
  char v79; // di
  int v80; // eax
  bool v81; // al
  __int64 v82; // rax
  __int64 v83; // rdx
  unsigned int v84; // ecx
  __int64 v85; // rax
  char v86; // si
  unsigned int v87; // ecx
  int v88; // eax
  bool v89; // al
  unsigned int v90; // ecx
  __int64 v91; // rax
  char v92; // si
  unsigned int v93; // ecx
  int v94; // eax
  __int64 v95; // [rsp+0h] [rbp-B0h]
  int v96; // [rsp+8h] [rbp-A8h]
  int v97; // [rsp+8h] [rbp-A8h]
  int v98; // [rsp+8h] [rbp-A8h]
  int v99; // [rsp+10h] [rbp-A0h]
  __int64 v100; // [rsp+10h] [rbp-A0h]
  int v101; // [rsp+10h] [rbp-A0h]
  unsigned int v102; // [rsp+10h] [rbp-A0h]
  __int64 v103; // [rsp+10h] [rbp-A0h]
  unsigned int v104; // [rsp+10h] [rbp-A0h]
  __int64 v105; // [rsp+18h] [rbp-98h]
  __int64 v106; // [rsp+20h] [rbp-90h]
  __int64 v107; // [rsp+20h] [rbp-90h]
  __int64 v108; // [rsp+20h] [rbp-90h]
  __int64 *v109; // [rsp+20h] [rbp-90h]
  __int64 v110; // [rsp+20h] [rbp-90h]
  __int64 v111; // [rsp+20h] [rbp-90h]
  unsigned int v112; // [rsp+20h] [rbp-90h]
  __int64 v113; // [rsp+20h] [rbp-90h]
  int v115; // [rsp+28h] [rbp-88h]
  int v116; // [rsp+28h] [rbp-88h]
  char v117; // [rsp+28h] [rbp-88h]
  __int64 v118; // [rsp+28h] [rbp-88h]
  int v119; // [rsp+28h] [rbp-88h]
  __int64 v120; // [rsp+28h] [rbp-88h]
  __int64 v121; // [rsp+28h] [rbp-88h]
  __int64 v122; // [rsp+28h] [rbp-88h]
  int v123; // [rsp+28h] [rbp-88h]
  __int64 v124; // [rsp+28h] [rbp-88h]
  __int64 v125; // [rsp+28h] [rbp-88h]
  __int64 v126; // [rsp+28h] [rbp-88h]
  unsigned int v127; // [rsp+28h] [rbp-88h]
  __int64 v128; // [rsp+28h] [rbp-88h]
  __int64 v129; // [rsp+28h] [rbp-88h]
  int v130; // [rsp+28h] [rbp-88h]
  int v131; // [rsp+28h] [rbp-88h]
  unsigned __int8 *v132; // [rsp+38h] [rbp-78h] BYREF
  __int64 v133[2]; // [rsp+40h] [rbp-70h] BYREF
  __int16 v134; // [rsp+50h] [rbp-60h]
  __int64 v135[2]; // [rsp+60h] [rbp-50h] BYREF
  __int16 v136; // [rsp+70h] [rbp-40h]

  v8 = *(_QWORD *)(a2 + 8);
  if ( !v8 )
    return 0;
  if ( *(_QWORD *)(v8 + 8) )
    return 0;
  v9 = a2;
  v10 = *(_QWORD *)(*(_QWORD *)(a2 - 48) + 8LL);
  if ( !v10 )
    return 0;
  v11 = *(_QWORD **)(v10 + 8);
  if ( v11 )
    return 0;
  v12 = *(unsigned __int16 *)(a2 + 18);
  BYTE1(v12) &= ~0x80u;
  if ( v12 != 32 )
    return 0;
  v14 = *(_QWORD *)(a2 - 24);
  if ( *(_BYTE *)(v14 + 16) > 0x10u )
    return 0;
  v106 = *(_QWORD *)(a2 - 24);
  v17 = sub_1593BB0(v106, a2, a3, v14);
  v18 = v106;
  v19 = a3;
  if ( !v17 )
  {
    if ( *(_BYTE *)(v106 + 16) == 13 )
    {
      a2 = *(unsigned int *)(v106 + 32);
      if ( (unsigned int)a2 <= 0x40 )
      {
        v48 = *(_QWORD *)(v106 + 24) == 0;
      }
      else
      {
        v47 = sub_16A57B0(v106 + 24);
        a2 = (unsigned int)a2;
        v19 = a3;
        v48 = (_DWORD)a2 == v47;
      }
    }
    else
    {
      if ( *(_BYTE *)(*(_QWORD *)v106 + 8LL) != 16 )
        return 0;
      v49 = (_BYTE *)v106;
      v108 = a3;
      v118 = v18;
      v50 = sub_15A1020(v49, a2, v19, v18);
      v51 = v118;
      v19 = v108;
      if ( !v50 || *(_BYTE *)(v50 + 16) != 13 )
      {
        v96 = *(_QWORD *)(*(_QWORD *)v118 + 32LL);
        if ( v96 )
        {
          LODWORD(a2) = 0;
          while ( 1 )
          {
            v100 = v19;
            v128 = v51;
            v78 = sub_15A0A60(v51, a2);
            if ( !v78 )
              return 0;
            v79 = *(_BYTE *)(v78 + 16);
            v51 = v128;
            v19 = v100;
            if ( v79 != 9 )
            {
              if ( v79 != 13 )
                return 0;
              if ( *(_DWORD *)(v78 + 32) <= 0x40u )
              {
                v81 = *(_QWORD *)(v78 + 24) == 0;
              }
              else
              {
                v95 = v100;
                v101 = *(_DWORD *)(v78 + 32);
                v80 = sub_16A57B0(v78 + 24);
                v51 = v128;
                v19 = v95;
                v81 = v101 == v80;
              }
              if ( !v81 )
                return 0;
            }
            a2 = (unsigned int)(a2 + 1);
            if ( v96 == (_DWORD)a2 )
              goto LABEL_8;
          }
        }
        goto LABEL_8;
      }
      if ( *(_DWORD *)(v50 + 32) <= 0x40u )
      {
        v48 = *(_QWORD *)(v50 + 24) == 0;
      }
      else
      {
        v119 = *(_DWORD *)(v50 + 32);
        v52 = sub_16A57B0(v50 + 24);
        v19 = v108;
        v48 = v119 == v52;
      }
    }
    if ( !v48 )
      return 0;
  }
LABEL_8:
  v20 = a4[16];
  if ( v20 == 13 )
  {
    if ( *((_DWORD *)a4 + 8) <= 0x40u )
    {
      if ( *((_QWORD *)a4 + 3) != 1 )
        return 0;
      goto LABEL_12;
    }
    v107 = v19;
    v115 = *((_DWORD *)a4 + 8);
    v21 = sub_16A57B0((__int64)(a4 + 24));
    v19 = v107;
    v22 = v115 - 1 == v21;
  }
  else
  {
    if ( *(_BYTE *)(*(_QWORD *)a4 + 8LL) != 16 || v20 > 0x10u )
      return 0;
    v120 = v19;
    v55 = sub_15A1020(a4, a2, v19, *(_QWORD *)a4);
    v19 = v120;
    if ( !v55 || *(_BYTE *)(v55 + 16) != 13 )
    {
      v73 = 0;
      v99 = *(_QWORD *)(*(_QWORD *)a4 + 32LL);
      if ( v99 )
      {
        while ( 1 )
        {
          v110 = v19;
          v127 = v73;
          v74 = sub_15A0A60((__int64)a4, v73);
          if ( !v74 )
            return 0;
          a2 = *(unsigned __int8 *)(v74 + 16);
          v75 = v127;
          v19 = v110;
          if ( (_BYTE)a2 != 9 )
          {
            if ( (_BYTE)a2 != 13 )
              return 0;
            a2 = *(unsigned int *)(v74 + 32);
            if ( (unsigned int)a2 <= 0x40 )
            {
              v77 = *(_QWORD *)(v74 + 24) == 1;
            }
            else
            {
              v76 = sub_16A57B0(v74 + 24);
              v75 = v127;
              v19 = v110;
              a2 = (unsigned int)(a2 - 1);
              v77 = (_DWORD)a2 == v76;
            }
            if ( !v77 )
              return 0;
          }
          v73 = v75 + 1;
          if ( v99 == v73 )
            goto LABEL_12;
        }
      }
      goto LABEL_12;
    }
    v56 = *(_DWORD *)(v55 + 32);
    if ( v56 <= 0x40 )
    {
      v22 = *(_QWORD *)(v55 + 24) == 1;
    }
    else
    {
      v57 = sub_16A57B0(v55 + 24);
      v19 = v120;
      v22 = v56 - 1 == v57;
    }
  }
  if ( !v22 )
    return 0;
LABEL_12:
  v23 = *(_QWORD *)(v19 + 8);
  if ( !v23 || *(_QWORD *)(v23 + 8) )
    return v11;
  v24 = *(_BYTE *)(v19 + 16);
  if ( v24 != 50 )
  {
    if ( v24 != 5 )
      return v11;
    if ( *(_WORD *)(v19 + 18) != 26 )
      return v11;
    v25 = *(_DWORD *)(v19 + 20) & 0xFFFFFFF;
    v26 = *(_QWORD *)(v19 - 24 * v25);
    if ( !v26 )
      return v11;
    v27 = *(_QWORD *)(v19 + 24 * (1 - v25));
    if ( *(_BYTE *)(v27 + 16) != 13 )
    {
      if ( *(_BYTE *)(*(_QWORD *)v27 + 8LL) != 16 )
        return v11;
      v129 = v27;
      v82 = sub_15A1020((_BYTE *)v27, 4 * v25, v27, 1 - v25);
      v83 = v129;
      if ( v82 && *(_BYTE *)(v82 + 16) == 13 )
      {
        if ( !sub_1455000(v82 + 24) )
          return v11;
      }
      else
      {
        v90 = 0;
        v131 = *(_DWORD *)(*(_QWORD *)v129 + 32LL);
        while ( v131 != v90 )
        {
          v104 = v90;
          v113 = v83;
          v91 = sub_15A0A60(v83, v90);
          if ( !v91 )
            return v11;
          v92 = *(_BYTE *)(v91 + 16);
          v83 = v113;
          v93 = v104;
          if ( v92 != 9 )
          {
            if ( v92 != 13 )
              return v11;
            if ( *(_DWORD *)(v91 + 32) <= 0x40u )
            {
              if ( *(_QWORD *)(v91 + 24) != 1 )
                return v11;
            }
            else
            {
              v98 = *(_DWORD *)(v91 + 32);
              v94 = sub_16A57B0(v91 + 24);
              v83 = v113;
              v93 = v104;
              if ( v94 != v98 - 1 )
                return v11;
            }
          }
          v90 = v93 + 1;
        }
      }
      goto LABEL_22;
    }
    goto LABEL_19;
  }
  v26 = *(_QWORD *)(v19 - 48);
  if ( !v26 )
    return v11;
  v27 = *(_QWORD *)(v19 - 24);
  v59 = *(_BYTE *)(v27 + 16);
  if ( v59 == 13 )
  {
LABEL_19:
    if ( *(_DWORD *)(v27 + 32) <= 0x40u )
    {
      v28 = *(_QWORD *)(v27 + 24) == 1;
    }
    else
    {
      v116 = *(_DWORD *)(v27 + 32);
      v28 = v116 - 1 == (unsigned int)sub_16A57B0(v27 + 24);
    }
    goto LABEL_21;
  }
  if ( *(_BYTE *)(*(_QWORD *)v27 + 8LL) != 16 || v59 > 0x10u )
    return v11;
  v122 = v27;
  v60 = sub_15A1020((_BYTE *)v27, a2, v27, *(_QWORD *)v27);
  v61 = v122;
  if ( v60 && *(_BYTE *)(v60 + 16) == 13 )
  {
    if ( *(_DWORD *)(v60 + 32) <= 0x40u )
    {
      v28 = *(_QWORD *)(v60 + 24) == 1;
    }
    else
    {
      v123 = *(_DWORD *)(v60 + 32);
      v28 = v123 - 1 == (unsigned int)sub_16A57B0(v60 + 24);
    }
LABEL_21:
    if ( !v28 )
      return v11;
LABEL_22:
    v29 = *(_QWORD *)(v26 + 8);
    if ( !v29 || *(_QWORD *)(v29 + 8) )
      goto LABEL_24;
    v53 = *(_BYTE *)(v26 + 16);
    if ( v53 == 48 )
    {
      v54 = *(_QWORD *)(v26 - 48);
      if ( v54 )
      {
        v105 = *(_QWORD *)(v26 - 24);
        if ( v105 )
          goto LABEL_60;
      }
    }
    else if ( v53 == 5 && *(_WORD *)(v26 + 18) == 24 )
    {
      v54 = *(_QWORD *)(v26 - 24LL * (*(_DWORD *)(v26 + 20) & 0xFFFFFFF));
      if ( v54 )
      {
        v105 = *(_QWORD *)(v26 + 24 * (1LL - (*(_DWORD *)(v26 + 20) & 0xFFFFFFF)));
        if ( v105 )
        {
LABEL_60:
          v26 = v54;
          v30 = 1;
LABEL_25:
          v31 = *(_QWORD *)(v9 - 48);
          v32 = *(_BYTE *)(v31 + 16);
          if ( v32 == 50 )
          {
            v34 = *(_QWORD *)(v31 - 24);
            if ( v26 != *(_QWORD *)(v31 - 48) )
            {
              if ( v26 != v34 )
                return 0;
              v34 = *(_QWORD *)(v31 - 48);
            }
          }
          else
          {
            if ( v32 != 5 || *(_WORD *)(v31 + 18) != 26 )
              return 0;
            v33 = *(_QWORD *)(v31 - 24LL * (*(_DWORD *)(v31 + 20) & 0xFFFFFFF));
            v34 = *(_QWORD *)(v31 + 24 * (1LL - (*(_DWORD *)(v31 + 20) & 0xFFFFFFF)));
            if ( v26 != v33 )
            {
              if ( v34 == v26 && v33 )
              {
                v34 = *(_QWORD *)(v31 - 24LL * (*(_DWORD *)(v31 + 20) & 0xFFFFFFF));
LABEL_32:
                v117 = v30;
                v35 = sub_15A0680(a1, 1, 0);
                v36 = v35;
                if ( v117 )
                {
                  v134 = 257;
                  if ( *(_BYTE *)(v35 + 16) > 0x10u || *(_BYTE *)(v105 + 16) > 0x10u )
                  {
                    v136 = 257;
                    v62 = sub_15FB440(23, (__int64 *)v35, v105, (__int64)v135, 0);
                    v63 = *(_QWORD *)(a5 + 8);
                    if ( v63 )
                    {
                      v124 = v62;
                      v109 = *(__int64 **)(a5 + 16);
                      sub_157E9D0(v63 + 40, v62);
                      v62 = v124;
                      v64 = *(_QWORD *)(v124 + 24);
                      v65 = *v109;
                      *(_QWORD *)(v124 + 32) = v109;
                      v65 &= 0xFFFFFFFFFFFFFFF8LL;
                      *(_QWORD *)(v124 + 24) = v65 | v64 & 7;
                      *(_QWORD *)(v65 + 8) = v124 + 24;
                      *v109 = *v109 & 7 | (v124 + 24);
                    }
                    v66 = v62;
                    v125 = v62;
                    sub_164B780(v62, v133);
                    v68 = *(_QWORD *)(a5 + 80) == 0;
                    v132 = (unsigned __int8 *)v125;
                    if ( v68 )
                      sub_4263D6(v66, v133, v67);
                    (*(void (__fastcall **)(__int64, unsigned __int8 **))(a5 + 88))(a5 + 64, &v132);
                    v69 = *(_QWORD *)a5;
                    v36 = v125;
                    if ( *(_QWORD *)a5 )
                    {
                      v132 = *(unsigned __int8 **)a5;
                      sub_1623A60((__int64)&v132, v69, 2);
                      v36 = v125;
                      v70 = *(_QWORD *)(v125 + 48);
                      v71 = v125 + 48;
                      if ( v70 )
                      {
                        sub_161E7C0(v125 + 48, v70);
                        v36 = v125;
                        v71 = v125 + 48;
                      }
                      v72 = v132;
                      *(_QWORD *)(v36 + 48) = v132;
                      if ( v72 )
                      {
                        v126 = v36;
                        sub_1623210((__int64)&v132, v72, v71);
                        v36 = v126;
                      }
                    }
                  }
                  else
                  {
                    v121 = sub_15A2D50((__int64 *)v35, v105, 0, 0, a6, a7, a8);
                    v58 = sub_14DBA30(v121, *(_QWORD *)(a5 + 96), 0);
                    v36 = v121;
                    if ( v58 )
                      v36 = v58;
                  }
                }
                v136 = 257;
                v37 = sub_172AC10(a5, v34, v36, v135, a6, a7, a8);
                v136 = 257;
                v38 = sub_1729500(a5, (unsigned __int8 *)v26, (__int64)v37, v135, a6, a7, a8);
                v136 = 257;
                v39 = (__int64)v38;
                v42 = sub_15A06D0(*(__int64 ***)v38, 257, v40, v41);
                if ( *(_BYTE *)(v39 + 16) > 0x10u || *(_BYTE *)(v42 + 16) > 0x10u )
                {
                  v43 = (__int64)sub_1790840(a5, 33, v39, v42, v135);
                }
                else
                {
                  v43 = sub_15A37B0(0x21u, (_QWORD *)v39, (_QWORD *)v42, 0);
                  v44 = sub_14DBA30(v43, *(_QWORD *)(a5 + 96), 0);
                  if ( v44 )
                    v43 = v44;
                }
                v136 = 257;
                v45 = sub_1648A60(56, 1u);
                v11 = v45;
                if ( v45 )
                  sub_15FC690((__int64)v45, v43, a1, (__int64)v135, 0);
                return v11;
              }
              return 0;
            }
          }
          if ( v34 )
            goto LABEL_32;
          return 0;
        }
      }
    }
LABEL_24:
    v30 = 0;
    goto LABEL_25;
  }
  v130 = *(_QWORD *)(*(_QWORD *)v122 + 32LL);
  if ( !v130 )
    goto LABEL_22;
  v84 = 0;
  while ( 1 )
  {
    v102 = v84;
    v111 = v61;
    v85 = sub_15A0A60(v61, v84);
    if ( !v85 )
      return v11;
    v86 = *(_BYTE *)(v85 + 16);
    v61 = v111;
    v87 = v102;
    if ( v86 != 9 )
    {
      if ( v86 != 13 )
        return v11;
      if ( *(_DWORD *)(v85 + 32) <= 0x40u )
      {
        v89 = *(_QWORD *)(v85 + 24) == 1;
      }
      else
      {
        v97 = *(_DWORD *)(v85 + 32);
        v103 = v111;
        v112 = v87;
        v88 = sub_16A57B0(v85 + 24);
        v87 = v112;
        v61 = v103;
        v89 = v97 - 1 == v88;
      }
      if ( !v89 )
        return v11;
    }
    v84 = v87 + 1;
    if ( v130 == v84 )
      goto LABEL_22;
  }
}
