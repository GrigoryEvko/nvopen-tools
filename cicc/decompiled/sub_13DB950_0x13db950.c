// Function: sub_13DB950
// Address: 0x13db950
//
__int64 **__fastcall sub_13DB950(unsigned int a1, _BYTE *a2, __int64 a3, __int64 *a4, int a5)
{
  _BYTE *v7; // rbx
  __int64 v8; // rdi
  __int64 v9; // r13
  __int64 v10; // rax
  char v11; // al
  _QWORD *v12; // r13
  __int64 v13; // r15
  char v14; // al
  char v15; // al
  char v16; // al
  __int16 v17; // ax
  __int64 v18; // rax
  char v19; // al
  __int16 v20; // ax
  _BYTE *v21; // rax
  char v22; // al
  _BYTE *v23; // rsi
  __int64 v24; // rcx
  unsigned int v25; // r8d
  __int64 v26; // rax
  __int64 v27; // rdx
  unsigned int v28; // ebx
  __int64 v30; // rax
  __int64 **result; // rax
  bool v32; // dl
  char v33; // r9
  char v34; // r10
  __int64 v35; // r11
  __int64 v36; // rax
  __int64 v37; // rsi
  __int64 v38; // rsi
  __int64 v39; // rsi
  char v40; // r9
  char v41; // al
  _BYTE *v42; // r10
  _BYTE *v43; // rax
  char v44; // al
  __int64 v45; // rax
  char v46; // al
  _BYTE *v47; // rax
  char v48; // al
  __int64 v49; // r10
  __int64 v50; // rax
  __int64 v51; // rdi
  __int64 v52; // rax
  _BYTE *v53; // rax
  char v54; // al
  char v55; // al
  char v56; // al
  char v57; // al
  __int64 v58; // r8
  bool v59; // al
  __int64 v60; // rdx
  int v61; // eax
  int v62; // eax
  __int64 v63; // rax
  _BYTE *v64; // rax
  __int64 v65; // rax
  _BYTE *v66; // rax
  __int64 v67; // r8
  unsigned __int8 v68; // al
  bool v69; // al
  unsigned int v70; // r8d
  __int64 v71; // rdx
  __int64 *v72; // rdi
  unsigned int v73; // esi
  __int64 v74; // rax
  __int64 v75; // rax
  _BYTE *v76; // rax
  __int64 v77; // rax
  __int64 v78; // r8
  char v79; // al
  char v80; // al
  char v81; // al
  int v82; // eax
  __int64 v83; // rax
  __int64 v84; // rax
  char v85; // al
  char v86; // al
  __int64 v87; // r15
  unsigned int v88; // ebx
  __int64 v89; // rax
  char v90; // dl
  __int64 v91; // rax
  int v92; // eax
  bool v93; // al
  unsigned int i; // r8d
  __int64 v95; // rax
  char v96; // dl
  unsigned int v97; // r8d
  int v98; // eax
  bool v99; // al
  bool v100; // [rsp+8h] [rbp-C8h]
  __int64 v101; // [rsp+8h] [rbp-C8h]
  __int64 v102; // [rsp+10h] [rbp-C0h]
  char v103; // [rsp+10h] [rbp-C0h]
  char v104; // [rsp+10h] [rbp-C0h]
  char v105; // [rsp+10h] [rbp-C0h]
  char v106; // [rsp+1Ch] [rbp-B4h]
  bool v107; // [rsp+1Ch] [rbp-B4h]
  bool v108; // [rsp+1Ch] [rbp-B4h]
  __int64 v109; // [rsp+20h] [rbp-B0h]
  char v110; // [rsp+20h] [rbp-B0h]
  __int64 v111; // [rsp+20h] [rbp-B0h]
  __int64 v112; // [rsp+20h] [rbp-B0h]
  __int64 v113; // [rsp+28h] [rbp-A8h]
  bool v114; // [rsp+28h] [rbp-A8h]
  bool v115; // [rsp+28h] [rbp-A8h]
  int v116; // [rsp+28h] [rbp-A8h]
  unsigned int v117; // [rsp+28h] [rbp-A8h]
  char v118; // [rsp+30h] [rbp-A0h]
  int v119; // [rsp+30h] [rbp-A0h]
  bool v120; // [rsp+30h] [rbp-A0h]
  int v121; // [rsp+30h] [rbp-A0h]
  unsigned int v122; // [rsp+30h] [rbp-A0h]
  int v123; // [rsp+30h] [rbp-A0h]
  int v124; // [rsp+30h] [rbp-A0h]
  __int64 v125; // [rsp+38h] [rbp-98h]
  int v126; // [rsp+38h] [rbp-98h]
  _BYTE *v127; // [rsp+38h] [rbp-98h]
  __int64 v128; // [rsp+38h] [rbp-98h]
  __int64 v129; // [rsp+40h] [rbp-90h]
  int v130; // [rsp+40h] [rbp-90h]
  int v131; // [rsp+40h] [rbp-90h]
  __int64 v132; // [rsp+40h] [rbp-90h]
  int v133; // [rsp+40h] [rbp-90h]
  __int64 v134; // [rsp+40h] [rbp-90h]
  int v135; // [rsp+40h] [rbp-90h]
  __int64 v136; // [rsp+40h] [rbp-90h]
  unsigned int v137; // [rsp+40h] [rbp-90h]
  int v138; // [rsp+40h] [rbp-90h]
  __int64 v140; // [rsp+50h] [rbp-80h]
  int v141; // [rsp+50h] [rbp-80h]
  __int64 v143; // [rsp+58h] [rbp-78h]
  __int64 v144; // [rsp+60h] [rbp-70h] BYREF
  int v145; // [rsp+68h] [rbp-68h]
  __int64 v146; // [rsp+70h] [rbp-60h] BYREF
  int v147; // [rsp+78h] [rbp-58h]
  __int64 *v148; // [rsp+80h] [rbp-50h] BYREF
  int v149; // [rsp+88h] [rbp-48h]
  __int64 v150; // [rsp+90h] [rbp-40h] BYREF
  int v151; // [rsp+98h] [rbp-38h]

  v7 = a2;
  v8 = **(_QWORD **)a2;
  if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) == 16 )
  {
    v9 = *(_QWORD *)(*(_QWORD *)a2 + 32LL);
    v10 = sub_1643320(v8);
    v140 = sub_16463B0(v10, (unsigned int)v9);
  }
  else
  {
    v140 = sub_1643320(v8);
  }
  v11 = a2[16];
  if ( (unsigned __int8)(v11 - 35) > 0x11u )
  {
    if ( (unsigned __int8)(*(_BYTE *)(a3 + 16) - 35) > 0x11u )
    {
      v12 = 0;
      v13 = 0;
      goto LABEL_6;
    }
    v13 = a3;
    v12 = 0;
    if ( !a5 )
      goto LABEL_78;
    v118 = 0;
    v32 = 0;
    v33 = 0;
    v12 = 0;
    v125 = 0;
    v129 = 0;
    goto LABEL_54;
  }
  if ( (unsigned __int8)(*(_BYTE *)(a3 + 16) - 35) > 0x11u )
  {
    v13 = 0;
    if ( !a5 )
      goto LABEL_104;
  }
  else
  {
    v13 = a3;
    if ( !a5 )
    {
LABEL_104:
      v12 = a2;
      goto LABEL_105;
    }
  }
  if ( v11 == 35 )
  {
    v118 = 1;
    v129 = *((_QWORD *)a2 - 6);
    v32 = a3 == v129;
    v125 = *((_QWORD *)a2 - 3);
    v33 = a3 == v129 || a3 == v125;
    if ( a1 - 32 > 1 )
    {
      v114 = v32 || a3 == v125;
      v120 = a3 == *((_QWORD *)a2 - 6);
      v54 = sub_15FF7E0(a1);
      v32 = v120;
      v33 = v114;
      if ( !v54 || (v55 = sub_15F2370(a2), v32 = v120, v33 = v114, (v118 = v55) == 0) )
      {
        v110 = v33;
        v115 = v32;
        v56 = sub_15FF7F0(a1);
        v32 = v115;
        v118 = v56;
        if ( v56 )
        {
          v86 = sub_15F2380(a2);
          v32 = v115;
          v118 = v86;
          v33 = v86 & v110;
        }
        else
        {
          v33 = 0;
        }
      }
    }
  }
  else
  {
    v118 = 0;
    v32 = 0;
    v33 = 0;
    v125 = 0;
    v129 = 0;
  }
  v12 = a2;
  if ( !v13 )
    goto LABEL_55;
LABEL_54:
  if ( *(_BYTE *)(v13 + 16) != 35 )
  {
LABEL_55:
    if ( !v33 )
      goto LABEL_77;
    v113 = 0;
    v34 = 0;
    v35 = 0;
    goto LABEL_57;
  }
  v35 = *(_QWORD *)(v13 - 48);
  v113 = *(_QWORD *)(v13 - 24);
  if ( a1 - 32 <= 1 )
    goto LABEL_197;
  v104 = v33;
  v107 = v32;
  v111 = *(_QWORD *)(v13 - 48);
  v79 = sub_15FF7E0(a1);
  v35 = v111;
  v32 = v107;
  v33 = v104;
  if ( !v79 )
    goto LABEL_236;
  v80 = sub_15F2370(v13);
  v35 = v111;
  v32 = v107;
  v33 = v104;
  if ( v80 )
  {
LABEL_197:
    v34 = 1;
  }
  else
  {
LABEL_236:
    v105 = v33;
    v108 = v32;
    v112 = v35;
    v81 = sub_15FF7F0(a1);
    v35 = v112;
    v32 = v108;
    v40 = v105;
    v34 = v81;
    if ( !v81 )
    {
      if ( !v105 )
        goto LABEL_65;
LABEL_57:
      v102 = v35;
      v106 = v34;
      v100 = v32;
      v36 = sub_15A06D0(*(_QWORD *)a3);
      v37 = v125;
      if ( !v100 )
        v37 = v129;
      result = sub_13D9330(a1, v37, v36, a4, a5 - 1);
      v34 = v106;
      v35 = v102;
      if ( result )
        return result;
      goto LABEL_60;
    }
    v85 = sub_15F2380(v13);
    v33 = v105;
    v32 = v108;
    v35 = v112;
    v34 = v85;
  }
  if ( v33 )
    goto LABEL_57;
LABEL_60:
  v38 = v113;
  v103 = v34 & (v7 == (_BYTE *)v35 || v7 == (_BYTE *)v113);
  if ( !v103 )
  {
    v40 = v34;
    goto LABEL_65;
  }
  v101 = v35;
  if ( v7 != (_BYTE *)v35 )
    v38 = v35;
  v109 = v38;
  v39 = sub_15A06D0(*(_QWORD *)v7);
  result = sub_13D9330(a1, v39, v109, a4, a5 - 1);
  if ( !result )
  {
    v35 = v101;
    v40 = v103;
LABEL_65:
    if ( !v129 || !v35 )
      goto LABEL_77;
    if ( v129 == v35 || v129 == v113 )
    {
      if ( !v118 || !v40 )
        goto LABEL_77;
      if ( v129 == v35 )
        v35 = v113;
    }
    else
    {
      if ( v125 != v35 && v125 != v113 || !v118 || !v40 )
        goto LABEL_77;
      if ( v125 == v35 )
        v35 = v113;
      v125 = v129;
    }
    result = sub_13D9330(a1, v125, v35, a4, a5 - 1);
    if ( result )
      return result;
LABEL_77:
    if ( !v12 )
      goto LABEL_78;
LABEL_105:
    v48 = *((_BYTE *)v12 + 16);
    if ( v48 == 51 )
    {
      v49 = *(v12 - 6);
      if ( !v49 )
        goto LABEL_78;
      v50 = *(v12 - 3);
      if ( !v50 )
        goto LABEL_78;
      if ( a3 == v50 )
        goto LABEL_111;
    }
    else
    {
      if ( v48 != 5 )
        goto LABEL_78;
      if ( *((_WORD *)v12 + 9) != 27 )
        goto LABEL_78;
      v49 = v12[-3 * (*((_DWORD *)v12 + 5) & 0xFFFFFFF)];
      if ( !v49 )
        goto LABEL_78;
      v50 = v12[3 * (1LL - (*((_DWORD *)v12 + 5) & 0xFFFFFFF))];
      if ( a3 == v50 )
      {
        if ( !v50 )
          goto LABEL_78;
LABEL_111:
        if ( a1 == 36 )
          return (__int64 **)sub_15A0640(v140);
        if ( a1 == 35 )
          return (__int64 **)sub_15A0600(v140);
        if ( a1 - 39 <= 1 )
        {
          v119 = v49;
          sub_14C2530((unsigned int)&v144, a3, *a4, 0, a4[3], a4[4], a4[2], 0);
          sub_14C2530((unsigned int)&v148, v119, *a4, 0, a4[3], a4[4], a4[2], 0);
          if ( sub_13D0200(&v144, v145 - 1) && sub_13D0200(&v150, v151 - 1) )
          {
            v51 = v140;
            if ( a1 != 40 )
              goto LABEL_185;
            goto LABEL_118;
          }
          if ( sub_13D0200(&v146, v147 - 1) || sub_13D0200((__int64 *)&v148, v149 - 1) )
          {
            v51 = v140;
            if ( a1 != 40 )
              goto LABEL_118;
LABEL_185:
            v52 = sub_15A0640(v51);
            goto LABEL_119;
          }
          sub_135E100(&v150);
          sub_135E100((__int64 *)&v148);
          sub_135E100(&v146);
          sub_135E100(&v144);
        }
LABEL_78:
        if ( !v13 )
          goto LABEL_85;
        v41 = *(_BYTE *)(v13 + 16);
        if ( v41 == 51 )
        {
          v42 = *(_BYTE **)(v13 - 48);
          if ( !v42 )
            goto LABEL_85;
          v43 = *(_BYTE **)(v13 - 24);
          if ( !v43 )
            goto LABEL_85;
          if ( v7 == v43 )
          {
LABEL_177:
            switch ( a1 )
            {
              case '%':
                return (__int64 **)sub_15A0600(v140);
              case '"':
                return (__int64 **)sub_15A0640(v140);
              case '&':
              case ')':
                v121 = (int)v42;
                sub_14C2530((unsigned int)&v144, (_DWORD)v7, *a4, 0, a4[3], a4[4], a4[2], 0);
                sub_14C2530((unsigned int)&v148, v121, *a4, 0, a4[3], a4[4], a4[2], 0);
                if ( sub_13D0200(&v144, v145 - 1) && sub_13D0200(&v150, v151 - 1) )
                {
                  v51 = v140;
                  if ( a1 != 38 )
                    goto LABEL_185;
LABEL_118:
                  v52 = sub_15A0600(v51);
LABEL_119:
                  v143 = v52;
                  sub_135E100(&v150);
                  sub_135E100((__int64 *)&v148);
                  sub_135E100(&v146);
                  sub_135E100(&v144);
                  return (__int64 **)v143;
                }
                if ( sub_13D0200(&v146, v147 - 1) || sub_13D0200((__int64 *)&v148, v149 - 1) )
                {
                  v51 = v140;
                  if ( a1 != 38 )
                    goto LABEL_118;
                  goto LABEL_185;
                }
                sub_135E100(&v150);
                sub_135E100((__int64 *)&v148);
                sub_135E100(&v146);
                sub_135E100(&v144);
                break;
            }
LABEL_85:
            if ( !v12 )
              goto LABEL_92;
            v44 = *((_BYTE *)v12 + 16);
            if ( v44 == 50 )
            {
              v63 = *(v12 - 3);
              if ( (a3 != v63 || !v63) && a3 != *(v12 - 6) )
                goto LABEL_92;
            }
            else
            {
              if ( v44 != 5 || *((_WORD *)v12 + 9) != 26 )
                goto LABEL_92;
              v45 = v12[3 * (1LL - (*((_DWORD *)v12 + 5) & 0xFFFFFFF))];
              if ( (a3 != v45 || !v45) && a3 != v12[-3 * (*((_DWORD *)v12 + 5) & 0xFFFFFFF)] )
                goto LABEL_92;
            }
            if ( a1 == 34 )
              return (__int64 **)sub_15A0640(v140);
            if ( a1 == 37 )
              return (__int64 **)sub_15A0600(v140);
LABEL_92:
            if ( !v13 )
              goto LABEL_6;
            v46 = *(_BYTE *)(v13 + 16);
            if ( v46 == 50 )
            {
              v64 = *(_BYTE **)(v13 - 24);
              if ( (v7 != v64 || !v64) && v7 != *(_BYTE **)(v13 - 48) )
                goto LABEL_6;
            }
            else
            {
              if ( v46 != 5 || *(_WORD *)(v13 + 18) != 26 )
                goto LABEL_6;
              v47 = *(_BYTE **)(v13 + 24 * (1LL - (*(_DWORD *)(v13 + 20) & 0xFFFFFFF)));
              if ( (!v47 || v7 != v47) && v7 != *(_BYTE **)(v13 - 24LL * (*(_DWORD *)(v13 + 20) & 0xFFFFFFF)) )
                goto LABEL_6;
            }
            if ( a1 == 35 )
              return (__int64 **)sub_15A0600(v140);
            if ( a1 == 36 )
              return (__int64 **)sub_15A0640(v140);
LABEL_6:
            if ( (unsigned __int8)sub_15FF7E0(a1) )
              goto LABEL_7;
            v57 = v7[16];
            if ( v57 == 37 )
            {
              v67 = *((_QWORD *)v7 - 6);
              v68 = *(_BYTE *)(v67 + 16);
              if ( v68 == 13 )
              {
                if ( *(_DWORD *)(v67 + 32) <= 0x40u )
                {
                  v69 = *(_QWORD *)(v67 + 24) == 0;
                }
                else
                {
                  v131 = *(_DWORD *)(v67 + 32);
                  v69 = v131 == (unsigned int)sub_16A57B0(v67 + 24);
                }
                if ( !v69 )
                  goto LABEL_7;
              }
              else
              {
                if ( *(_BYTE *)(*(_QWORD *)v67 + 8LL) != 16 || v68 > 0x10u )
                  goto LABEL_7;
                v132 = *((_QWORD *)v7 - 6);
                v77 = sub_15A1020(v132);
                v78 = v132;
                if ( !v77 || *(_BYTE *)(v77 + 16) != 13 )
                {
                  v123 = *(_QWORD *)(*(_QWORD *)v132 + 32LL);
                  if ( !v123 )
                    goto LABEL_204;
                  v136 = v13;
                  v87 = v78;
                  v127 = v7;
                  v88 = 0;
                  while ( 1 )
                  {
                    v89 = sub_15A0A60(v87, v88);
                    if ( !v89 )
                      break;
                    v90 = *(_BYTE *)(v89 + 16);
                    if ( v90 != 9 )
                    {
                      if ( v90 != 13 )
                        break;
                      if ( *(_DWORD *)(v89 + 32) <= 0x40u )
                      {
                        if ( *(_QWORD *)(v89 + 24) )
                          break;
                      }
                      else
                      {
                        v116 = *(_DWORD *)(v89 + 32);
                        if ( v116 != (unsigned int)sub_16A57B0(v89 + 24) )
                          break;
                      }
                    }
                    if ( v123 == ++v88 )
                    {
                      v13 = v136;
                      v7 = v127;
                      goto LABEL_204;
                    }
                  }
                  v13 = v136;
                  v7 = v127;
                  goto LABEL_7;
                }
                if ( *(_DWORD *)(v77 + 32) > 0x40u )
                {
                  v133 = *(_DWORD *)(v77 + 32);
                  if ( v133 == (unsigned int)sub_16A57B0(v77 + 24) )
                    goto LABEL_204;
                  goto LABEL_7;
                }
                if ( *(_QWORD *)(v77 + 24) )
                  goto LABEL_7;
              }
LABEL_204:
              v60 = *((_QWORD *)v7 - 3);
              v61 = *(unsigned __int8 *)(v60 + 16);
              if ( (unsigned __int8)v61 <= 0x17u )
                goto LABEL_163;
              goto LABEL_205;
            }
            if ( v57 != 5 || *((_WORD *)v7 + 9) != 13 )
              goto LABEL_7;
            v58 = *(_QWORD *)&v7[-24 * (*((_DWORD *)v7 + 5) & 0xFFFFFFF)];
            if ( *(_BYTE *)(v58 + 16) == 13 )
            {
              if ( *(_DWORD *)(v58 + 32) <= 0x40u )
              {
                v59 = *(_QWORD *)(v58 + 24) == 0;
              }
              else
              {
                v130 = *(_DWORD *)(v58 + 32);
                v59 = v130 == (unsigned int)sub_16A57B0(v58 + 24);
              }
            }
            else
            {
              if ( *(_BYTE *)(*(_QWORD *)v58 + 8LL) != 16 )
                goto LABEL_7;
              v134 = *(_QWORD *)&v7[-24 * (*((_DWORD *)v7 + 5) & 0xFFFFFFF)];
              v84 = sub_15A1020(v58);
              if ( !v84 || *(_BYTE *)(v84 + 16) != 13 )
              {
                v124 = *(_QWORD *)(*(_QWORD *)v134 + 32LL);
                if ( v124 )
                {
                  v128 = v134;
                  for ( i = 0; i != v124; i = v97 + 1 )
                  {
                    v137 = i;
                    v95 = sub_15A0A60(v128, i);
                    if ( !v95 )
                      goto LABEL_7;
                    v96 = *(_BYTE *)(v95 + 16);
                    v97 = v137;
                    if ( v96 != 9 )
                    {
                      if ( v96 != 13 )
                        goto LABEL_7;
                      if ( *(_DWORD *)(v95 + 32) <= 0x40u )
                      {
                        v99 = *(_QWORD *)(v95 + 24) == 0;
                      }
                      else
                      {
                        v117 = v137;
                        v138 = *(_DWORD *)(v95 + 32);
                        v98 = sub_16A57B0(v95 + 24);
                        v97 = v117;
                        v99 = v138 == v98;
                      }
                      if ( !v99 )
                        goto LABEL_7;
                    }
                  }
                }
LABEL_162:
                v60 = *(_QWORD *)&v7[24 * (1LL - (*((_DWORD *)v7 + 5) & 0xFFFFFFF))];
                v61 = *(unsigned __int8 *)(v60 + 16);
                if ( (unsigned __int8)v61 <= 0x17u )
                {
LABEL_163:
                  if ( (_BYTE)v61 != 5 )
                    goto LABEL_7;
                  v62 = *(unsigned __int16 *)(v60 + 18);
LABEL_206:
                  if ( v62 != 37 || *(_BYTE *)(a3 + 16) != 13 )
                    goto LABEL_7;
                  v70 = *(_DWORD *)(a3 + 32);
                  v71 = *(_QWORD *)(a3 + 24);
                  v72 = (__int64 *)(a3 + 24);
                  v73 = v70 - 1;
                  v74 = 1LL << ((unsigned __int8)v70 - 1);
                  if ( v70 > 0x40 )
                  {
                    if ( (*(_QWORD *)(v71 + 8LL * (v73 >> 6)) & v74) != 0 )
                      goto LABEL_210;
                    v122 = v70 - 1;
                    v126 = *(_DWORD *)(a3 + 32);
                    v82 = sub_16A57B0(v72);
                    v72 = (__int64 *)(a3 + 24);
                    v73 = v122;
                    if ( v126 == v82 )
                      goto LABEL_210;
                  }
                  else if ( (v74 & v71) != 0 || !v71 )
                  {
                    goto LABEL_210;
                  }
                  switch ( a1 )
                  {
                    case '(':
                      goto LABEL_262;
                    case '\'':
                    case ' ':
                      goto LABEL_213;
                    case '!':
                      goto LABEL_262;
                  }
LABEL_210:
                  if ( sub_13D0200(v72, v73) )
                    goto LABEL_7;
                  if ( a1 != 41 )
                  {
                    if ( a1 == 38 )
                    {
LABEL_213:
                      v75 = sub_16498A0(a3);
                      return (__int64 **)sub_159C540(v75);
                    }
LABEL_7:
                    if ( !v12 )
                      goto LABEL_11;
                    v14 = *((_BYTE *)v12 + 16);
                    if ( v14 == 44 )
                    {
                      v65 = *(v12 - 3);
                      if ( a3 != v65 )
                        goto LABEL_11;
                    }
                    else
                    {
                      if ( v14 != 5 )
                        goto LABEL_11;
                      if ( *((_WORD *)v12 + 9) != 20 )
                        goto LABEL_11;
                      v65 = v12[3 * (1LL - (*((_DWORD *)v12 + 5) & 0xFFFFFFF))];
                      if ( a3 != v65 )
                        goto LABEL_11;
                    }
                    if ( v65 )
                    {
                      switch ( a1 )
                      {
                        case ' ':
                        case '"':
                        case '#':
                          return (__int64 **)sub_15A0640(v140);
                        case '!':
                        case '$':
                        case '%':
                          return (__int64 **)sub_15A0600(v140);
                        case '&':
                        case '\'':
                          sub_14C2530((unsigned int)&v148, a3, *a4, 0, a4[3], a4[4], a4[2], 0);
                          if ( !sub_13D0200((__int64 *)&v148, v149 - 1) )
                            goto LABEL_252;
LABEL_217:
                          sub_135E100(&v150);
                          sub_135E100((__int64 *)&v148);
                          return (__int64 **)sub_15A0640(v140);
                        case '(':
                        case ')':
                          sub_14C2530((unsigned int)&v148, a3, *a4, 0, a4[3], a4[4], a4[2], 0);
                          if ( sub_13D0200((__int64 *)&v148, v149 - 1) )
                          {
LABEL_219:
                            sub_135E100(&v150);
                            sub_135E100((__int64 *)&v148);
                            return (__int64 **)sub_15A0600(v140);
                          }
LABEL_252:
                          sub_135E100(&v150);
                          sub_135E100((__int64 *)&v148);
                          break;
                        default:
                          break;
                      }
                    }
LABEL_11:
                    if ( !v13 )
                      goto LABEL_15;
                    v15 = *(_BYTE *)(v13 + 16);
                    if ( v15 == 44 )
                    {
                      v53 = *(_BYTE **)(v13 - 24);
                      if ( v7 != v53 || !v53 )
                        goto LABEL_15;
                    }
                    else
                    {
                      if ( v15 != 5 )
                        goto LABEL_15;
                      if ( *(_WORD *)(v13 + 18) != 20 )
                        goto LABEL_15;
                      v76 = *(_BYTE **)(v13 + 24 * (1LL - (*(_DWORD *)(v13 + 20) & 0xFFFFFFF)));
                      if ( !v76 || v7 != v76 )
                        goto LABEL_15;
                    }
                    switch ( a1 )
                    {
                      case ' ':
                      case '$':
                      case '%':
                        return (__int64 **)sub_15A0640(v140);
                      case '!':
                      case '"':
                      case '#':
                        return (__int64 **)sub_15A0600(v140);
                      case '&':
                      case '\'':
                        sub_14C2530((unsigned int)&v148, (_DWORD)v7, *a4, 0, a4[3], a4[4], a4[2], 0);
                        if ( sub_13D0200((__int64 *)&v148, v149 - 1) )
                          goto LABEL_219;
                        goto LABEL_224;
                      case '(':
                      case ')':
                        sub_14C2530((unsigned int)&v148, (_DWORD)v7, *a4, 0, a4[3], a4[4], a4[2], 0);
                        if ( sub_13D0200((__int64 *)&v148, v149 - 1) )
                          goto LABEL_217;
LABEL_224:
                        sub_135E100(&v150);
                        sub_135E100((__int64 *)&v148);
                        break;
                      default:
                        break;
                    }
LABEL_15:
                    if ( !v12 )
                      goto LABEL_21;
                    v16 = *((_BYTE *)v12 + 16);
                    if ( v16 != 48 )
                    {
                      if ( v16 == 5 )
                      {
                        v17 = *((_WORD *)v12 + 9);
                        if ( v17 != 24 && v17 != 17 )
                          goto LABEL_21;
                        v18 = v12[-3 * (*((_DWORD *)v12 + 5) & 0xFFFFFFF)];
                        if ( a3 != v18 )
                          goto LABEL_21;
                        goto LABEL_137;
                      }
                      if ( v16 != 41 )
                        goto LABEL_21;
                    }
                    v18 = *(v12 - 6);
                    if ( a3 != v18 )
                      goto LABEL_21;
LABEL_137:
                    if ( v18 )
                    {
                      if ( a1 == 34 )
                        return (__int64 **)sub_15A0640(v140);
                      if ( a1 == 37 )
                        return (__int64 **)sub_15A0600(v140);
                    }
LABEL_21:
                    if ( !v13 )
                      goto LABEL_30;
                    v19 = *(_BYTE *)(v13 + 16);
                    if ( v19 == 48 )
                    {
                      v66 = *(_BYTE **)(v13 - 48);
                      if ( v66 && v7 == v66 )
                      {
LABEL_28:
                        if ( a1 != 36 )
                        {
                          if ( a1 != 35 )
                            goto LABEL_30;
                          return (__int64 **)sub_15A0600(v140);
                        }
                        return (__int64 **)sub_15A0640(v140);
                      }
                    }
                    else
                    {
                      if ( v19 == 5 )
                      {
                        v20 = *(_WORD *)(v13 + 18);
                        if ( v20 != 24 && v20 != 17 )
                          goto LABEL_30;
                        v21 = *(_BYTE **)(v13 - 24LL * (*(_DWORD *)(v13 + 20) & 0xFFFFFFF));
                        if ( v7 != v21 )
                          goto LABEL_30;
                      }
                      else
                      {
                        if ( v19 != 41 )
                          goto LABEL_30;
                        v21 = *(_BYTE **)(v13 - 48);
                        if ( v7 != v21 )
                          goto LABEL_30;
                      }
                      if ( v21 )
                        goto LABEL_28;
                    }
LABEL_30:
                    if ( *(_BYTE *)(a3 + 16) != 13 || !v12 )
                      goto LABEL_127;
                    v148 = &v144;
                    v22 = *((_BYTE *)v12 + 16);
                    if ( v22 == 47 )
                    {
                      v23 = (_BYTE *)*(v12 - 6);
                      if ( !(unsigned __int8)sub_13D2630(&v148, v23) )
                        goto LABEL_127;
                    }
                    else
                    {
                      if ( v22 != 5 )
                        goto LABEL_127;
                      if ( *((_WORD *)v12 + 9) != 23 )
                        goto LABEL_127;
                      v23 = (_BYTE *)v12[-3 * (*((_DWORD *)v12 + 5) & 0xFFFFFFF)];
                      if ( !(unsigned __int8)sub_13D7780(&v148, v23) )
                        goto LABEL_127;
                    }
                    if ( *(_DWORD *)(v144 + 8) > 0x40u )
                    {
                      if ( (unsigned int)sub_16A5940(v144) != 1 )
                        goto LABEL_127;
                    }
                    else if ( !*(_QWORD *)v144 || (*(_QWORD *)v144 & (*(_QWORD *)v144 - 1LL)) != 0 )
                    {
                      goto LABEL_127;
                    }
                    if ( *(_DWORD *)(a3 + 32) > 0x40u )
                    {
                      if ( (unsigned int)sub_16A5940(a3 + 24) == 1 )
                        goto LABEL_45;
                    }
                    else
                    {
                      v26 = *(_QWORD *)(a3 + 24);
                      if ( v26 )
                      {
                        v27 = v26 - 1;
                        if ( (v26 & (v26 - 1)) == 0 )
                          goto LABEL_45;
                      }
                    }
                    if ( (unsigned __int8)sub_15F2380(v12)
                      || (unsigned __int8)sub_15F2370(v12)
                      || ((v27 = *(unsigned int *)(v144 + 8), (unsigned int)v27 <= 0x40)
                        ? (v93 = *(_QWORD *)v144 == 1)
                        : (v141 = *(_DWORD *)(v144 + 8),
                           v92 = sub_16A57B0(v144),
                           v27 = (unsigned int)(v141 - 1),
                           v93 = (_DWORD)v27 == v92),
                          v93 || !sub_13D01C0(a3 + 24)) )
                    {
                      if ( a1 == 32 )
                        goto LABEL_302;
                      if ( a1 == 33 )
                      {
LABEL_51:
                        v30 = sub_16498A0(a3);
                        return (__int64 **)sub_159C4F0(v30);
                      }
                    }
LABEL_45:
                    if ( !(unsigned __int8)sub_13CFF40((__int64 *)(a3 + 24), (__int64)v23, v27, v24, v25) )
                      goto LABEL_127;
                    v28 = *(_DWORD *)(v144 + 8);
                    if ( !(v28 <= 0x40 ? *(_QWORD *)v144 == 1 : v28 - 1 == (unsigned int)sub_16A57B0(v144)) )
                      goto LABEL_127;
                    if ( a1 != 34 )
                    {
                      if ( a1 == 37 )
                        goto LABEL_51;
LABEL_127:
                      if ( v12 != 0
                        && a5 != 0
                        && v13
                        && *(_BYTE *)(v13 + 16) == *((_BYTE *)v12 + 16)
                        && *(_QWORD *)(v13 - 24) == *(v12 - 3) )
                      {
                        switch ( *((_BYTE *)v12 + 16) )
                        {
                          case ')':
                          case '0':
                            if ( (unsigned __int8)sub_15FF7F0(a1) )
                              return 0;
                            goto LABEL_240;
                          case '*':
                            if ( a1 - 32 > 1 )
                              return 0;
                            goto LABEL_240;
                          case '/':
                            if ( (unsigned __int8)sub_15F2370(v12) && (unsigned __int8)sub_15F2370(v13) )
                            {
                              if ( (!(unsigned __int8)sub_15F2380(v12) || !(unsigned __int8)sub_15F2380(v13))
                                && (unsigned __int8)sub_15FF7F0(a1) )
                              {
                                return 0;
                              }
                              return sub_13D9330(a1, *(v12 - 6), *(_QWORD *)(v13 - 48), a4, a5 - 1);
                            }
                            if ( (unsigned __int8)sub_15F2380(v12) && (unsigned __int8)sub_15F2380(v13) )
                              return sub_13D9330(a1, *(v12 - 6), *(_QWORD *)(v13 - 48), a4, a5 - 1);
                            break;
                          case '1':
LABEL_240:
                            if ( (unsigned __int8)sub_15F23D0(v12) && (unsigned __int8)sub_15F23D0(v13) )
                              return sub_13D9330(a1, *(v12 - 6), *(_QWORD *)(v13 - 48), a4, a5 - 1);
                            return 0;
                          default:
                            return 0;
                        }
                      }
                      return 0;
                    }
LABEL_302:
                    v91 = sub_16498A0(a3);
                    return (__int64 **)sub_159C540(v91);
                  }
LABEL_262:
                  v83 = sub_16498A0(a3);
                  return (__int64 **)sub_159C4F0(v83);
                }
LABEL_205:
                v62 = v61 - 24;
                goto LABEL_206;
              }
              if ( *(_DWORD *)(v84 + 32) <= 0x40u )
              {
                v59 = *(_QWORD *)(v84 + 24) == 0;
              }
              else
              {
                v135 = *(_DWORD *)(v84 + 32);
                v59 = v135 == (unsigned int)sub_16A57B0(v84 + 24);
              }
            }
            if ( !v59 )
              goto LABEL_7;
            goto LABEL_162;
          }
        }
        else
        {
          if ( v41 != 5 )
            goto LABEL_85;
          if ( *(_WORD *)(v13 + 18) != 27 )
            goto LABEL_85;
          v42 = *(_BYTE **)(v13 - 24LL * (*(_DWORD *)(v13 + 20) & 0xFFFFFFF));
          if ( !v42 )
            goto LABEL_85;
          v43 = *(_BYTE **)(v13 + 24 * (1LL - (*(_DWORD *)(v13 + 20) & 0xFFFFFFF)));
          if ( v7 == v43 )
          {
            if ( !v43 )
              goto LABEL_85;
            goto LABEL_177;
          }
        }
        if ( !v43 || v7 != v42 )
          goto LABEL_85;
        LODWORD(v42) = (_DWORD)v43;
        goto LABEL_177;
      }
    }
    if ( !v50 || a3 != v49 )
      goto LABEL_78;
    LODWORD(v49) = v50;
    goto LABEL_111;
  }
  return result;
}
