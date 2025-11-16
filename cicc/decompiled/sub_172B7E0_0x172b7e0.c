// Function: sub_172B7E0
// Address: 0x172b7e0
//
unsigned __int8 *__fastcall sub_172B7E0(__int64 a1, __int64 a2, __int64 a3, double a4, double a5, double a6)
{
  __int64 v9; // rsi
  __int64 v10; // rdx
  __int64 ***v11; // r14
  __int64 ***v12; // r8
  __int64 ***v13; // r9
  __int64 ***v14; // r15
  int v15; // r15d
  int v16; // r15d
  bool v17; // al
  __int64 v18; // r9
  char v19; // di
  __int64 v21; // rax
  _QWORD *v22; // r14
  _QWORD *v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rax
  __int64 v26; // rdi
  __int64 v27; // rax
  int v28; // eax
  __int64 v29; // rcx
  bool v30; // al
  __int64 v31; // rdx
  __int64 v32; // r9
  __int64 v33; // r8
  __int64 v34; // rcx
  unsigned __int8 v35; // al
  bool v36; // al
  __int64 v37; // rdx
  _BYTE *v38; // r8
  __int64 v39; // rcx
  char v40; // al
  __int64 v41; // rax
  __int64 v42; // r13
  __int64 v43; // r12
  unsigned __int8 *v44; // rax
  char v45; // al
  __int64 v46; // rdx
  char v47; // al
  __int64 v48; // rdx
  __int64 v49; // rdx
  __int64 v50; // rcx
  __int64 v51; // rdi
  int v52; // eax
  bool v53; // al
  __int64 v54; // rdi
  int v55; // eax
  bool v56; // al
  bool v57; // al
  __int64 v58; // rax
  __int64 v59; // r13
  __int64 v60; // r12
  unsigned __int8 *v61; // rax
  __int64 v62; // rax
  __int64 v63; // r9
  bool v64; // al
  __int64 ***v65; // rdi
  __int64 v66; // rax
  bool v67; // al
  __int64 v68; // rdi
  int v69; // eax
  bool v70; // al
  __int64 v71; // rax
  char v72; // di
  bool v73; // al
  __int64 v74; // rax
  char v75; // di
  bool v76; // al
  __int64 v77; // rax
  __int64 v78; // r8
  __int64 v79; // rax
  unsigned int v80; // ecx
  __int64 v81; // rax
  char v82; // si
  unsigned int v83; // ecx
  int v84; // eax
  bool v85; // al
  __int64 v86; // rax
  char v87; // di
  int v88; // eax
  bool v89; // al
  __int64 v90; // [rsp+0h] [rbp-A0h]
  unsigned int v91; // [rsp+8h] [rbp-98h]
  unsigned int v92; // [rsp+8h] [rbp-98h]
  __int64 v93; // [rsp+8h] [rbp-98h]
  __int64 v94; // [rsp+8h] [rbp-98h]
  int v95; // [rsp+8h] [rbp-98h]
  unsigned int v96; // [rsp+10h] [rbp-90h]
  __int64 v97; // [rsp+10h] [rbp-90h]
  _BYTE *v98; // [rsp+10h] [rbp-90h]
  __int64 v99; // [rsp+10h] [rbp-90h]
  unsigned int v100; // [rsp+10h] [rbp-90h]
  int v101; // [rsp+10h] [rbp-90h]
  unsigned int v102; // [rsp+10h] [rbp-90h]
  __int64 ***v103; // [rsp+18h] [rbp-88h]
  unsigned int v104; // [rsp+18h] [rbp-88h]
  unsigned int v105; // [rsp+18h] [rbp-88h]
  unsigned int v106; // [rsp+18h] [rbp-88h]
  unsigned int v107; // [rsp+18h] [rbp-88h]
  __int64 v108; // [rsp+18h] [rbp-88h]
  unsigned int v109; // [rsp+18h] [rbp-88h]
  __int64 v110; // [rsp+18h] [rbp-88h]
  unsigned int v111; // [rsp+18h] [rbp-88h]
  unsigned int v112; // [rsp+18h] [rbp-88h]
  unsigned int v113; // [rsp+20h] [rbp-80h]
  _BYTE *v114; // [rsp+20h] [rbp-80h]
  __int64 v115; // [rsp+20h] [rbp-80h]
  __int64 v116; // [rsp+20h] [rbp-80h]
  unsigned int v117; // [rsp+20h] [rbp-80h]
  __int64 v118; // [rsp+20h] [rbp-80h]
  unsigned int v119; // [rsp+20h] [rbp-80h]
  __int64 v120; // [rsp+20h] [rbp-80h]
  __int64 v121; // [rsp+20h] [rbp-80h]
  __int64 v122; // [rsp+20h] [rbp-80h]
  __int64 v123; // [rsp+20h] [rbp-80h]
  __int64 v124; // [rsp+20h] [rbp-80h]
  __int64 ***v125; // [rsp+28h] [rbp-78h]
  __int64 v126; // [rsp+28h] [rbp-78h]
  _BYTE *v127; // [rsp+28h] [rbp-78h]
  __int64 ***v128; // [rsp+28h] [rbp-78h]
  int v129; // [rsp+28h] [rbp-78h]
  __int64 v130; // [rsp+28h] [rbp-78h]
  _BYTE *v131; // [rsp+28h] [rbp-78h]
  __int64 v132; // [rsp+28h] [rbp-78h]
  __int64 v133; // [rsp+28h] [rbp-78h]
  __int64 v134; // [rsp+28h] [rbp-78h]
  int v135; // [rsp+28h] [rbp-78h]
  int v136; // [rsp+28h] [rbp-78h]
  int v137; // [rsp+28h] [rbp-78h]
  int v138; // [rsp+28h] [rbp-78h]
  int v139; // [rsp+28h] [rbp-78h]
  __int64 v140[2]; // [rsp+30h] [rbp-70h] BYREF
  __int16 v141; // [rsp+40h] [rbp-60h]
  __int64 v142[2]; // [rsp+50h] [rbp-50h] BYREF
  __int16 v143; // [rsp+60h] [rbp-40h]

  v9 = *(_WORD *)(a3 + 18) & 0x7FFF;
  if ( !sub_14CF780(*(_WORD *)(a2 + 18) & 0x7FFF, v9) )
  {
    v12 = *(__int64 ****)(a3 - 24);
    v14 = *(__int64 ****)(a3 - 48);
    v13 = *(__int64 ****)(a2 - 24);
    v11 = *(__int64 ****)(a2 - 48);
    goto LABEL_9;
  }
  v11 = *(__int64 ****)(a2 - 48);
  v12 = *(__int64 ****)(a3 - 24);
  v13 = *(__int64 ****)(a2 - 24);
  v14 = *(__int64 ****)(a3 - 48);
  if ( v11 == v12 && v14 == v13 )
  {
    v9 = a2 - 24;
    *(_WORD *)(a2 + 18) = sub_15FF5D0(*(_WORD *)(a2 + 18) & 0x7FFF) | *(_WORD *)(a2 + 18) & 0x8000;
    sub_16484A0((__int64 *)(a2 - 48), (__int64 *)(a2 - 24));
    v14 = *(__int64 ****)(a3 - 48);
    v11 = *(__int64 ****)(a2 - 48);
    v12 = *(__int64 ****)(a3 - 24);
    v13 = *(__int64 ****)(a2 - 24);
  }
  if ( v14 != v11 || v13 != v12 )
  {
LABEL_9:
    v21 = *(_QWORD *)(a2 + 8);
    if ( !v21 || *(_QWORD *)(v21 + 8) )
    {
      v27 = *(_QWORD *)(a3 + 8);
      if ( !v27 || *(_QWORD *)(v27 + 8) || *v14 != *v11 )
        goto LABEL_12;
    }
    else if ( *v14 != *v11 )
    {
      goto LABEL_12;
    }
    v28 = *(unsigned __int16 *)(a2 + 18);
    v29 = *(unsigned __int16 *)(a3 + 18);
    BYTE1(v28) &= ~0x80u;
    BYTE1(v29) &= ~0x80u;
    if ( v28 == 38 )
    {
      v105 = v29;
      v115 = (__int64)v12;
      v128 = v13;
      v45 = sub_17279D0(v13, v9, v10, v29);
      if ( v105 == 38 && v45 )
      {
        if ( !(unsigned __int8)sub_17279D0((_BYTE *)v115, v9, v46, v105) )
        {
          sub_17279D0(v128, v9, v31, v34);
          goto LABEL_12;
        }
LABEL_64:
        v58 = sub_15A06D0(*v11, v9, v31, v34);
        v59 = *(_QWORD *)(a1 + 8);
        v60 = v58;
        v143 = 257;
        v141 = 257;
        v61 = sub_172B670(v59, (__int64)v11, (__int64)v14, v140, a4, a5, a6);
        return sub_17203D0(v59, 40, (__int64)v61, v60, v142);
      }
      v47 = sub_17279D0(v128, v9, v46, v105);
      if ( v105 == 40 && v47 && *(_BYTE *)(v115 + 16) <= 0x10u )
      {
        if ( sub_1593BB0(v115, v9, v48, v105) )
          goto LABEL_40;
        if ( *(_BYTE *)(v115 + 16) == 13 )
        {
          if ( *(_DWORD *)(v115 + 32) <= 0x40u )
          {
            if ( *(_QWORD *)(v115 + 24) )
              goto LABEL_12;
            goto LABEL_40;
          }
          v129 = *(_DWORD *)(v115 + 32);
          v40 = v129 == (unsigned int)sub_16A57B0(v115 + 24);
        }
        else
        {
          if ( *(_BYTE *)(*(_QWORD *)v115 + 8LL) != 16 )
            goto LABEL_12;
          v77 = sub_15A1020((_BYTE *)v115, v9, v49, v50);
          v78 = v115;
          if ( !v77 || *(_BYTE *)(v77 + 16) != 13 )
          {
            v80 = 0;
            v138 = *(_DWORD *)(*(_QWORD *)v115 + 32LL);
            while ( v138 != v80 )
            {
              v112 = v80;
              v123 = v78;
              v81 = sub_15A0A60(v78, v80);
              if ( !v81 )
                goto LABEL_12;
              v82 = *(_BYTE *)(v81 + 16);
              v78 = v123;
              v83 = v112;
              if ( v82 != 9 )
              {
                if ( v82 != 13 )
                  goto LABEL_12;
                if ( *(_DWORD *)(v81 + 32) <= 0x40u )
                {
                  v85 = *(_QWORD *)(v81 + 24) == 0;
                }
                else
                {
                  v101 = *(_DWORD *)(v81 + 32);
                  v84 = sub_16A57B0(v81 + 24);
                  v78 = v123;
                  v83 = v112;
                  v85 = v101 == v84;
                }
                if ( !v85 )
                  goto LABEL_12;
              }
              v80 = v83 + 1;
            }
            goto LABEL_40;
          }
          if ( *(_DWORD *)(v77 + 32) <= 0x40u )
          {
            v40 = *(_QWORD *)(v77 + 24) == 0;
          }
          else
          {
            v137 = *(_DWORD *)(v77 + 32);
            v40 = v137 == (unsigned int)sub_16A57B0(v77 + 24);
          }
        }
LABEL_39:
        if ( !v40 )
          goto LABEL_12;
LABEL_40:
        v41 = sub_15A04A0(*v11);
        v42 = *(_QWORD *)(a1 + 8);
        v143 = 257;
        v43 = v41;
        v141 = 257;
        v44 = sub_172B670(v42, (__int64)v11, (__int64)v14, v140, a4, a5, a6);
        return sub_17203D0(v42, 38, (__int64)v44, v43, v142);
      }
      goto LABEL_12;
    }
    if ( v28 != 40 )
      goto LABEL_12;
    v113 = v29;
    v126 = (__int64)v12;
    if ( *((_BYTE *)v13 + 16) > 0x10u )
      goto LABEL_12;
    v103 = v13;
    v30 = sub_1593BB0((__int64)v13, v9, v10, v29);
    v32 = (__int64)v103;
    v33 = v126;
    v34 = v113;
    if ( !v30 )
    {
      v35 = *((_BYTE *)v103 + 16);
      if ( v35 == 13 )
      {
        v31 = *((unsigned int *)v103 + 8);
        if ( (unsigned int)v31 <= 0x40 )
        {
          v53 = v103[3] == 0;
        }
        else
        {
          v51 = (__int64)(v103 + 3);
          v96 = *((_DWORD *)v103 + 8);
          v106 = v113;
          v116 = v126;
          v130 = v32;
          v52 = sub_16A57B0(v51);
          v31 = v96;
          v32 = v130;
          v33 = v116;
          v34 = v106;
          v53 = v96 == v52;
        }
        if ( !v53 )
          goto LABEL_36;
      }
      else
      {
        v9 = (__int64)*v103;
        if ( *((_BYTE *)*v103 + 8) != 16 )
        {
LABEL_35:
          if ( v35 > 0x10u )
            goto LABEL_12;
LABEL_36:
          v104 = v34;
          v114 = (_BYTE *)v33;
          v127 = (_BYTE *)v32;
          v36 = sub_1593BB0(v32, v9, v31, v34);
          v38 = v114;
          v39 = v104;
          if ( v36 )
            goto LABEL_37;
          if ( v127[16] == 13 )
          {
            v37 = *((unsigned int *)v127 + 8);
            if ( (unsigned int)v37 <= 0x40 )
            {
              if ( *((_QWORD *)v127 + 3) )
                goto LABEL_12;
            }
            else
            {
              v54 = (__int64)(v127 + 24);
              v107 = *((_DWORD *)v127 + 8);
              v117 = v39;
              v131 = v38;
              v55 = sub_16A57B0(v54);
              v37 = v107;
              v38 = v131;
              v39 = v117;
              if ( v107 != v55 )
                goto LABEL_12;
            }
            goto LABEL_37;
          }
          if ( *(_BYTE *)(*(_QWORD *)v127 + 8LL) != 16 )
            goto LABEL_12;
          v62 = sub_15A1020(v127, v9, v37, v104);
          v63 = (__int64)v127;
          v38 = v114;
          v39 = v104;
          if ( v62 && *(_BYTE *)(v62 + 16) == 13 )
          {
            v64 = sub_13D01C0(v62 + 24);
            v38 = v114;
            v39 = v104;
            if ( !v64 )
              goto LABEL_12;
            goto LABEL_37;
          }
          v135 = *(_QWORD *)(*(_QWORD *)v127 + 32LL);
          if ( !v135 )
          {
LABEL_37:
            if ( (_DWORD)v39 != 38 )
              goto LABEL_12;
            v40 = sub_17279D0(v38, v9, v37, v39);
            goto LABEL_39;
          }
          LODWORD(v9) = 0;
          while ( 1 )
          {
            v91 = v39;
            v98 = v38;
            v120 = v63;
            v71 = sub_15A0A60(v63, v9);
            if ( !v71 )
              break;
            v72 = *(_BYTE *)(v71 + 16);
            v63 = v120;
            v38 = v98;
            v39 = v91;
            if ( v72 != 9 )
            {
              if ( v72 != 13 )
                break;
              v73 = sub_13D01C0(v71 + 24);
              v38 = v98;
              v39 = v91;
              v63 = v120;
              if ( !v73 )
                break;
            }
            v9 = (unsigned int)(v9 + 1);
            if ( v135 == (_DWORD)v9 )
              goto LABEL_37;
          }
LABEL_12:
          v22 = sub_13E1140(27, (unsigned __int8 *)a2, (unsigned __int8 *)a3, (_QWORD *)(a1 + 2672));
          if ( v22 )
          {
            v23 = sub_13E1140(26, (unsigned __int8 *)a2, (unsigned __int8 *)a3, (_QWORD *)(a1 + 2672));
            if ( v23 )
            {
              if ( (_QWORD *)a2 == v22 && (_QWORD *)a3 == v23 )
              {
                v24 = *(_QWORD *)(a3 + 8);
                if ( v24 )
                {
                  if ( !*(_QWORD *)(v24 + 8) )
                  {
                    *(_WORD *)(a3 + 18) = sub_15FF0F0(*(_WORD *)(a3 + 18) & 0x7FFF) | *(_WORD *)(a3 + 18) & 0x8000;
                    goto LABEL_23;
                  }
                }
              }
              if ( (_QWORD *)a3 == v22 && (_QWORD *)a2 == v23 )
              {
                v25 = *(_QWORD *)(a2 + 8);
                if ( v25 )
                {
                  if ( !*(_QWORD *)(v25 + 8) )
                  {
                    *(_WORD *)(a2 + 18) = sub_15FF0F0(*(_WORD *)(a2 + 18) & 0x7FFF) | *(_WORD *)(a2 + 18) & 0x8000;
LABEL_23:
                    v26 = *(_QWORD *)(a1 + 8);
                    v143 = 257;
                    return sub_1729500(v26, (unsigned __int8 *)a2, a3, v142, a4, a5, a6);
                  }
                }
              }
            }
          }
          return 0;
        }
        v65 = v103;
        v109 = v113;
        v118 = v126;
        v133 = v32;
        v66 = sub_15A1020(v65, v9, v31, v34);
        v32 = v133;
        v33 = v118;
        v34 = v109;
        if ( v66 && *(_BYTE *)(v66 + 16) == 13 )
        {
          v110 = v133;
          v119 = v34;
          v134 = v33;
          v67 = sub_13D01C0(v66 + 24);
          v33 = v134;
          v34 = v119;
          v32 = v110;
          if ( !v67 )
            goto LABEL_34;
        }
        else
        {
          v136 = *(_QWORD *)(*(_QWORD *)v133 + 32LL);
          if ( v136 )
          {
            LODWORD(v9) = 0;
            do
            {
              v92 = v34;
              v99 = v33;
              v121 = v32;
              v74 = sub_15A0A60(v32, v9);
              v32 = v121;
              v9 = (unsigned int)v9;
              v33 = v99;
              v34 = v92;
              if ( !v74 )
                goto LABEL_34;
              v75 = *(_BYTE *)(v74 + 16);
              if ( v75 != 9 )
              {
                if ( v75 != 13 )
                  goto LABEL_34;
                v93 = v121;
                v100 = v34;
                v122 = v33;
                v76 = sub_13D01C0(v74 + 24);
                v33 = v122;
                v9 = (unsigned int)v9;
                v34 = v100;
                v32 = v93;
                if ( !v76 )
                  goto LABEL_34;
              }
              v9 = (unsigned int)(v9 + 1);
            }
            while ( v136 != (_DWORD)v9 );
          }
        }
      }
    }
    if ( (_DWORD)v34 == 40 && *(_BYTE *)(v33 + 16) <= 0x10u )
    {
      v108 = v32;
      v132 = v33;
      v57 = sub_1593BB0(v33, v9, v31, v34);
      v33 = v132;
      v34 = 40;
      v32 = v108;
      if ( v57 )
        goto LABEL_64;
      if ( *(_BYTE *)(v132 + 16) == 13 )
      {
        v31 = *(unsigned int *)(v132 + 32);
        if ( (unsigned int)v31 > 0x40 )
        {
          v97 = v108;
          v68 = v132 + 24;
          v111 = *(_DWORD *)(v132 + 32);
          goto LABEL_83;
        }
        if ( !*(_QWORD *)(v132 + 24) )
          goto LABEL_64;
      }
      else if ( *(_BYTE *)(*(_QWORD *)v132 + 8LL) == 16 )
      {
        v79 = sub_15A1020((_BYTE *)v132, v9, v31, 40);
        v33 = v132;
        v34 = 40;
        v32 = v108;
        if ( !v79 || *(_BYTE *)(v79 + 16) != 13 )
        {
          v9 = 0;
          v139 = *(_DWORD *)(*(_QWORD *)v132 + 32LL);
          while ( 1 )
          {
            v94 = v32;
            v102 = v34;
            if ( v139 == (_DWORD)v9 )
              goto LABEL_64;
            v124 = v33;
            v86 = sub_15A0A60(v33, v9);
            v33 = v124;
            v9 = (unsigned int)v9;
            v34 = v102;
            v32 = v94;
            if ( !v86 )
              goto LABEL_34;
            v87 = *(_BYTE *)(v86 + 16);
            if ( v87 != 9 )
            {
              if ( v87 != 13 )
                goto LABEL_34;
              if ( *(_DWORD *)(v86 + 32) <= 0x40u )
              {
                v89 = *(_QWORD *)(v86 + 24) == 0;
              }
              else
              {
                v90 = v94;
                v95 = *(_DWORD *)(v86 + 32);
                v88 = sub_16A57B0(v86 + 24);
                v33 = v124;
                v34 = v102;
                v9 = (unsigned int)v9;
                v32 = v90;
                v89 = v95 == v88;
              }
              if ( !v89 )
                goto LABEL_34;
            }
            v9 = (unsigned int)(v9 + 1);
          }
        }
        v31 = *(unsigned int *)(v79 + 32);
        if ( (unsigned int)v31 <= 0x40 )
        {
          v70 = *(_QWORD *)(v79 + 24) == 0;
LABEL_84:
          if ( v70 )
            goto LABEL_64;
          goto LABEL_34;
        }
        v97 = v108;
        v68 = v79 + 24;
        v111 = *(_DWORD *)(v79 + 32);
LABEL_83:
        v69 = sub_16A57B0(v68);
        v31 = v111;
        v33 = v132;
        v34 = 40;
        v32 = v97;
        v70 = v111 == v69;
        goto LABEL_84;
      }
    }
LABEL_34:
    v35 = *(_BYTE *)(v32 + 16);
    goto LABEL_35;
  }
  v125 = v13;
  v15 = sub_14CF5F0(a2, 0);
  v16 = sub_14CF5F0(a3, 0) ^ v15;
  v17 = sub_15FF7F0(*(_WORD *)(a2 + 18) & 0x7FFF);
  v18 = (__int64)v125;
  v19 = 1;
  if ( !v17 )
  {
    v56 = sub_15FF7F0(*(_WORD *)(a3 + 18) & 0x7FFF);
    v18 = (__int64)v125;
    v19 = v56;
  }
  return sub_1727CB0(v19, v16, (__int64)v11, v18, *(_QWORD *)(a1 + 8));
}
