// Function: sub_1C5BCD0
// Address: 0x1c5bcd0
//
_BYTE *__fastcall sub_1C5BCD0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __m128i a7,
        __m128i a8,
        __int64 *a9)
{
  _BYTE **v11; // rax
  __int64 v12; // rbx
  __int64 *v13; // rbx
  __int64 v14; // r12
  __int64 v15; // r14
  __int64 *v16; // r14
  _BYTE *v17; // rdx
  _BYTE *v18; // rsi
  __int64 v19; // rax
  __int64 v20; // rbx
  unsigned __int64 v21; // r12
  __int64 v22; // r13
  __int64 v23; // rbx
  _QWORD *v24; // rax
  __int64 v25; // rax
  __int64 v26; // rax
  _BYTE *v27; // rsi
  _BYTE *v28; // rsi
  char v29; // al
  __int64 v30; // rdx
  __int64 v31; // rax
  __int64 v32; // rdi
  __int64 v33; // rax
  _BYTE *v34; // rsi
  __int64 v35; // r8
  __int64 v36; // rax
  __int64 v37; // rax
  __int64 v38; // rax
  char *v39; // rax
  __int64 v40; // rax
  unsigned __int8 *v41; // rdx
  __int64 v42; // r8
  unsigned __int8 v43; // al
  __int64 v44; // rbx
  __int64 v45; // rax
  __int64 v47; // rax
  __int64 v48; // rdx
  __int64 v49; // r14
  __int64 v50; // rbx
  unsigned __int8 *v51; // rax
  char *v52; // rsi
  __int64 v53; // rsi
  __int64 **v54; // r14
  unsigned __int64 v55; // r13
  __int64 v56; // rax
  __int64 ***v57; // rax
  __int64 v58; // r13
  _BYTE *v59; // rsi
  __int64 v60; // r12
  bool v61; // r8
  char *v62; // rax
  __int64 v63; // rax
  __int64 v64; // r14
  __int64 v65; // rbx
  __int64 v66; // r12
  __int64 v67; // rbx
  _QWORD *v68; // rdi
  unsigned __int8 v69; // al
  unsigned int v70; // esi
  __int64 v71; // rax
  __int64 v72; // rsi
  _QWORD *v73; // r10
  unsigned __int64 v74; // rax
  _QWORD *v75; // rdx
  __int64 v76; // rcx
  __int64 v77; // rax
  __int64 *v78; // rdx
  __int64 v79; // rax
  unsigned __int8 *v80; // rsi
  char *v81; // rax
  __int64 v82; // rax
  int v83; // eax
  __int64 v84; // rdx
  unsigned int v85; // edx
  bool v89; // [rsp+1Ch] [rbp-184h]
  __int64 v90; // [rsp+20h] [rbp-180h]
  _BYTE *v91; // [rsp+28h] [rbp-178h]
  unsigned int v93; // [rsp+38h] [rbp-168h]
  __int64 v94; // [rsp+40h] [rbp-160h]
  __int64 v95; // [rsp+40h] [rbp-160h]
  __int64 v96; // [rsp+40h] [rbp-160h]
  __int64 v97; // [rsp+40h] [rbp-160h]
  __int64 *v98; // [rsp+48h] [rbp-158h]
  char **v99; // [rsp+48h] [rbp-158h]
  char *v101; // [rsp+50h] [rbp-150h]
  __int64 v102; // [rsp+58h] [rbp-148h]
  __int64 v103; // [rsp+58h] [rbp-148h]
  __int64 *v104; // [rsp+60h] [rbp-140h]
  __int64 v105; // [rsp+60h] [rbp-140h]
  __int64 v106; // [rsp+68h] [rbp-138h]
  __int64 v107; // [rsp+70h] [rbp-130h] BYREF
  char *v108; // [rsp+78h] [rbp-128h] BYREF
  __int64 v109; // [rsp+80h] [rbp-120h] BYREF
  _BYTE *v110; // [rsp+88h] [rbp-118h]
  _BYTE *v111; // [rsp+90h] [rbp-110h]
  __int64 v112; // [rsp+A0h] [rbp-100h] BYREF
  _BYTE *v113; // [rsp+A8h] [rbp-F8h]
  _BYTE *v114; // [rsp+B0h] [rbp-F0h]
  __int64 v115; // [rsp+C0h] [rbp-E0h] BYREF
  _BYTE *v116; // [rsp+C8h] [rbp-D8h]
  _BYTE *v117; // [rsp+D0h] [rbp-D0h]
  __int64 **v118; // [rsp+E0h] [rbp-C0h] BYREF
  _BYTE *v119; // [rsp+E8h] [rbp-B8h]
  _BYTE *v120; // [rsp+F0h] [rbp-B0h]
  unsigned __int8 *v121[2]; // [rsp+100h] [rbp-A0h] BYREF
  char v122; // [rsp+110h] [rbp-90h]
  char v123; // [rsp+111h] [rbp-8Fh]
  __int64 **v124; // [rsp+120h] [rbp-80h] BYREF
  char *v125; // [rsp+128h] [rbp-78h]
  __int64 v126; // [rsp+130h] [rbp-70h]
  __int64 v127; // [rsp+138h] [rbp-68h]
  __int64 v128; // [rsp+140h] [rbp-60h]
  int v129; // [rsp+148h] [rbp-58h]
  __int64 v130; // [rsp+150h] [rbp-50h]
  __int64 v131; // [rsp+158h] [rbp-48h]

  if ( !a2 )
    return 0;
  v89 = (*(_BYTE *)(a5 + 17) & 2) != 0;
  v102 = sub_1456E10(*(_QWORD *)(a1 + 184), *(_QWORD *)a5);
  if ( (*(_BYTE *)(a5 + 23) & 0x40) != 0 )
    v11 = *(_BYTE ***)(a5 - 8);
  else
    v11 = (_BYTE **)(a5 - 24LL * (*(_DWORD *)(a5 + 20) & 0xFFFFFFF));
  v91 = *v11;
  v106 = sub_145CF80(*(_QWORD *)(a1 + 184), v102, 0, 0);
  if ( (*(_BYTE *)(a5 + 23) & 0x40) != 0 )
    v12 = *(_QWORD *)(a5 - 8);
  else
    v12 = a5 - 24LL * (*(_DWORD *)(a5 + 20) & 0xFFFFFFF);
  v13 = (__int64 *)(v12 + 24);
  v109 = 0;
  v110 = 0;
  v111 = 0;
  v14 = sub_16348C0(a5) | 4;
  v112 = 0;
  v113 = 0;
  v114 = 0;
  v115 = 0;
  v116 = 0;
  v117 = 0;
  if ( (*(_BYTE *)(a5 + 23) & 0x40) != 0 )
  {
    v15 = *(_QWORD *)(a5 - 8);
    v98 = (__int64 *)(v15 + 24LL * (*(_DWORD *)(a5 + 20) & 0xFFFFFFF));
  }
  else
  {
    v98 = (__int64 *)a5;
    v15 = a5 - 24LL * (*(_DWORD *)(a5 + 20) & 0xFFFFFFF);
  }
  v16 = (__int64 *)(v15 + 24);
  if ( v16 != v98 )
  {
    v104 = v13;
    v17 = 0;
    v18 = 0;
    v93 = v89 ? 4 : 0;
    while ( 1 )
    {
      v19 = *v16;
      v107 = *v16;
      if ( v18 == v17 )
      {
        sub_1287830((__int64)&v112, v18, &v107);
      }
      else
      {
        if ( v18 )
        {
          *(_QWORD *)v18 = v19;
          v18 = v113;
        }
        v113 = v18 + 8;
      }
      v20 = v14;
      v21 = v14 & 0xFFFFFFFFFFFFFFF8LL;
      v22 = v21;
      v23 = (v20 >> 2) & 1;
      if ( a6 == v106 )
      {
LABEL_47:
        v124 = 0;
        sub_1C56780((__int64)&v109, (char **)&v124);
        goto LABEL_31;
      }
      if ( (_BYTE)v23 )
      {
        v30 = v21;
        if ( v21 )
          goto LABEL_40;
      }
      else if ( v21 )
      {
        v24 = *(_QWORD **)(v107 + 24);
        if ( *(_DWORD *)(v107 + 32) > 0x40u )
          v24 = (_QWORD *)*v24;
        if ( (_DWORD)v24 )
          goto LABEL_68;
        v25 = sub_145D250(*(_QWORD *)(a1 + 184), v102, v21, 0);
        v26 = sub_13A5B00(*(_QWORD *)(a1 + 184), v106, v25, 0, 0);
        v124 = 0;
        v27 = v110;
        v106 = v26;
        if ( v110 == v111 )
        {
          sub_1C565F0((__int64)&v109, v110, &v124);
        }
        else
        {
          if ( v110 )
          {
            *(_QWORD *)v110 = 0;
            v27 = v110;
          }
          v110 = v27 + 8;
        }
        v124 = 0;
        v28 = v116;
        if ( v116 == v117 )
        {
          sub_1C565F0((__int64)&v115, v116, &v124);
        }
        else
        {
          if ( v116 )
          {
            *(_QWORD *)v116 = 0;
            v28 = v116;
          }
          v116 = v28 + 8;
        }
        goto LABEL_31;
      }
      v30 = sub_1643D30(0, *v104);
LABEL_40:
      v31 = sub_145D050(*(_QWORD *)(a1 + 184), v102, v30);
      v32 = *(_QWORD *)(a1 + 184);
      v108 = (char *)v31;
      v33 = sub_146F1B0(v32, v107);
      v34 = v116;
      v35 = v33;
      if ( v116 == v117 )
      {
        v96 = v33;
        sub_1C55B50((__int64)&v115, v116, &v108);
        v35 = v96;
      }
      else
      {
        if ( v116 )
        {
          *(_QWORD *)v116 = v108;
          v34 = v116;
        }
        v116 = v34 + 8;
      }
      v36 = sub_1483BD0(*(_QWORD **)(a1 + 184), v35, v102, a7, a8);
      v37 = sub_13A5B60(*(_QWORD *)(a1 + 184), v36, (__int64)v108, v93, 0);
      if ( *(_WORD *)(v37 + 24) )
      {
        sub_1C54F70((__int64 *)v121, v37, *(_QWORD *)(a1 + 184), a7, a8);
        if ( sub_14560B0((__int64)v121[0]) )
          goto LABEL_47;
        v95 = *(_QWORD *)(a1 + 184);
        v38 = sub_146F1B0(v95, v107);
        sub_1C54F70((__int64 *)&v124, v38, v95, a7, a8);
        v39 = v125;
        if ( !v125 )
          v39 = (char *)sub_145CF80(*(_QWORD *)(a1 + 184), v102, 0, 0);
        v118 = (__int64 **)v39;
        sub_1C56780((__int64)&v109, (char **)&v118);
        v40 = sub_1483BD0(*(_QWORD **)(a1 + 184), (__int64)v124, v102, a7, a8);
        v41 = (unsigned __int8 *)sub_13A5B60(*(_QWORD *)(a1 + 184), v40, (__int64)v108, v93, 0);
        if ( v41 != v121[0] )
          goto LABEL_68;
        v106 = sub_13A5B00(*(_QWORD *)(a1 + 184), v106, (__int64)v41, 0, 0);
      }
      else
      {
        v94 = v37;
        v106 = sub_13A5B00(*(_QWORD *)(a1 + 184), v106, v37, 0, 0);
        if ( sub_14560B0(v94) )
          goto LABEL_47;
        v124 = (__int64 **)sub_145CF80(*(_QWORD *)(a1 + 184), v102, 0, 0);
        sub_1C56780((__int64)&v109, (char **)&v124);
      }
LABEL_31:
      v16 += 3;
      if ( !(_BYTE)v23 || !v21 )
        v22 = sub_1643D30(v21, *v104);
      v29 = *(_BYTE *)(v22 + 8);
      if ( ((v29 - 14) & 0xFD) != 0 )
      {
        v14 = 0;
        if ( v29 == 13 )
          v14 = v22;
      }
      else
      {
        v14 = *(_QWORD *)(v22 + 24) | 4LL;
      }
      v104 += 3;
      if ( v16 == v98 )
        break;
      v18 = v113;
      v17 = v114;
    }
  }
  v107 = 0;
  if ( a6 == v106 )
  {
    v103 = 0;
    goto LABEL_67;
  }
  v103 = sub_14806B0(*(_QWORD *)(a1 + 184), a6, v106, 0, 0);
  v42 = sub_1649C60((__int64)v91);
  v43 = *(_BYTE *)(v42 + 16);
  if ( v43 <= 0x17u )
  {
    if ( v43 == 5 && *(_WORD *)(v42 + 18) == 32 )
    {
LABEL_61:
      v44 = sub_1C5BCD0(a1, a2, a3, *(_QWORD *)v91, v42, v103, (__int64)&v107);
      v45 = v103;
      if ( v44 )
      {
        if ( v103 == v107 && sub_14560B0(v106) )
          goto LABEL_68;
        v91 = (_BYTE *)v44;
        goto LABEL_77;
      }
      goto LABEL_96;
    }
  }
  else if ( v43 == 56 )
  {
    goto LABEL_61;
  }
  v45 = v103;
LABEL_96:
  v107 = v45;
LABEL_67:
  if ( !sub_14560B0(v106) )
  {
    if ( !v91 )
      goto LABEL_69;
LABEL_77:
    v47 = v109;
    v48 = (__int64)&v110[-v109] >> 3;
    if ( *(_BYTE *)(a5 + 16) != 56 )
    {
      v124 = 0;
      v125 = 0;
      v126 = 0;
      if ( !(_DWORD)v48 )
      {
        v55 = 0;
        v54 = 0;
LABEL_89:
        v56 = sub_16348C0(a5);
        BYTE4(v121[0]) = 0;
        v57 = (__int64 ***)sub_15A2E80(v56, (__int64)v91, v54, v55, v89 & 0x7F, (__int64)v121, 0);
        v91 = v57;
        *a9 = v107;
        if ( (__int64 **)a4 != *v57 )
          v91 = (_BYTE *)sub_15A4510(v57, (__int64 **)a4, 0);
        if ( v124 )
          j_j___libc_free_0(v124, v126 - (_QWORD)v124);
        goto LABEL_69;
      }
      v49 = 0;
      v50 = 8LL * (unsigned int)(v48 - 1);
      while ( 1 )
      {
        v53 = *(_QWORD *)(v47 + v49);
        if ( v53 )
        {
          v51 = (unsigned __int8 *)sub_38767A0(a3, v53, 0, a2);
          v52 = v125;
          v121[0] = v51;
          if ( v125 == (char *)v126 )
            goto LABEL_87;
        }
        else
        {
          v52 = v125;
          v51 = *(unsigned __int8 **)(v112 + v49);
          v121[0] = v51;
          if ( v125 == (char *)v126 )
          {
LABEL_87:
            sub_12F5DA0((__int64)&v124, v52, v121);
            if ( v50 == v49 )
              goto LABEL_88;
            goto LABEL_84;
          }
        }
        if ( v52 )
        {
          *(_QWORD *)v52 = v51;
          v52 = v125;
        }
        v125 = v52 + 8;
        if ( v50 == v49 )
        {
LABEL_88:
          v54 = v124;
          v55 = (v125 - (char *)v124) >> 3;
          goto LABEL_89;
        }
LABEL_84:
        v47 = v109;
        v49 += 8;
      }
    }
    v118 = 0;
    v119 = 0;
    v120 = 0;
    if ( !(_DWORD)v48 )
    {
      v97 = 0;
LABEL_146:
      if ( v106 == v97 && v107 == v103 )
        goto LABEL_126;
      v79 = sub_16498A0(a5);
      v80 = *(unsigned __int8 **)(a5 + 48);
      v124 = 0;
      v127 = v79;
      v81 = *(char **)(a5 + 40);
      v128 = 0;
      v125 = v81;
      v129 = 0;
      v130 = 0;
      v131 = 0;
      v126 = a5 + 24;
      v121[0] = v80;
      if ( v80 )
      {
        sub_1623A60((__int64)v121, (__int64)v80, 2);
        if ( v124 )
          sub_161E7C0((__int64)&v124, (__int64)v124);
        v124 = (__int64 **)v121[0];
        if ( v121[0] )
          sub_1623210((__int64)v121, v121[0], (__int64)&v124);
      }
      v121[0] = "newGep";
      v123 = 1;
      v122 = 3;
      v91 = (_BYTE *)sub_1BBF860(
                       (__int64 *)&v124,
                       *(_QWORD *)(a5 + 56),
                       v91,
                       v118,
                       (v119 - (_BYTE *)v118) >> 3,
                       (__int64 *)v121);
      if ( v97 )
      {
        if ( v107 )
          *a9 = sub_13A5B00(*(_QWORD *)(a1 + 184), v107, v97, 0, 0);
        else
          *a9 = v97;
      }
      else
      {
        *a9 = v107;
      }
      v82 = *(_QWORD *)v91;
      if ( a4 != *(_QWORD *)v91 )
      {
        if ( *(_BYTE *)(v82 + 8) == 16 )
          v82 = **(_QWORD **)(v82 + 16);
        v83 = *(_DWORD *)(v82 + 8) >> 8;
        v84 = a4;
        if ( *(_BYTE *)(a4 + 8) == 16 )
          v84 = **(_QWORD **)(a4 + 16);
        v85 = *(_DWORD *)(v84 + 8);
        v123 = 1;
        v122 = 3;
        v121[0] = "newBit";
        if ( v85 >> 8 == v83 )
          v91 = (_BYTE *)sub_12AA3B0((__int64 *)&v124, 0x2Fu, (__int64)v91, a4, (__int64)v121);
        else
          v91 = (_BYTE *)sub_12AA3B0((__int64 *)&v124, 0x30u, (__int64)v91, a4, (__int64)v121);
      }
      if ( v124 )
        sub_161E7C0((__int64)&v124, (__int64)v124);
LABEL_127:
      if ( v118 )
        j_j___libc_free_0(v118, v120 - (_BYTE *)v118);
      goto LABEL_69;
    }
    v58 = 0;
    v97 = 0;
    v105 = 8LL * (unsigned int)(v48 - 1);
    while ( 1 )
    {
      v60 = *(_QWORD *)(v47 + v58);
      if ( !v60 )
        break;
      v108 = 0;
      v61 = sub_14560B0(v60);
      v62 = v108;
      if ( !v61 )
      {
        v63 = sub_146F1B0(*(_QWORD *)(a1 + 184), *(_QWORD *)(v112 + v58));
        v101 = (char *)v63;
        if ( *(_WORD *)(v63 + 24) )
        {
          sub_1C54F70((__int64 *)&v124, v63, *(_QWORD *)(a1 + 184), a7, a8);
          v101 = (char *)v124;
        }
        v64 = *(_QWORD *)(v58 + v112);
        v99 = (char **)(v58 + v112);
        if ( *(_BYTE *)(v64 + 16) > 0x17u && *(_QWORD *)(a2 + 40) != *(_QWORD *)(v64 + 40) )
        {
          v65 = *(_QWORD *)(v64 + 8);
          if ( v65 )
          {
            if ( *(_QWORD *)(v65 + 8) )
            {
              v90 = v60;
              v66 = *(_QWORD *)(v64 + 8);
              v67 = *(_QWORD *)(v64 + 40);
              while ( 1 )
              {
                v68 = sub_1648700(v66);
                v69 = *((_BYTE *)v68 + 16);
                if ( v69 <= 0x17u )
                  BUG();
                if ( v69 == 77 )
                {
                  v70 = *((_DWORD *)v68 + 5) & 0xFFFFFFF;
                  if ( v70 )
                  {
                    v71 = 3LL * v70;
                    v72 = 8LL * v70;
                    v73 = &v68[-v71];
                    v74 = 0;
                    while ( 1 )
                    {
                      v75 = v73;
                      if ( (*((_BYTE *)v68 + 23) & 0x40) != 0 )
                        v75 = (_QWORD *)*(v68 - 1);
                      v76 = v75[3 * v74 / 8];
                      if ( v76 )
                      {
                        if ( v64 == v76 && v67 != v75[3 * *((unsigned int *)v68 + 14) + 1 + v74 / 8] )
                          break;
                      }
                      v74 += 8LL;
                      if ( v74 == v72 )
                        goto LABEL_130;
                    }
LABEL_126:
                    v91 = 0;
                    goto LABEL_127;
                  }
                }
                else if ( v67 != v68[5] )
                {
                  goto LABEL_126;
                }
LABEL_130:
                v66 = *(_QWORD *)(v66 + 8);
                if ( !v66 )
                {
                  v60 = v90;
                  break;
                }
              }
            }
          }
        }
        v121[0] = 0;
        v124 = (__int64 **)*v99;
        v125 = v101;
        v62 = (char *)sub_1C5B330(a1, (__int64 *)&v124, (__int64 *)v121, a7, a8);
        v108 = v62;
        if ( !v62 )
          goto LABEL_126;
        if ( v121[0] )
        {
          v77 = sub_13A5B60(*(_QWORD *)(a1 + 184), *(_QWORD *)(v115 + v58), (__int64)v121[0], 0, 0);
          if ( v97 )
            v97 = sub_13A5B00(*(_QWORD *)(a1 + 184), v97, v77, 0, 0);
          else
            v97 = v77;
          v62 = v108;
        }
      }
      if ( v62 )
      {
        v59 = v119;
        if ( v119 != v120 )
          goto LABEL_102;
        v78 = (__int64 *)&v108;
LABEL_143:
        sub_1287830((__int64)&v118, v59, v78);
        goto LABEL_105;
      }
      v62 = (char *)sub_38767A0(a3, v60, 0, a5);
      v59 = v119;
      v124 = (__int64 **)v62;
      if ( v119 != v120 )
      {
LABEL_102:
        if ( v59 )
        {
          *(_QWORD *)v59 = v62;
          v59 = v119;
        }
LABEL_104:
        v119 = v59 + 8;
        goto LABEL_105;
      }
      sub_12879C0((__int64)&v118, v119, &v124);
LABEL_105:
      if ( v105 == v58 )
        goto LABEL_146;
      v47 = v109;
      v58 += 8;
    }
    v59 = v119;
    v78 = (__int64 *)(v58 + v112);
    if ( v119 != v120 )
    {
      if ( v119 )
      {
        *(_QWORD *)v119 = *v78;
        v59 = v119;
      }
      goto LABEL_104;
    }
    goto LABEL_143;
  }
LABEL_68:
  v91 = 0;
LABEL_69:
  if ( v115 )
    j_j___libc_free_0(v115, &v117[-v115]);
  if ( v112 )
    j_j___libc_free_0(v112, &v114[-v112]);
  if ( v109 )
    j_j___libc_free_0(v109, &v111[-v109]);
  return v91;
}
