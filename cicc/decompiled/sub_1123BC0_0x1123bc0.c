// Function: sub_1123BC0
// Address: 0x1123bc0
//
__int64 __fastcall sub_1123BC0(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, _BYTE *a6)
{
  __int64 result; // rax
  unsigned __int8 *v11; // r10
  unsigned __int64 v12; // r11
  __int64 v13; // rcx
  unsigned int v14; // eax
  __int64 v15; // r14
  __int64 v16; // rdi
  unsigned int v17; // r15d
  int v18; // eax
  __int64 v19; // rax
  __int64 v20; // rsi
  __int64 v21; // rax
  __int64 v22; // r8
  unsigned int i; // r15d
  __int64 v24; // rcx
  unsigned __int64 v25; // rdx
  __int64 v26; // r9
  char v27; // cl
  __int64 v28; // rdx
  _BYTE *v29; // r10
  unsigned int v30; // r15d
  int v31; // r13d
  unsigned __int8 *v32; // r14
  int v33; // eax
  int v34; // eax
  unsigned __int8 *v35; // rax
  int v36; // ecx
  int v37; // eax
  int v38; // eax
  int v39; // eax
  int v40; // eax
  __int64 v41; // rbx
  bool v42; // al
  unsigned __int64 v43; // r11
  __int64 v44; // r10
  __int64 v45; // rax
  __int64 v46; // rdx
  __int64 v47; // rax
  __int64 v48; // rbx
  _BYTE *v49; // rax
  __int64 v50; // r13
  unsigned int **v51; // rdi
  __int64 v52; // r13
  _BYTE *v53; // rax
  unsigned int **v54; // rdi
  __int64 v55; // rax
  __int64 v56; // rax
  __int64 v57; // rbx
  _BYTE *v58; // rax
  _BYTE *v59; // r13
  unsigned int **v60; // rdi
  __int64 v61; // r13
  _BYTE *v62; // rax
  unsigned int **v63; // rdi
  __int64 v64; // rax
  unsigned __int64 v65; // rax
  __int64 v66; // rdx
  unsigned int **v67; // rdi
  __int64 v68; // rax
  __int64 v69; // rax
  __int64 v70; // rax
  __int64 v71; // r13
  __int64 v72; // rax
  __int64 v73; // r12
  __int64 v74; // rax
  __int64 v75; // r13
  __int64 v76; // rax
  __int64 v77; // r12
  __int64 v78; // rcx
  __int64 v79; // rdx
  __int64 v80; // rax
  _BYTE *v81; // rax
  unsigned int **v82; // rdi
  _BYTE *v83; // rax
  unsigned int **v84; // rdi
  __int64 v85; // r13
  __int64 v86; // rax
  unsigned int **v87; // rdi
  __int64 v88; // rax
  __int64 *v89; // r9
  __int64 v90; // rbx
  __int64 v91; // rax
  __int64 v92; // rax
  unsigned int **v93; // r12
  _BYTE *v94; // rbx
  _BYTE *v95; // rax
  __int64 v96; // r12
  __int64 v97; // rax
  __int64 v98; // r13
  __int64 v99; // r13
  unsigned __int64 v100; // [rsp+8h] [rbp-108h]
  unsigned __int64 v101; // [rsp+8h] [rbp-108h]
  __int64 v102; // [rsp+10h] [rbp-100h]
  __int64 v103; // [rsp+10h] [rbp-100h]
  __int64 v104; // [rsp+18h] [rbp-F8h]
  int v105; // [rsp+20h] [rbp-F0h]
  int v106; // [rsp+24h] [rbp-ECh]
  int v107; // [rsp+28h] [rbp-E8h]
  unsigned int v108; // [rsp+2Ch] [rbp-E4h]
  int v109; // [rsp+2Ch] [rbp-E4h]
  unsigned __int8 *v110; // [rsp+30h] [rbp-E0h]
  __int64 v111; // [rsp+30h] [rbp-E0h]
  __int64 v112; // [rsp+30h] [rbp-E0h]
  unsigned __int64 v113; // [rsp+30h] [rbp-E0h]
  unsigned __int64 v114; // [rsp+38h] [rbp-D8h]
  int v115; // [rsp+38h] [rbp-D8h]
  unsigned __int8 *v116; // [rsp+38h] [rbp-D8h]
  __int64 v117; // [rsp+38h] [rbp-D8h]
  __int64 v118; // [rsp+38h] [rbp-D8h]
  unsigned __int8 *v119; // [rsp+40h] [rbp-D0h]
  __int64 v120; // [rsp+40h] [rbp-D0h]
  __int64 v121; // [rsp+40h] [rbp-D0h]
  unsigned __int64 v122; // [rsp+40h] [rbp-D0h]
  char v123; // [rsp+40h] [rbp-D0h]
  unsigned int v124; // [rsp+40h] [rbp-D0h]
  unsigned __int64 v125; // [rsp+48h] [rbp-C8h]
  int v126; // [rsp+48h] [rbp-C8h]
  int v127; // [rsp+48h] [rbp-C8h]
  __int64 v128; // [rsp+50h] [rbp-C0h]
  int v129; // [rsp+50h] [rbp-C0h]
  __int64 v130; // [rsp+50h] [rbp-C0h]
  __int64 v131; // [rsp+50h] [rbp-C0h]
  __int64 v132; // [rsp+50h] [rbp-C0h]
  __int64 v133; // [rsp+50h] [rbp-C0h]
  __int64 *v134; // [rsp+50h] [rbp-C0h]
  __int64 v135; // [rsp+58h] [rbp-B8h] BYREF
  int v136; // [rsp+6Ch] [rbp-A4h] BYREF
  _QWORD v137[4]; // [rsp+70h] [rbp-A0h] BYREF
  unsigned int *v138; // [rsp+90h] [rbp-80h] BYREF
  __int64 v139; // [rsp+98h] [rbp-78h]
  _BYTE v140[16]; // [rsp+A0h] [rbp-70h] BYREF
  unsigned __int64 v141; // [rsp+B0h] [rbp-60h] BYREF
  __int64 v142; // [rsp+B8h] [rbp-58h]
  __int16 v143; // [rsp+D0h] [rbp-40h]

  v135 = a3;
  if ( (*(_BYTE *)(a2 + 2) & 1) != 0 )
    return 0;
  if ( *(_QWORD *)(v135 + 80) != *(_QWORD *)(a2 + 8) )
    return 0;
  if ( *(_QWORD *)(v135 + 72) != *(_QWORD *)(a4 + 24) )
    return 0;
  if ( (*(_BYTE *)(a4 + 80) & 1) == 0 )
    return 0;
  if ( sub_B2FC80(a4) )
    return 0;
  if ( (unsigned __int8)sub_B2F6B0(a4) )
    return 0;
  if ( (*(_BYTE *)(a4 + 80) & 2) != 0 )
    return 0;
  v11 = *(unsigned __int8 **)(a4 - 32);
  if ( *v11 != 15 && *v11 != 9 )
    return 0;
  v12 = *(_QWORD *)(*((_QWORD *)v11 + 1) + 32LL);
  if ( a1[2] < v12 )
    return 0;
  v13 = v135;
  v14 = *(_DWORD *)(v135 + 4) & 0x7FFFFFF;
  if ( v14 <= 2 )
    return 0;
  v15 = v14;
  v16 = *(_QWORD *)(v135 + 32 * (1LL - v14));
  if ( *(_BYTE *)v16 != 17 )
    return 0;
  v17 = *(_DWORD *)(v16 + 32);
  if ( v17 <= 0x40 )
  {
    if ( *(_QWORD *)(v16 + 24) )
      return 0;
  }
  else
  {
    v119 = v11;
    v125 = *(_QWORD *)(*((_QWORD *)v11 + 1) + 32LL);
    v18 = sub_C444A0(v16 + 24);
    v13 = v135;
    v12 = v125;
    v11 = v119;
    if ( v17 != v18 )
      return 0;
  }
  if ( **(_BYTE **)(v13 + 32 * (2 - v15)) <= 0x15u )
    return 0;
  v139 = 0x400000000LL;
  v19 = *((_QWORD *)v11 + 1);
  v138 = (unsigned int *)v140;
  v20 = *(_DWORD *)(v13 + 4) & 0x7FFFFFF;
  v21 = **(_QWORD **)(v19 + 16);
  if ( (_DWORD)v20 != 3 )
  {
    v22 = (unsigned int)(v20 - 1);
    for ( i = 3; ; ++i )
    {
      v24 = *(_QWORD *)(v13 + 32 * (i - v20));
      if ( *(_BYTE *)v24 != 17 )
        break;
      v25 = *(_QWORD *)(v24 + 24);
      if ( *(_DWORD *)(v24 + 32) > 0x40u )
        v25 = *(_QWORD *)v25;
      v26 = (unsigned int)v25;
      if ( v25 != (unsigned int)v25 )
        break;
      v27 = *(_BYTE *)(v21 + 8);
      if ( v27 == 15 )
      {
        v21 = *(_QWORD *)(*(_QWORD *)(v21 + 16) + 8 * v25);
      }
      else
      {
        if ( v27 != 16 || *(_QWORD *)(v21 + 32) <= v25 )
          break;
        v21 = *(_QWORD *)(v21 + 24);
      }
      v28 = (unsigned int)v139;
      if ( (unsigned __int64)(unsigned int)v139 + 1 > HIDWORD(v139) )
      {
        v108 = v22;
        v110 = v11;
        v114 = v12;
        v120 = v21;
        v126 = v26;
        sub_C8D5F0((__int64)&v138, v140, (unsigned int)v139 + 1LL, 4u, v22, v26);
        v28 = (unsigned int)v139;
        v22 = v108;
        v11 = v110;
        v12 = v114;
        v21 = v120;
        LODWORD(v26) = v126;
      }
      v138[v28] = v26;
      LODWORD(v139) = v139 + 1;
      if ( (_DWORD)v22 == i )
        goto LABEL_39;
      v13 = v135;
      v20 = *(_DWORD *)(v135 + 4) & 0x7FFFFFF;
    }
LABEL_31:
    result = 0;
    goto LABEL_32;
  }
LABEL_39:
  v115 = v12;
  v111 = *(_QWORD *)(a5 - 32);
  if ( (_DWORD)v12 )
  {
    v30 = 0;
    v121 = a5;
    v31 = -1;
    v32 = v11;
    v104 = 0;
    v127 = -2;
    v129 = -2;
    v107 = -2;
    v105 = -2;
    v109 = -2;
    v106 = -2;
    v100 = v12;
    while ( 1 )
    {
      v20 = v30;
      v29 = (_BYTE *)sub_AD69F0(v32, v30);
      if ( !v29 )
        goto LABEL_31;
      if ( (_DWORD)v139 )
      {
        v20 = (__int64)v138;
        v29 = (_BYTE *)sub_AAADB0((__int64)v29, v138, (unsigned int)v139);
        if ( !v29 )
          goto LABEL_31;
      }
      if ( a6 )
      {
        v20 = (__int64)v29;
        v29 = (_BYTE *)sub_96E6C0(0x1Cu, (__int64)v29, a6, a1[11]);
        if ( !v29 )
          goto LABEL_31;
      }
      v20 = (__int64)v29;
      v35 = (unsigned __int8 *)sub_9719A0(*(_WORD *)(v121 + 2) & 0x3F, v29, v111, a1[11], a1[9], 0);
      if ( !v35 )
        goto LABEL_31;
      v36 = *v35;
      if ( (unsigned int)(v36 - 12) <= 1 )
      {
        v33 = v129;
        if ( v129 == v31 )
          v33 = v30;
        v129 = v33;
        v34 = v127;
        if ( v127 == v31 )
          v34 = v30;
        v127 = v34;
      }
      else
      {
        if ( (_BYTE)v36 != 17 )
          goto LABEL_31;
        if ( sub_9867B0((__int64)(v35 + 24)) )
        {
          if ( v105 == -2 )
          {
            v127 = v30;
            v105 = v30;
          }
          else
          {
            v39 = -3;
            if ( v107 == -2 )
              v39 = v30;
            v107 = v39;
            v40 = -3;
            if ( v127 == v31 )
              v40 = v30;
            v127 = v40;
          }
LABEL_69:
          if ( (v30 & 8) == 0 && v30 > 0x3F && v109 == -3 )
          {
            if ( v107 == -3 && v129 == -3 )
            {
              if ( v127 == -3 )
                goto LABEL_31;
              v129 = -3;
              v107 = -3;
              v109 = -3;
            }
            else
            {
              v109 = -3;
            }
          }
          goto LABEL_46;
        }
        if ( v106 == -2 )
        {
          v129 = v30;
          v106 = v30;
        }
        else
        {
          v37 = -3;
          if ( v109 == -2 )
            v37 = v30;
          v109 = v37;
          v38 = -3;
          if ( v129 == v31 )
            v38 = v30;
          v129 = v38;
        }
        if ( v30 > 0x3F )
          goto LABEL_69;
        v104 |= 1LL << v30;
      }
LABEL_46:
      ++v30;
      ++v31;
      if ( v115 == v30 )
      {
        v11 = v32;
        v12 = v100;
        a5 = v121;
        goto LABEL_80;
      }
    }
  }
  v104 = 0;
  v127 = -2;
  v129 = -2;
  v107 = -2;
  v105 = -2;
  v109 = -2;
  v106 = -2;
LABEL_80:
  v116 = v11;
  v122 = v12;
  v41 = *(_QWORD *)(v135 + 32 * (2LL - (*(_DWORD *)(v135 + 4) & 0x7FFFFFF)));
  v42 = sub_B4DE30(v135);
  v43 = v122;
  v44 = (__int64)v116;
  if ( !v42 )
  {
    v103 = (__int64)v116;
    v113 = v122;
    v118 = sub_AE4570(a1[11], *(_QWORD *)(v135 + 8));
    v124 = *(_DWORD *)(v118 + 8) >> 8;
    v65 = sub_BCAE30(*(_QWORD *)(v41 + 8));
    v43 = v113;
    v141 = v65;
    v142 = v66;
    v44 = v103;
    if ( v124 < v65 )
    {
      v67 = (unsigned int **)a1[4];
      v143 = 257;
      v68 = sub_A82DA0(v67, v41, v118, (__int64)&v141, 0, 0);
      v44 = v103;
      v43 = v113;
      v41 = v68;
    }
  }
  v101 = v43;
  v102 = v44;
  v117 = a1[11];
  v112 = **(_QWORD **)(*(_QWORD *)(v44 + 8) + 16LL);
  v123 = sub_AE5020(v117, v112);
  v45 = sub_9208B0(v117, v112);
  v142 = v46;
  v141 = (((unsigned __int64)(v45 + 7) >> 3) + (1LL << v123) - 1) >> v123 << v123;
  v137[2] = a1;
  v136 = sub_CA1930(&v141);
  v137[0] = &v135;
  v137[1] = &v136;
  if ( v109 != -3 )
  {
    v47 = sub_1113590((__int64)v137, v41);
    v48 = v47;
    if ( v106 == -2 )
    {
      v69 = sub_ACD720(*(__int64 **)(a1[4] + 72LL));
      v20 = a5;
      result = (__int64)sub_F162A0((__int64)a1, a5, v69);
    }
    else
    {
      v49 = (_BYTE *)sub_AD64C0(*(_QWORD *)(v47 + 8), v106, 0);
      v50 = (__int64)v49;
      if ( v109 == -2 )
      {
        v143 = 257;
        v20 = unk_3F10FD0;
        result = (__int64)sub_BD2C40(72, unk_3F10FD0);
        if ( result )
        {
          v20 = 32;
          v130 = result;
          sub_1113300(result, 32, v48, v50, (__int64)&v141);
          result = v130;
        }
      }
      else
      {
        v51 = (unsigned int **)a1[4];
        v143 = 257;
        v52 = sub_92B530(v51, 0x20u, v48, v49, (__int64)&v141);
        v53 = (_BYTE *)sub_AD64C0(*(_QWORD *)(v48 + 8), v109, 0);
        v54 = (unsigned int **)a1[4];
        v143 = 257;
        v55 = sub_92B530(v54, 0x20u, v48, v53, (__int64)&v141);
        v20 = v52;
        v143 = 257;
        result = sub_B504D0(29, v52, v55, (__int64)&v141, 0, 0);
      }
    }
    goto LABEL_32;
  }
  if ( v107 == -3 )
  {
    if ( v129 != -3 )
    {
      v70 = sub_1113590((__int64)v137, v41);
      v71 = v70;
      if ( v106 )
      {
        v81 = (_BYTE *)sub_AD64C0(*(_QWORD *)(v70 + 8), -v106, 0);
        v82 = (unsigned int **)a1[4];
        v143 = 257;
        v71 = sub_929C50(v82, (_BYTE *)v71, v81, (__int64)&v141, 0, 0);
      }
      v72 = sub_AD64C0(*(_QWORD *)(v71 + 8), v129 - v106 + 1, 0);
      v143 = 257;
      v73 = v72;
      v20 = unk_3F10FD0;
      result = (__int64)sub_BD2C40(72, unk_3F10FD0);
      if ( result )
      {
        v20 = 36;
        v131 = result;
        sub_1113300(result, 36, v71, v73, (__int64)&v141);
        result = v131;
      }
      goto LABEL_32;
    }
    if ( v127 != -3 )
    {
      v74 = sub_1113590((__int64)v137, v41);
      v75 = v74;
      if ( v105 )
      {
        v83 = (_BYTE *)sub_AD64C0(*(_QWORD *)(v74 + 8), -v105, 0);
        v84 = (unsigned int **)a1[4];
        v143 = 257;
        v75 = sub_929C50(v84, (_BYTE *)v75, v83, (__int64)&v141, 0, 0);
      }
      v76 = sub_AD64C0(*(_QWORD *)(v75 + 8), v127 - v105, 0);
      v143 = 257;
      v77 = v76;
      v20 = unk_3F10FD0;
      result = (__int64)sub_BD2C40(72, unk_3F10FD0);
      if ( result )
      {
        v20 = 34;
        v132 = result;
        sub_1113300(result, 34, v75, v77, (__int64)&v141);
        result = v132;
      }
      goto LABEL_32;
    }
    v85 = *(_QWORD *)(v41 + 8);
    if ( *(_DWORD *)(v85 + 8) >> 8 < v101 )
    {
      v99 = a1[11];
      v20 = sub_BD5C60(v102);
      v85 = sub_AE44B0(v99, v20, v101);
      if ( !v85 )
        goto LABEL_31;
    }
    v86 = sub_1113590((__int64)v137, v41);
    v87 = (unsigned int **)a1[4];
    v143 = 257;
    v88 = sub_921630(v87, v86, v85, 0, (__int64)&v141);
    v89 = (__int64 *)a1[4];
    v90 = v88;
    v143 = 257;
    v134 = v89;
    v91 = sub_AD64C0(v85, v104, 0);
    v92 = sub_F94560(v134, v91, v90, (__int64)&v141, 0);
    v93 = (unsigned int **)a1[4];
    v94 = (_BYTE *)v92;
    v143 = 257;
    v95 = (_BYTE *)sub_AD64C0(v85, 1, 0);
    v96 = sub_A82350(v93, v95, v94, (__int64)&v141);
    v97 = sub_AD64C0(v85, 0, 0);
    v143 = 257;
    v98 = v97;
    v20 = unk_3F10FD0;
    result = (__int64)sub_BD2C40(72, unk_3F10FD0);
    if ( result )
    {
      v78 = v98;
      v79 = v96;
      goto LABEL_107;
    }
  }
  else
  {
    v56 = sub_1113590((__int64)v137, v41);
    v57 = v56;
    if ( v105 == -2 )
    {
      v80 = sub_ACD6D0(*(__int64 **)(a1[4] + 72LL));
      v20 = a5;
      result = (__int64)sub_F162A0((__int64)a1, a5, v80);
    }
    else
    {
      v58 = (_BYTE *)sub_AD64C0(*(_QWORD *)(v56 + 8), v105, 0);
      v59 = v58;
      if ( v107 != -2 )
      {
        v60 = (unsigned int **)a1[4];
        v143 = 257;
        v61 = sub_92B530(v60, 0x21u, v57, v58, (__int64)&v141);
        v62 = (_BYTE *)sub_AD64C0(*(_QWORD *)(v57 + 8), v107, 0);
        v63 = (unsigned int **)a1[4];
        v143 = 257;
        v64 = sub_92B530(v63, 0x21u, v57, v62, (__int64)&v141);
        v20 = v61;
        v143 = 257;
        result = sub_B504D0(28, v61, v64, (__int64)&v141, 0, 0);
        goto LABEL_32;
      }
      v143 = 257;
      v20 = unk_3F10FD0;
      result = (__int64)sub_BD2C40(72, unk_3F10FD0);
      if ( result )
      {
        v78 = (__int64)v59;
        v79 = v57;
LABEL_107:
        v20 = 33;
        v133 = result;
        sub_1113300(result, 33, v79, v78, (__int64)&v141);
        result = v133;
      }
    }
  }
LABEL_32:
  if ( v138 != (unsigned int *)v140 )
  {
    v128 = result;
    _libc_free(v138, v20);
    return v128;
  }
  return result;
}
