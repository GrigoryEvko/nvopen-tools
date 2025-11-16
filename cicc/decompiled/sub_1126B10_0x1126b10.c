// Function: sub_1126B10
// Address: 0x1126b10
//
unsigned __int8 *__fastcall sub_1126B10(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // r13
  unsigned int v7; // r14d
  unsigned int v8; // r15d
  __int64 v10; // r14
  _QWORD *v11; // rax
  _QWORD *v12; // r12
  __int64 v13; // r15
  unsigned int v14; // r14d
  __int64 v15; // rdi
  __int64 v16; // r14
  unsigned int v17; // r15d
  unsigned int v18; // r15d
  bool v19; // al
  __int64 v20; // rdx
  _BYTE *v21; // rax
  __int64 v23; // rdx
  _BYTE *v24; // rax
  unsigned __int64 v25; // rax
  bool v26; // al
  __int64 *v27; // r14
  unsigned int v28; // eax
  bool v29; // al
  unsigned int v30; // r14d
  bool v31; // al
  unsigned int v32; // eax
  unsigned int v33; // eax
  __int64 v34; // r15
  _QWORD *v35; // rax
  __int64 v36; // r15
  __int64 v37; // rdx
  __int16 v38; // dx
  unsigned int v39; // eax
  unsigned __int64 v40; // rsi
  __int64 v41; // rdx
  unsigned __int64 v42; // rax
  unsigned __int64 v43; // rcx
  bool v44; // zf
  unsigned int v45; // eax
  __int64 v46; // rax
  __int64 v47; // r14
  _QWORD *v48; // rax
  __int64 v49; // rax
  unsigned int v50; // r14d
  unsigned int v51; // esi
  _BYTE *v52; // r14
  const char *v53; // rax
  __int64 v54; // rdx
  __int64 v55; // r13
  __int64 v56; // r15
  _QWORD *v57; // rax
  __int64 v58; // rdi
  __int16 v59; // r15
  int v60; // r12d
  int v61; // eax
  __int64 v62; // r13
  __int64 v63; // r14
  _QWORD *v64; // rax
  __int64 v65; // r14
  __int64 v66; // rax
  __int64 v67; // r13
  _QWORD *v68; // rax
  unsigned int v69; // eax
  unsigned int v70; // eax
  unsigned int v71; // eax
  __int64 v72; // r14
  _QWORD *v73; // rax
  unsigned int v74; // eax
  __int64 v75; // r15
  _QWORD *v76; // rax
  unsigned int v77; // eax
  int v78; // r14d
  __int64 v79; // rdi
  int v80; // eax
  unsigned int v81; // eax
  unsigned int v82; // eax
  __int64 v83; // rax
  __int64 v84; // r14
  _QWORD *v85; // rax
  unsigned int v86; // eax
  __int64 v87; // rdx
  __int64 v88; // rcx
  unsigned int v89; // r8d
  unsigned int v90; // eax
  __int64 v91; // rax
  __int64 v92; // r15
  _QWORD *v93; // rax
  unsigned __int64 v94; // rcx
  int v95; // edx
  int v96; // ecx
  unsigned int v97; // eax
  __int64 v98; // rdx
  __int64 v99; // rcx
  unsigned int v100; // r8d
  __int64 v101; // rdx
  __int64 v102; // rcx
  __int64 v103; // rax
  __int64 v104; // r15
  _QWORD *v105; // rax
  __int64 v106; // rax
  __int64 v107; // r14
  _QWORD *v108; // rax
  char v109; // [rsp+20h] [rbp-140h]
  char v110; // [rsp+28h] [rbp-138h]
  char v111; // [rsp+28h] [rbp-138h]
  bool v112; // [rsp+40h] [rbp-120h]
  unsigned __int64 v113; // [rsp+50h] [rbp-110h]
  __int64 v114; // [rsp+50h] [rbp-110h]
  char v116; // [rsp+60h] [rbp-100h]
  bool v117; // [rsp+60h] [rbp-100h]
  char v118; // [rsp+60h] [rbp-100h]
  unsigned int v119; // [rsp+60h] [rbp-100h]
  unsigned int v120; // [rsp+68h] [rbp-F8h]
  int v121; // [rsp+6Ch] [rbp-F4h]
  __int16 v122; // [rsp+70h] [rbp-F0h]
  unsigned int **v124; // [rsp+78h] [rbp-E8h]
  unsigned __int64 v125; // [rsp+80h] [rbp-E0h] BYREF
  int v126; // [rsp+88h] [rbp-D8h]
  __int64 v127; // [rsp+90h] [rbp-D0h] BYREF
  int v128; // [rsp+98h] [rbp-C8h]
  __int64 v129; // [rsp+A0h] [rbp-C0h] BYREF
  int v130; // [rsp+A8h] [rbp-B8h]
  unsigned __int64 v131; // [rsp+B0h] [rbp-B0h] BYREF
  int v132; // [rsp+B8h] [rbp-A8h]
  unsigned __int64 v133; // [rsp+C0h] [rbp-A0h] BYREF
  unsigned int v134; // [rsp+C8h] [rbp-98h]
  unsigned __int64 v135; // [rsp+D0h] [rbp-90h] BYREF
  unsigned int v136; // [rsp+D8h] [rbp-88h]
  unsigned __int64 v137; // [rsp+E0h] [rbp-80h] BYREF
  unsigned int v138; // [rsp+E8h] [rbp-78h]
  unsigned __int64 v139; // [rsp+F0h] [rbp-70h] BYREF
  unsigned int v140; // [rsp+F8h] [rbp-68h]
  unsigned __int64 v141; // [rsp+100h] [rbp-60h] BYREF
  __int64 v142; // [rsp+108h] [rbp-58h]
  char *v143; // [rsp+110h] [rbp-50h]
  __int16 v144; // [rsp+120h] [rbp-40h]

  v6 = *(_QWORD *)(a3 - 64);
  v7 = (*(_WORD *)(a2 + 2) & 0x3F) - 32;
  v121 = *(_WORD *)(a2 + 2) & 0x3F;
  v122 = *(_WORD *)(a2 + 2) & 0x3F;
  if ( v7 <= 1 && sub_B44E60(a3) )
  {
    v8 = *(_DWORD *)(a4 + 8);
    if ( v8 <= 0x40 ? *(_QWORD *)a4 == 0 : v8 == (unsigned int)sub_C444A0(a4) )
    {
      v10 = *(_QWORD *)(a2 - 32);
      v144 = 257;
      v11 = sub_BD2C40(72, unk_3F10FD0);
      v12 = v11;
      if ( v11 )
        sub_1113300((__int64)v11, v121, v6, v10, (__int64)&v141);
      return (unsigned __int8 *)v12;
    }
  }
  v13 = v6 + 24;
  v116 = *(_BYTE *)a3;
  if ( *(_BYTE *)v6 != 17 )
  {
    v20 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v6 + 8) + 8LL) - 17;
    if ( (unsigned int)v20 > 1 )
      goto LABEL_14;
    if ( *(_BYTE *)v6 > 0x15u )
      goto LABEL_14;
    v21 = sub_AD7630(v6, 0, v20);
    if ( !v21 || *v21 != 17 )
      goto LABEL_14;
    v13 = (__int64)(v21 + 24);
    v7 = (*(_WORD *)(a2 + 2) & 0x3F) - 32;
  }
  if ( v7 > 1 )
  {
    if ( v116 == 56 )
      goto LABEL_14;
    v14 = *(_DWORD *)(v13 + 8);
    if ( sub_986C60((__int64 *)v13, v14 - 1) )
    {
      if ( sub_9893F0(v121, a4, &v137) )
      {
        v65 = *(_QWORD *)(a3 - 32);
        v66 = sub_AD6530(*(_QWORD *)(v6 + 8), a4);
        v144 = 257;
        v67 = v66;
        v68 = sub_BD2C40(72, unk_3F10FD0);
        v12 = v68;
        if ( v68 )
          sub_1113300((__int64)v68, ((_BYTE)v137 == 0) + 32, v65, v67, (__int64)&v141);
        return (unsigned __int8 *)v12;
      }
      v14 = *(_DWORD *)(v13 + 8);
    }
    if ( v14 > 0x40 )
    {
      if ( (unsigned int)sub_C44630(v13) != 1 )
        goto LABEL_14;
    }
    else if ( !*(_QWORD *)v13 || (*(_QWORD *)v13 & (*(_QWORD *)v13 - 1LL)) != 0 )
    {
      goto LABEL_14;
    }
    if ( ((v122 - 34) & 0xFFFD) == 0 )
    {
      if ( v122 == 34 )
      {
        v58 = v13;
        v59 = 36;
        v60 = sub_9871A0(a4);
        v61 = sub_9871A0(v58);
        v62 = sub_AD64C0(*(_QWORD *)(a3 + 8), (unsigned int)(v60 - v61), 0);
      }
      else
      {
        sub_9865C0((__int64)&v139, a4);
        sub_C46F20((__int64)&v139, 1u);
        v77 = v140;
        v140 = 0;
        LODWORD(v142) = v77;
        v141 = v139;
        v78 = sub_9871A0((__int64)&v141);
        sub_969240((__int64 *)&v141);
        sub_969240((__int64 *)&v139);
        v79 = v13;
        v59 = 35;
        v80 = sub_9871A0(v79);
        v62 = sub_AD64C0(*(_QWORD *)(a3 + 8), (unsigned int)(v78 - v80), 0);
      }
      v63 = *(_QWORD *)(a3 - 32);
      v144 = 257;
      v64 = sub_BD2C40(72, unk_3F10FD0);
      v12 = v64;
      if ( v64 )
        sub_1113300((__int64)v64, v59, v63, v62, (__int64)&v141);
      return (unsigned __int8 *)v12;
    }
LABEL_14:
    v15 = *(_QWORD *)(a3 - 32);
    v16 = v15 + 24;
    if ( *(_BYTE *)v15 != 17 )
    {
      v23 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v15 + 8) + 8LL) - 17;
      if ( (unsigned int)v23 > 1 )
        return 0;
      if ( *(_BYTE *)v15 > 0x15u )
        return 0;
      v24 = sub_AD7630(v15, 0, v23);
      if ( !v24 || *v24 != 17 )
        return 0;
      v16 = (__int64)(v24 + 24);
    }
    v17 = *(_DWORD *)(v16 + 8);
    v120 = *(_DWORD *)(a4 + 8);
    if ( v17 > 0x40 )
    {
      v113 = *(unsigned int *)(a4 + 8);
      if ( v17 - (unsigned int)sub_C444A0(v16) > 0x40 )
        return 0;
      v25 = **(_QWORD **)v16;
      if ( v113 < v25 )
        return 0;
      v18 = **(_QWORD **)v16;
      v19 = v120 <= (unsigned int)v25 || (_DWORD)v25 == 0;
    }
    else
    {
      if ( (unsigned __int64)*(unsigned int *)(a4 + 8) < *(_QWORD *)v16 )
        return 0;
      v18 = *(_QWORD *)v16;
      v19 = v120 <= v18 || v18 == 0;
    }
    if ( v19 )
      return 0;
    v26 = sub_B44E60(a3);
    v114 = *(_QWORD *)(a3 + 8);
    if ( v116 != 56 )
    {
      if ( v122 != 36 && (v121 != 34 || !v26) )
      {
LABEL_47:
        if ( v122 == 34 )
        {
          sub_9865C0((__int64)&v137, a4);
          sub_C46A40((__int64)&v137, 1);
          v69 = v138;
          v138 = 0;
          v140 = v69;
          v139 = v137;
          sub_9865C0((__int64)&v141, (__int64)&v139);
          sub_1110B10((__int64)&v141, v18);
          sub_C46F20((__int64)&v141, 1u);
          v132 = v142;
          v131 = v141;
          sub_969240((__int64 *)&v139);
          sub_969240((__int64 *)&v137);
          sub_9865C0((__int64)&v139, a4);
          sub_C46A40((__int64)&v139, 1);
          v70 = v140;
          v140 = 0;
          LODWORD(v142) = v70;
          v141 = v139;
          sub_9865C0((__int64)&v133, (__int64)&v131);
          sub_C46A40((__int64)&v133, 1);
          v71 = v134;
          v134 = 0;
          v136 = v71;
          v135 = v133;
          sub_9865C0((__int64)&v137, (__int64)&v135);
          if ( v138 > 0x40 )
          {
            sub_C482E0((__int64)&v137, v18);
          }
          else if ( v18 == v138 )
          {
            v137 = 0;
          }
          else
          {
            v137 >>= v18;
          }
          v110 = sub_AAD8B0((__int64)&v137, &v141);
          sub_969240((__int64 *)&v137);
          sub_969240((__int64 *)&v135);
          sub_969240((__int64 *)&v133);
          sub_969240((__int64 *)&v141);
          sub_969240((__int64 *)&v139);
          if ( v110 )
          {
            v72 = sub_AD8D80(v114, (__int64)&v131);
            v144 = 257;
            v73 = sub_BD2C40(72, unk_3F10FD0);
            v12 = v73;
            if ( v73 )
              sub_1113300((__int64)v73, 34, v6, v72, (__int64)&v141);
            sub_969240((__int64 *)&v131);
            return (unsigned __int8 *)v12;
          }
          sub_969240((__int64 *)&v131);
        }
LABEL_48:
        if ( (*(_WORD *)(a2 + 2) & 0x3Fu) - 32 <= 1 )
        {
          if ( !sub_B44E60(a3) )
          {
            v30 = *(_DWORD *)(a4 + 8);
            if ( v30 <= 0x40 )
              v31 = *(_QWORD *)a4 == 0;
            else
              v31 = v30 == (unsigned int)sub_C444A0(a4);
            if ( v31 )
            {
              if ( v122 != 32 )
              {
                sub_9865C0((__int64)&v133, a4);
                sub_C46A40((__int64)&v133, 1);
                v32 = v134;
                v134 = 0;
                v136 = v32;
                v135 = v133;
                sub_9865C0((__int64)&v137, (__int64)&v135);
                sub_1110B10((__int64)&v137, v18);
                sub_C46F20((__int64)&v137, 1u);
                v33 = v138;
                v138 = 0;
                v140 = v33;
                v139 = v137;
                v34 = sub_AD8D80(v114, (__int64)&v139);
                v144 = 257;
                v35 = sub_BD2C40(72, unk_3F10FD0);
                v12 = v35;
                if ( v35 )
                  sub_1113300((__int64)v35, 34, v6, v34, (__int64)&v141);
                sub_969240((__int64 *)&v139);
                sub_969240((__int64 *)&v137);
                sub_969240((__int64 *)&v135);
                sub_969240((__int64 *)&v133);
                return (unsigned __int8 *)v12;
              }
              sub_9865C0((__int64)&v135, a4);
              sub_C46A40((__int64)&v135, 1);
              v74 = v136;
              v136 = 0;
              v138 = v74;
              v137 = v135;
              sub_9865C0((__int64)&v139, (__int64)&v137);
              sub_1110B10((__int64)&v139, v18);
              v75 = sub_AD8D80(v114, (__int64)&v139);
              v144 = 257;
              v76 = sub_BD2C40(72, unk_3F10FD0);
              v12 = v76;
              if ( v76 )
                sub_1113300((__int64)v76, 36, v6, v75, (__int64)&v141);
              sub_969240((__int64 *)&v139);
              sub_969240((__int64 *)&v137);
              goto LABEL_119;
            }
            v49 = *(_QWORD *)(a3 + 16);
            if ( v49 && !*(_QWORD *)(v49 + 8) )
            {
              v138 = v120;
              v50 = v18 - v120;
              if ( v120 > 0x40 )
              {
                sub_C43690((__int64)&v137, 0, 0);
                v120 = v138;
                v51 = v138 + v50;
              }
              else
              {
                v137 = 0;
                v51 = v18;
              }
              if ( v120 != v51 )
              {
                if ( v51 > 0x3F || v120 > 0x40 )
                  sub_C43C90(&v137, v51, v120);
                else
                  v137 |= 0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v50 + 64) << v51;
              }
              v52 = (_BYTE *)sub_AD8D80(v114, (__int64)&v137);
              v124 = *(unsigned int ***)(a1 + 32);
              v53 = sub_BD5D20(a3);
              v144 = 773;
              v141 = (unsigned __int64)v53;
              v142 = v54;
              v143 = ".mask";
              v55 = sub_A82350(v124, (_BYTE *)v6, v52, (__int64)&v141);
              sub_9865C0((__int64)&v139, a4);
              sub_1110B10((__int64)&v139, v18);
              v56 = sub_AD8D80(v114, (__int64)&v139);
              v144 = 257;
              v57 = sub_BD2C40(72, unk_3F10FD0);
              v12 = v57;
              if ( v57 )
                sub_1113300((__int64)v57, v121, v55, v56, (__int64)&v141);
              sub_969240((__int64 *)&v139);
              sub_969240((__int64 *)&v137);
              return (unsigned __int8 *)v12;
            }
            return 0;
          }
          v27 = (__int64 *)&v139;
          sub_9865C0((__int64)&v139, a4);
          sub_1110B10((__int64)&v139, v18);
LABEL_60:
          v36 = sub_AD8D80(v114, (__int64)&v139);
          v144 = 257;
          v12 = sub_BD2C40(72, unk_3F10FD0);
          if ( !v12 )
          {
LABEL_62:
            sub_969240(v27);
            return (unsigned __int8 *)v12;
          }
LABEL_61:
          sub_1113300((__int64)v12, v121, v6, v36, (__int64)&v141);
          goto LABEL_62;
        }
        return 0;
      }
      v140 = v120;
      v27 = (__int64 *)&v139;
      if ( v120 > 0x40 )
        sub_C43780((__int64)&v139, (const void **)a4);
      else
        v139 = *(_QWORD *)a4;
      sub_1110B10((__int64)&v139, v18);
      v28 = v140;
      LODWORD(v142) = v140;
      if ( v140 > 0x40 )
      {
        sub_C43780((__int64)&v141, (const void **)&v139);
        v28 = v142;
        if ( (unsigned int)v142 > 0x40 )
        {
          sub_C482E0((__int64)&v141, v18);
          if ( (unsigned int)v142 > 0x40 )
          {
            v29 = sub_C43C50((__int64)&v141, (const void **)a4);
            if ( v141 )
            {
              v117 = v29;
              j_j___libc_free_0_0(v141);
              v29 = v117;
            }
            goto LABEL_45;
          }
LABEL_44:
          v29 = v141 == *(_QWORD *)a4;
LABEL_45:
          if ( v29 )
            goto LABEL_60;
          sub_969240((__int64 *)&v139);
          goto LABEL_47;
        }
      }
      else
      {
        v141 = v139;
      }
      if ( v18 == v28 )
        v141 = 0;
      else
        v141 >>= v18;
      goto LABEL_44;
    }
    v37 = *(_QWORD *)(a3 + 16);
    if ( !v37 || *(_QWORD *)(v37 + 8) )
      goto LABEL_48;
    v38 = (v122 - 36) & 0xFFFB;
    if ( v26 )
    {
      if ( !v38 )
      {
        v27 = (__int64 *)&v139;
        sub_9865C0((__int64)&v139, a4);
        sub_C46F20((__int64)&v139, 1u);
        v81 = v140;
        v140 = 0;
        LODWORD(v142) = v81;
        v141 = v139;
        if ( sub_986BA0((__int64)&v141) )
        {
          v119 = sub_9871A0(a4);
          sub_969240((__int64 *)&v141);
          sub_969240((__int64 *)&v139);
          if ( v119 > v18 )
          {
            sub_9865C0((__int64)&v137, a4);
            sub_C46F20((__int64)&v137, 1u);
            v82 = v138;
            v138 = 0;
            v140 = v82;
            v139 = v137;
            sub_9865C0((__int64)&v141, (__int64)&v139);
            sub_1110B10((__int64)&v141, v18);
            sub_C46A40((__int64)&v141, 1);
            v136 = v142;
            v135 = v141;
            sub_969240((__int64 *)&v139);
            sub_969240((__int64 *)&v137);
            v83 = sub_AD8D80(v114, (__int64)&v135);
            v144 = 257;
            v84 = v83;
            v85 = sub_BD2C40(72, unk_3F10FD0);
            v12 = v85;
            if ( v85 )
              sub_1113300((__int64)v85, v121, v6, v84, (__int64)&v141);
LABEL_119:
            sub_969240((__int64 *)&v135);
            return (unsigned __int8 *)v12;
          }
        }
        else
        {
          sub_969240((__int64 *)&v141);
          sub_969240((__int64 *)&v139);
        }
        v112 = v121 == 36;
        goto LABEL_126;
      }
      v112 = v121 == 36;
    }
    else
    {
      v112 = v121 == 36;
      if ( v38 )
        goto LABEL_67;
    }
    v27 = (__int64 *)&v139;
LABEL_126:
    sub_9865C0((__int64)&v139, a4);
    sub_1110B10((__int64)&v139, v18);
    sub_9865C0((__int64)&v141, (__int64)&v139);
    sub_D94900((__int64)&v141, v18);
    v118 = sub_AAD8B0((__int64)&v141, (_QWORD *)a4);
    sub_969240((__int64 *)&v141);
    if ( v118 )
    {
      v36 = sub_AD8D80(v114, (__int64)&v139);
      v144 = 257;
      v12 = sub_BD2C40(72, unk_3F10FD0);
      if ( !v12 )
        goto LABEL_62;
      goto LABEL_61;
    }
    sub_969240((__int64 *)&v139);
LABEL_67:
    if ( v122 == 38 )
    {
      sub_9865C0((__int64)&v137, a4);
      sub_C46A40((__int64)&v137, 1);
      v86 = v138;
      v138 = 0;
      v140 = v86;
      v139 = v137;
      sub_9865C0((__int64)&v141, (__int64)&v139);
      sub_1110B10((__int64)&v141, v18);
      sub_C46F20((__int64)&v141, 1u);
      v126 = v142;
      v125 = v141;
      sub_969240((__int64 *)&v139);
      v27 = (__int64 *)&v125;
      sub_969240((__int64 *)&v137);
      if ( !(unsigned __int8)sub_1112D20(a4, 1, v87, v88, v89) )
      {
        sub_9865C0((__int64)&v127, a4);
        sub_C46A40((__int64)&v127, 1);
        v130 = v128;
        v129 = v127;
        v128 = 0;
        sub_9865C0((__int64)&v131, (__int64)&v129);
        sub_1110B10((__int64)&v131, v18);
        if ( (unsigned __int8)sub_986B30((__int64 *)&v131, v18, v101, v102, (unsigned int)&v131) )
        {
          sub_969240((__int64 *)&v131);
          sub_969240(&v129);
          sub_969240(&v127);
        }
        else
        {
          sub_9865C0((__int64)&v139, a4);
          sub_C46A40((__int64)&v139, 1);
          LODWORD(v142) = v140;
          v140 = 0;
          v141 = v139;
          sub_9865C0((__int64)&v133, (__int64)&v125);
          sub_C46A40((__int64)&v133, 1);
          v136 = v134;
          v134 = 0;
          v135 = v133;
          sub_9865C0((__int64)&v137, (__int64)&v135);
          sub_D94900((__int64)&v137, v18);
          v111 = sub_AAD8B0((__int64)&v137, &v141);
          sub_969240((__int64 *)&v137);
          sub_969240((__int64 *)&v135);
          sub_969240((__int64 *)&v133);
          sub_969240((__int64 *)&v141);
          sub_969240((__int64 *)&v139);
          sub_969240((__int64 *)&v131);
          sub_969240(&v129);
          sub_969240(&v127);
          if ( v111 )
          {
            v103 = sub_AD8D80(v114, (__int64)&v125);
            v144 = 257;
            v104 = v103;
            v105 = sub_BD2C40(72, unk_3F10FD0);
            v12 = v105;
            if ( v105 )
              sub_1113300((__int64)v105, 38, v6, v104, (__int64)&v141);
            goto LABEL_62;
          }
        }
      }
    }
    else
    {
      if ( v122 != 34 )
        goto LABEL_69;
      sub_9865C0((__int64)&v137, a4);
      sub_C46A40((__int64)&v137, 1);
      v90 = v138;
      v138 = 0;
      v140 = v90;
      v139 = v137;
      sub_9865C0((__int64)&v141, (__int64)&v139);
      sub_1110B10((__int64)&v141, v18);
      sub_C46F20((__int64)&v141, 1u);
      v126 = v142;
      v125 = v141;
      sub_969240((__int64 *)&v139);
      sub_969240((__int64 *)&v137);
      sub_9865C0((__int64)&v133, a4);
      sub_C46A40((__int64)&v133, 1);
      v27 = (__int64 *)&v125;
      v136 = v134;
      v134 = 0;
      v135 = v133;
      sub_9865C0((__int64)&v127, (__int64)&v125);
      sub_C46A40((__int64)&v127, 1);
      v130 = v128;
      v129 = v127;
      v128 = 0;
      sub_9865C0((__int64)&v131, (__int64)&v129);
      sub_D94900((__int64)&v131, v18);
      if ( sub_AAD8B0((__int64)&v131, &v135) )
      {
        sub_969240((__int64 *)&v131);
        sub_969240(&v129);
        sub_969240(&v127);
        sub_969240((__int64 *)&v135);
        sub_969240((__int64 *)&v133);
        goto LABEL_145;
      }
      sub_9865C0((__int64)&v137, a4);
      sub_C46A40((__int64)&v137, 1);
      v97 = v138;
      v138 = 0;
      v140 = v97;
      v139 = v137;
      sub_9865C0((__int64)&v141, (__int64)&v139);
      sub_1110B10((__int64)&v141, v18);
      v109 = sub_986B30((__int64 *)&v141, v18, v98, v99, v100);
      sub_969240((__int64 *)&v141);
      sub_969240((__int64 *)&v139);
      sub_969240((__int64 *)&v137);
      sub_969240((__int64 *)&v131);
      sub_969240(&v129);
      sub_969240(&v127);
      sub_969240((__int64 *)&v135);
      sub_969240((__int64 *)&v133);
      if ( v109 )
      {
LABEL_145:
        v91 = sub_AD8D80(v114, (__int64)&v125);
        v144 = 257;
        v92 = v91;
        v93 = sub_BD2C40(72, unk_3F10FD0);
        v12 = v93;
        if ( v93 )
          sub_1113300((__int64)v93, 34, v6, v92, (__int64)&v141);
        goto LABEL_62;
      }
    }
    sub_969240((__int64 *)&v125);
LABEL_69:
    v39 = *(_DWORD *)(a4 + 8);
    if ( v39 > 2 )
    {
      v40 = *(_QWORD *)a4;
      v41 = 1LL << ((unsigned __int8)v39 - 1);
      if ( v39 > 0x40 )
      {
        v45 = (*(_QWORD *)(v40 + 8LL * ((v39 - 1) >> 6)) & v41) != 0 ? sub_C44500(a4) : sub_C444A0(a4);
      }
      else if ( (v41 & v40) != 0 )
      {
        v42 = ~(v40 << (64 - (unsigned __int8)v39));
        _BitScanReverse64(&v43, v42);
        v44 = v42 == 0;
        v45 = v43 ^ 0x3F;
        if ( v44 )
          v45 = 64;
      }
      else
      {
        _BitScanReverse64(&v94, v40);
        v95 = 64;
        v96 = v94 ^ 0x3F;
        if ( v40 )
          v95 = v96;
        v45 = v39 + v95 - 64;
      }
      if ( v18 >= v45 )
      {
        if ( v122 == 34 )
        {
          v106 = sub_AD6530(v114, v40);
          v144 = 257;
          v107 = v106;
          v108 = sub_BD2C40(72, unk_3F10FD0);
          v12 = v108;
          if ( v108 )
            sub_1113300((__int64)v108, 40, v6, v107, (__int64)&v141);
          return (unsigned __int8 *)v12;
        }
        if ( v112 )
        {
          v46 = sub_AD62B0(v114);
          v144 = 257;
          v47 = v46;
          v48 = sub_BD2C40(72, unk_3F10FD0);
          v12 = v48;
          if ( v48 )
            sub_1113300((__int64)v48, 38, v6, v47, (__int64)&v141);
          return (unsigned __int8 *)v12;
        }
      }
    }
    goto LABEL_48;
  }
  return sub_1126070(a1, a2, *(_QWORD *)(a3 - 32), a4, v13);
}
