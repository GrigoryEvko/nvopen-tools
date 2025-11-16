// Function: sub_10BD490
// Address: 0x10bd490
//
__int64 __fastcall sub_10BD490(
        __int64 a1,
        _BYTE *a2,
        unsigned __int8 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        int a8,
        __int64 a9)
{
  __int64 v12; // r12
  _QWORD **v14; // r13
  _BYTE *v15; // rsi
  unsigned __int64 v16; // rax
  unsigned int v17; // edx
  _QWORD *v18; // rax
  unsigned int v19; // ecx
  int v20; // eax
  __int64 v21; // rax
  char v22; // al
  __int64 result; // rax
  __int64 v24; // rdx
  _BYTE *v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rdx
  _BYTE *v28; // rax
  int v29; // eax
  _BYTE *v30; // rax
  __int64 v31; // r13
  __int64 v32; // rsi
  __int64 v33; // rcx
  int v34; // r8d
  int v35; // edi
  __int64 v36; // rcx
  __int64 v37; // r8
  __int64 v38; // rdi
  __int64 v39; // rdx
  __int64 *v40; // rax
  __int64 v41; // r15
  _DWORD *v42; // rax
  _DWORD *v43; // rbx
  bool v44; // r9
  __int64 v45; // rax
  __int64 v46; // rdx
  __int64 v47; // rcx
  __int64 v48; // rax
  char v49; // r9
  __int64 v50; // rax
  unsigned __int8 *v51; // rax
  unsigned int v52; // edx
  unsigned __int64 v53; // rax
  unsigned __int64 v54; // rax
  _DWORD *v55; // rax
  unsigned int v56; // eax
  unsigned __int64 v57; // rdx
  __int64 v58; // rdx
  unsigned int v59; // edx
  unsigned __int64 v60; // rax
  unsigned __int64 v61; // rax
  unsigned __int64 v62; // rax
  unsigned __int64 v63; // rax
  _BYTE *v64; // rbx
  _BYTE *v65; // r14
  __int64 v66; // rax
  int v67; // eax
  unsigned __int64 v68; // r15
  _DWORD *v69; // r15
  bool v70; // al
  unsigned int v71; // r15d
  unsigned __int64 v72; // rax
  _DWORD *v73; // rax
  unsigned __int64 v74; // r13
  bool v75; // bl
  unsigned int v76; // ecx
  unsigned int v77; // edx
  __int64 v78; // rax
  __int64 v79; // rax
  unsigned __int64 v80; // r15
  _DWORD *v81; // r15
  bool v82; // r13
  unsigned int v83; // edx
  _DWORD *v84; // rax
  bool v85; // r12
  unsigned int v86; // ecx
  int v87; // eax
  unsigned __int64 v88; // r15
  _DWORD *v89; // r15
  bool v90; // al
  unsigned int v91; // eax
  unsigned int v92; // eax
  unsigned int v93; // eax
  unsigned int v94; // eax
  unsigned int v95; // [rsp+Ch] [rbp-F4h]
  unsigned int v96; // [rsp+20h] [rbp-E0h]
  char v97; // [rsp+20h] [rbp-E0h]
  unsigned int v98; // [rsp+20h] [rbp-E0h]
  unsigned int v99; // [rsp+20h] [rbp-E0h]
  unsigned int v100; // [rsp+28h] [rbp-D8h]
  unsigned int v101; // [rsp+28h] [rbp-D8h]
  unsigned int v102; // [rsp+28h] [rbp-D8h]
  _DWORD **v103; // [rsp+28h] [rbp-D8h]
  unsigned int v104; // [rsp+30h] [rbp-D0h]
  bool v105; // [rsp+30h] [rbp-D0h]
  bool v106; // [rsp+30h] [rbp-D0h]
  __int64 v107; // [rsp+38h] [rbp-C8h]
  __int64 v108; // [rsp+38h] [rbp-C8h]
  __int64 v109; // [rsp+38h] [rbp-C8h]
  __int64 v110; // [rsp+38h] [rbp-C8h]
  unsigned __int64 v111; // [rsp+40h] [rbp-C0h] BYREF
  unsigned int v112; // [rsp+48h] [rbp-B8h]
  __int64 v113; // [rsp+50h] [rbp-B0h] BYREF
  unsigned int v114; // [rsp+58h] [rbp-A8h]
  __int64 v115; // [rsp+60h] [rbp-A0h] BYREF
  unsigned int v116; // [rsp+68h] [rbp-98h]
  __int64 v117; // [rsp+70h] [rbp-90h] BYREF
  unsigned int v118; // [rsp+78h] [rbp-88h]
  unsigned __int64 v119; // [rsp+80h] [rbp-80h] BYREF
  unsigned int v120; // [rsp+88h] [rbp-78h]
  unsigned __int64 v121; // [rsp+90h] [rbp-70h] BYREF
  unsigned int v122; // [rsp+98h] [rbp-68h]
  unsigned __int64 v123; // [rsp+A0h] [rbp-60h] BYREF
  unsigned int v124; // [rsp+A8h] [rbp-58h]
  __int16 v125; // [rsp+C0h] [rbp-40h]

  v12 = a5 + 24;
  if ( *(_BYTE *)a5 != 17 )
  {
    v27 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(a5 + 8) + 8LL) - 17;
    if ( (unsigned int)v27 > 1 )
      return 0;
    if ( *(_BYTE *)a5 > 0x15u )
      return 0;
    v28 = sub_AD7630(a5, 0, v27);
    if ( !v28 || *v28 != 17 )
      return 0;
    v12 = (__int64)(v28 + 24);
  }
  if ( *(_BYTE *)a6 == 17 )
  {
    v14 = (_QWORD **)(a6 + 24);
  }
  else
  {
    v24 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(a6 + 8) + 8LL) - 17;
    if ( (unsigned int)v24 > 1 )
      return 0;
    if ( *(_BYTE *)a6 > 0x15u )
      return 0;
    v25 = sub_AD7630(a6, 0, v24);
    if ( !v25 || *v25 != 17 )
      return 0;
    v14 = (_QWORD **)(v25 + 24);
  }
  if ( *(_BYTE *)a7 == 17 )
  {
    v15 = (_BYTE *)(a7 + 24);
    goto LABEL_6;
  }
  v26 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(a7 + 8) + 8LL) - 17;
  if ( (unsigned int)v26 > 1 )
    return 0;
  if ( *(_BYTE *)a7 > 0x15u )
    return 0;
  v30 = sub_AD7630(a7, 0, v26);
  if ( !v30 )
    return 0;
  v15 = v30 + 24;
  if ( *v30 != 17 )
    return 0;
LABEL_6:
  v104 = (a3 == 0) + 32;
  v112 = *((_DWORD *)v15 + 2);
  if ( v112 <= 0x40 )
  {
    v16 = *(_QWORD *)v15;
    v111 = *(_QWORD *)v15;
    if ( (a3 == 0) + 32 == a8 )
      goto LABEL_8;
    goto LABEL_40;
  }
  sub_C43780((__int64)&v111, (const void **)v15);
  if ( v104 != a8 )
  {
    if ( v112 <= 0x40 )
    {
      v16 = v111;
LABEL_40:
      v111 = (unsigned __int64)*v14 ^ v16;
      goto LABEL_8;
    }
    sub_C43C10(&v111, (__int64 *)v14);
  }
LABEL_8:
  v17 = *(_DWORD *)(v12 + 8);
  if ( v17 > 0x40 )
  {
    v102 = *(_DWORD *)(v12 + 8);
    v29 = sub_C444A0(v12);
    v17 = v102;
    if ( v102 - v29 > 0x40 )
      goto LABEL_11;
    v18 = **(_QWORD ***)v12;
  }
  else
  {
    v18 = *(_QWORD **)v12;
  }
  if ( !v18 )
    goto LABEL_19;
LABEL_11:
  v19 = *((_DWORD *)v14 + 2);
  if ( v19 <= 0x40 )
  {
    v21 = (__int64)*v14;
  }
  else
  {
    v96 = *((_DWORD *)v14 + 2);
    v100 = v17;
    v20 = sub_C444A0((__int64)v14);
    v19 = v96;
    v17 = v100;
    if ( v96 - v20 > 0x40 )
      goto LABEL_15;
    v21 = **v14;
  }
  if ( !v21 )
  {
LABEL_19:
    result = 0;
    goto LABEL_20;
  }
LABEL_15:
  if ( v17 <= 0x40 )
  {
    if ( ((unsigned __int64)*v14 & *(_QWORD *)v12) == 0 )
    {
LABEL_17:
      if ( v19 <= 0x40 )
      {
        if ( *v14 != (_QWORD *)v111 )
          goto LABEL_19;
      }
      else if ( !sub_C43C50((__int64)v14, (const void **)&v111) )
      {
        goto LABEL_19;
      }
      if ( *(_BYTE *)a4 != 78 )
        goto LABEL_19;
      v31 = *(_QWORD *)(a4 - 32);
      v32 = *(_QWORD *)(a4 + 8);
      v33 = *(_QWORD *)(v31 + 8);
      v34 = *(unsigned __int8 *)(v32 + 8);
      v35 = *(unsigned __int8 *)(v33 + 8);
      if ( (unsigned int)(v35 - 17) <= 1 != (unsigned int)(v34 - 17) <= 1
        || (unsigned int)(v35 - 17) <= 1
        && (*(_DWORD *)(v32 + 32) != *(_DWORD *)(v33 + 32) || ((_BYTE)v34 == 18) != ((_BYTE)v35 == 18)) )
      {
        goto LABEL_19;
      }
      if ( (unsigned __int8)sub_B2D610(*(_QWORD *)(*(_QWORD *)(a9 + 48) + 72LL), 72) )
        goto LABEL_19;
      v38 = *(_QWORD *)(v31 + 8);
      v39 = *(unsigned __int8 *)(v38 + 8);
      if ( (unsigned int)(v39 - 17) <= 1 )
      {
        v40 = *(__int64 **)(v38 + 16);
        v38 = *v40;
        v39 = *(unsigned __int8 *)(*v40 + 8);
      }
      if ( (unsigned __int8)v39 > 3u )
      {
        result = 0;
        if ( (_BYTE)v39 != 5 )
          goto LABEL_20;
      }
      v41 = sub_BCAC60(v38, 72, v39, v36, v37);
      v42 = sub_C33340();
      v43 = v42;
      if ( (_DWORD *)v41 == v42 )
        sub_C3C500(&v123, (__int64)v42);
      else
        sub_C373C0(&v123, v41);
      if ( (_DWORD *)v123 == v43 )
        sub_C3CF20((__int64)&v123, 0);
      else
        sub_C36EF0((_DWORD **)&v123, 0);
      if ( (_DWORD *)v123 == v43 )
        sub_C3E660((__int64)&v119, (__int64)&v123);
      else
        sub_C3A850((__int64)&v119, (__int64 *)&v123);
      sub_91D830(&v123);
      if ( v112 <= 0x40 )
      {
        if ( v111 == v119 )
          goto LABEL_63;
        v45 = 0;
      }
      else
      {
        v44 = sub_C43C50((__int64)&v111, (const void **)&v119);
        v45 = 0;
        if ( v44 )
        {
LABEL_63:
          sub_9865C0((__int64)&v123, (__int64)&v119);
          sub_987160((__int64)&v123, (__int64)&v119, v46, v47, (__int64)&v123);
          v122 = v124;
          v121 = v123;
          v48 = ~(1LL << ((unsigned __int8)v124 - 1));
          if ( v124 > 0x40 )
            *(_QWORD *)(v123 + 8LL * ((v124 - 1) >> 6)) &= v48;
          else
            v121 = v123 & v48;
          v49 = sub_AAD8B0(v12, &v121);
          v50 = 0;
          if ( v49 )
          {
            v125 = 257;
            v51 = sub_AD9290(*(_QWORD *)(v31 + 8), 0);
            HIDWORD(v117) = 0;
            v50 = sub_B35C90(
                    a9,
                    7 - ((unsigned int)(a3 == 0) - 1),
                    v31,
                    (__int64)v51,
                    (__int64)&v123,
                    0,
                    (unsigned int)v117,
                    0);
          }
          v108 = v50;
          sub_969240((__int64 *)&v121);
          v45 = v108;
        }
      }
      v109 = v45;
      sub_969240((__int64 *)&v119);
      result = v109;
      goto LABEL_20;
    }
  }
  else
  {
    v101 = v19;
    v22 = sub_C446A0((__int64 *)v12, (__int64 *)v14);
    v19 = v101;
    if ( !v22 )
      goto LABEL_17;
  }
  v52 = *(_DWORD *)(v12 + 8);
  v120 = v52;
  if ( v52 <= 0x40 )
  {
    v53 = *(_QWORD *)v12;
    v119 = *(_QWORD *)v12;
LABEL_75:
    v54 = (unsigned __int64)*v14 & v53;
    v120 = 0;
    v119 = v54;
LABEL_76:
    v55 = (_DWORD *)(v111 & v54);
    v124 = v52;
    v121 = (unsigned __int64)v55;
    v123 = (unsigned __int64)v55;
    v122 = 0;
    v103 = (_DWORD **)v55;
    goto LABEL_77;
  }
  sub_C43780((__int64)&v119, (const void **)v12);
  v52 = v120;
  if ( v120 <= 0x40 )
  {
    v53 = v119;
    goto LABEL_75;
  }
  sub_C43B90(&v119, (__int64 *)v14);
  v52 = v120;
  v54 = v119;
  v120 = 0;
  v122 = v52;
  v121 = v119;
  if ( v52 <= 0x40 )
    goto LABEL_76;
  sub_C43B90(&v121, (__int64 *)&v111);
  v52 = v122;
  v122 = 0;
  v124 = v52;
  v103 = (_DWORD **)v121;
  v123 = v121;
  if ( v52 <= 0x40 )
  {
    v55 = (_DWORD *)v121;
  }
  else
  {
    v98 = v52;
    v67 = sub_C444A0((__int64)&v123);
    v52 = v98;
    if ( v98 - v67 > 0x40 )
      goto LABEL_78;
    v55 = *v103;
  }
LABEL_77:
  if ( !v55 )
  {
    v99 = v52;
    sub_9865C0((__int64)&v113, v12);
    v76 = v114;
    v77 = v99;
    if ( v114 > 0x40 )
    {
      sub_C43C10(&v113, (__int64 *)v14);
      v76 = v114;
      v78 = v113;
      v114 = 0;
      v77 = v99;
      v116 = v76;
      v115 = v113;
      if ( v76 > 0x40 )
      {
        sub_C43B90(&v115, (__int64 *)v12);
        v86 = v116;
        v79 = v115;
        v116 = 0;
        v77 = v99;
        v118 = v86;
        v117 = v115;
        if ( v86 > 0x40 )
        {
          v87 = sub_C44630((__int64)&v117);
          v77 = v99;
          if ( v87 == 1 )
          {
LABEL_121:
            v97 = 1;
LABEL_122:
            v95 = v77;
            sub_969240(&v117);
            sub_969240(&v115);
            sub_969240(&v113);
            v52 = v95;
            goto LABEL_79;
          }
LABEL_140:
          v97 = 0;
          goto LABEL_122;
        }
LABEL_119:
        if ( v79 && (v79 & (v79 - 1)) == 0 )
          goto LABEL_121;
        goto LABEL_140;
      }
    }
    else
    {
      v78 = (unsigned __int64)*v14 ^ v113;
      v116 = v114;
      v113 = v78;
      v115 = v78;
      v114 = 0;
    }
    v79 = *(_QWORD *)v12 & v78;
    v118 = v76;
    v115 = v79;
    v117 = v79;
    v116 = 0;
    goto LABEL_119;
  }
LABEL_78:
  v97 = 0;
LABEL_79:
  if ( v52 > 0x40 && v103 )
    j_j___libc_free_0_0(v103);
  if ( v122 > 0x40 && v121 )
    j_j___libc_free_0_0(v121);
  if ( v120 > 0x40 && v119 )
    j_j___libc_free_0_0(v119);
  v56 = *(_DWORD *)(v12 + 8);
  v124 = v56;
  if ( v97 )
  {
    if ( v56 > 0x40 )
    {
      sub_C43780((__int64)&v123, (const void **)v12);
      v56 = v124;
      if ( v124 > 0x40 )
      {
        sub_C43BD0(&v123, (__int64 *)v14);
        v56 = v124;
        v58 = v123;
LABEL_92:
        v116 = v56;
        v115 = v58;
        v59 = *(_DWORD *)(v12 + 8);
        v120 = v59;
        if ( v59 > 0x40 )
        {
          sub_C43780((__int64)&v119, (const void **)v12);
          v59 = v120;
          if ( v120 > 0x40 )
          {
            sub_C43C10(&v119, (__int64 *)v14);
            v59 = v120;
            v61 = v119;
            v120 = 0;
            v122 = v59;
            v121 = v119;
            if ( v59 > 0x40 )
            {
              sub_C43B90(&v121, (__int64 *)v12);
              v59 = v122;
              v62 = v121;
              v122 = 0;
              v124 = v59;
              v123 = v121;
              if ( v59 > 0x40 )
              {
                sub_C43BD0(&v123, (__int64 *)&v111);
                v59 = v124;
                v63 = v123;
                goto LABEL_97;
              }
LABEL_96:
              v63 = v111 | v62;
LABEL_97:
              v118 = v59;
              v117 = v63;
              sub_969240((__int64 *)&v121);
              sub_969240((__int64 *)&v119);
              v64 = (_BYTE *)sub_AD8D80(*(_QWORD *)(a4 + 8), (__int64)&v115);
              v65 = (_BYTE *)sub_AD8D80(*(_QWORD *)(a4 + 8), (__int64)&v117);
              v125 = 257;
              v66 = sub_A82350((unsigned int **)a9, (_BYTE *)a4, v64, (__int64)&v123);
              v125 = 257;
              v110 = sub_92B530((unsigned int **)a9, v104, v66, v65, (__int64)&v123);
              sub_969240(&v117);
              sub_969240(&v115);
              result = v110;
              goto LABEL_20;
            }
LABEL_95:
            v62 = *(_QWORD *)v12 & v61;
            v122 = 0;
            v121 = v62;
            goto LABEL_96;
          }
          v60 = v119;
        }
        else
        {
          v60 = *(_QWORD *)v12;
          v119 = *(_QWORD *)v12;
        }
        v61 = (unsigned __int64)*v14 ^ v60;
        v122 = v59;
        v119 = v61;
        v121 = v61;
        v120 = 0;
        goto LABEL_95;
      }
      v57 = v123;
    }
    else
    {
      v57 = *(_QWORD *)v12;
      v123 = *(_QWORD *)v12;
    }
    v58 = (unsigned __int64)*v14 | v57;
    goto LABEL_92;
  }
  if ( v56 <= 0x40 )
  {
    v68 = *(_QWORD *)v12;
    v123 = *(_QWORD *)v12;
LABEL_105:
    v69 = (_DWORD *)((unsigned __int64)*v14 & v68);
    v124 = 0;
    v123 = (unsigned __int64)v69;
LABEL_106:
    v70 = *(_QWORD *)v12 == (_QWORD)v69;
    goto LABEL_107;
  }
  sub_C43780((__int64)&v123, (const void **)v12);
  if ( v124 <= 0x40 )
  {
    v68 = v123;
    goto LABEL_105;
  }
  sub_C43B90(&v123, (__int64 *)v14);
  v91 = v124;
  v69 = (_DWORD *)v123;
  v124 = 0;
  v122 = v91;
  v121 = v123;
  if ( v91 <= 0x40 )
    goto LABEL_106;
  v70 = sub_C43C50((__int64)&v121, (const void **)v12);
  if ( v69 )
  {
    v105 = v70;
    j_j___libc_free_0_0(v69);
    v70 = v105;
    if ( v124 > 0x40 )
    {
      if ( v123 )
      {
        j_j___libc_free_0_0(v123);
        v70 = v105;
      }
    }
  }
LABEL_107:
  if ( v70 )
    goto LABEL_108;
  v124 = *(_DWORD *)(v12 + 8);
  if ( v124 <= 0x40 )
  {
    v88 = *(_QWORD *)v12;
    v123 = *(_QWORD *)v12;
LABEL_143:
    v89 = (_DWORD *)((unsigned __int64)*v14 & v88);
    v124 = 0;
    v123 = (unsigned __int64)v89;
LABEL_144:
    v90 = *v14 == (_QWORD *)v89;
    goto LABEL_145;
  }
  sub_C43780((__int64)&v123, (const void **)v12);
  if ( v124 <= 0x40 )
  {
    v88 = v123;
    goto LABEL_143;
  }
  sub_C43B90(&v123, (__int64 *)v14);
  v94 = v124;
  v89 = (_DWORD *)v123;
  v124 = 0;
  v122 = v94;
  v121 = v123;
  if ( v94 <= 0x40 )
    goto LABEL_144;
  v90 = sub_C43C50((__int64)&v121, (const void **)v14);
  if ( v89 )
  {
    v106 = v90;
    j_j___libc_free_0_0(v89);
    v90 = v106;
    if ( v124 > 0x40 )
    {
      if ( v123 )
      {
        j_j___libc_free_0_0(v123);
        v90 = v106;
      }
    }
  }
LABEL_145:
  if ( !v90 )
    goto LABEL_19;
LABEL_108:
  v71 = v112;
  if ( v112 <= 0x40 )
  {
    if ( !v111 )
    {
LABEL_110:
      v124 = *(_DWORD *)(v12 + 8);
      if ( v124 > 0x40 )
      {
        sub_C43780((__int64)&v123, (const void **)v12);
        if ( v124 > 0x40 )
        {
          sub_C43B90(&v123, (__int64 *)v14);
          v92 = v124;
          v74 = v123;
          v124 = 0;
          v122 = v92;
          v121 = v123;
          if ( v92 > 0x40 )
          {
            v75 = sub_C43C50((__int64)&v121, (const void **)v12);
            if ( v74 )
            {
              j_j___libc_free_0_0(v74);
              if ( v124 > 0x40 )
              {
                if ( v123 )
                  j_j___libc_free_0_0(v123);
              }
            }
LABEL_114:
            if ( !v75 )
              goto LABEL_19;
            goto LABEL_115;
          }
LABEL_113:
          v75 = *(_QWORD *)v12 == v74;
          goto LABEL_114;
        }
        v72 = v123;
      }
      else
      {
        v72 = *(_QWORD *)v12;
        v123 = *(_QWORD *)v12;
      }
      v73 = (_DWORD *)((unsigned __int64)*v14 & v72);
      v124 = 0;
      v123 = (unsigned __int64)v73;
      v74 = (unsigned __int64)v73;
      goto LABEL_113;
    }
  }
  else if ( v71 == (unsigned int)sub_C444A0((__int64)&v111) )
  {
    goto LABEL_110;
  }
  v124 = *(_DWORD *)(v12 + 8);
  if ( v124 <= 0x40 )
  {
    v80 = *(_QWORD *)v12;
    v123 = *(_QWORD *)v12;
LABEL_128:
    v81 = (_DWORD *)((unsigned __int64)*v14 & v80);
    v124 = 0;
    v123 = (unsigned __int64)v81;
LABEL_129:
    v82 = *v14 == (_QWORD *)v81;
    goto LABEL_130;
  }
  sub_C43780((__int64)&v123, (const void **)v12);
  if ( v124 <= 0x40 )
  {
    v80 = v123;
    goto LABEL_128;
  }
  sub_C43B90(&v123, (__int64 *)v14);
  v93 = v124;
  v81 = (_DWORD *)v123;
  v124 = 0;
  v122 = v93;
  v121 = v123;
  if ( v93 <= 0x40 )
    goto LABEL_129;
  v82 = sub_C43C50((__int64)&v121, (const void **)v14);
  if ( v81 )
  {
    j_j___libc_free_0_0(v81);
    if ( v124 > 0x40 )
    {
      if ( v123 )
        j_j___libc_free_0_0(v123);
    }
  }
LABEL_130:
  if ( v82 )
    goto LABEL_134;
  sub_9865C0((__int64)&v121, v12);
  v83 = v122;
  if ( v122 > 0x40 )
  {
    sub_C43B90(&v121, (__int64 *)&v111);
    v83 = v122;
    v84 = (_DWORD *)v121;
  }
  else
  {
    v84 = (_DWORD *)(v111 & v121);
    v121 &= v111;
  }
  v124 = v83;
  v123 = (unsigned __int64)v84;
  v122 = 0;
  v85 = sub_D94970((__int64)&v123, 0);
  sub_969240((__int64 *)&v123);
  sub_969240((__int64 *)&v121);
  if ( !v85 )
  {
LABEL_134:
    if ( *a2 == 82 )
      a2[1] &= ~2u;
    result = (__int64)a2;
    goto LABEL_20;
  }
LABEL_115:
  result = sub_AD64C0(*(_QWORD *)(a1 + 8), a3 ^ 1u, 0);
LABEL_20:
  if ( v112 > 0x40 )
  {
    if ( v111 )
    {
      v107 = result;
      j_j___libc_free_0_0(v111);
      return v107;
    }
  }
  return result;
}
