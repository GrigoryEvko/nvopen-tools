// Function: sub_1E88360
// Address: 0x1e88360
//
__int64 __fastcall sub_1E88360(__int64 a1, __int64 a2, signed int a3, unsigned int a4)
{
  __int64 v4; // r15
  __int64 *v6; // rax
  __int64 *i; // r14
  __int64 v8; // r12
  __int64 *v9; // rsi
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rcx
  __int64 v13; // r11
  unsigned __int64 v14; // rdx
  unsigned __int64 v15; // rax
  __int64 v16; // rcx
  __int64 v17; // rsi
  __int64 v18; // rdx
  int v19; // edi
  __int64 v20; // r10
  unsigned int v21; // r9d
  __int16 v22; // r8
  _WORD *v23; // r9
  __int16 *v24; // r10
  unsigned __int16 v25; // r8
  __int16 *v26; // rdi
  __int16 v27; // r9
  __int64 v28; // rsi
  __int64 v29; // rdi
  __int64 *v30; // r14
  __int64 result; // rax
  unsigned int *v32; // rbx
  __int64 v33; // rax
  __int64 v34; // r12
  __int64 v35; // rdi
  __int64 v36; // r8
  __int64 v37; // rdx
  unsigned __int64 v38; // rcx
  __int64 v39; // r12
  __int64 v40; // rsi
  __int64 v41; // rax
  __int64 v42; // rdx
  __int64 v43; // r8
  __int64 v44; // rax
  unsigned __int64 v45; // rcx
  unsigned __int64 v46; // r11
  __int64 v47; // r14
  __int64 v48; // rdx
  __int64 *v49; // r12
  __int64 v50; // r14
  __int64 v51; // rdx
  __int64 *v52; // rcx
  __int64 v53; // r15
  unsigned __int64 v54; // rdx
  unsigned int *v55; // r11
  __int64 v56; // rdx
  void *v57; // rax
  __int64 v58; // rax
  __int64 v59; // rax
  __int64 v60; // rdx
  __int64 v61; // rax
  __int64 v62; // rax
  __int64 v63; // rax
  __int64 v64; // rdx
  void *v65; // rax
  __int64 v66; // rdx
  __int64 v67; // rax
  unsigned __int64 j; // rax
  __int64 v69; // rcx
  __int64 v70; // rsi
  __int64 v71; // rdx
  char v72; // r12
  unsigned __int16 v73; // di
  int v74; // r14d
  char v75; // r10
  __int64 v76; // rsi
  __int64 v77; // rdi
  __int64 *v78; // rdx
  const char *v79; // rsi
  __int64 v80; // rax
  bool v81; // r10
  bool v82; // di
  int v83; // ecx
  __int64 v84; // rdx
  const char *v85; // rsi
  char v86; // di
  __int64 *v87; // [rsp+10h] [rbp-D0h]
  __int64 v88; // [rsp+20h] [rbp-C0h]
  __int64 *v89; // [rsp+28h] [rbp-B8h]
  char v90; // [rsp+30h] [rbp-B0h]
  bool v91; // [rsp+30h] [rbp-B0h]
  char v92; // [rsp+30h] [rbp-B0h]
  __int64 v93; // [rsp+30h] [rbp-B0h]
  char v94; // [rsp+38h] [rbp-A8h]
  __int64 v95; // [rsp+38h] [rbp-A8h]
  __int64 v96; // [rsp+38h] [rbp-A8h]
  __int64 v97; // [rsp+40h] [rbp-A0h]
  __int64 v98; // [rsp+40h] [rbp-A0h]
  unsigned __int64 v99; // [rsp+40h] [rbp-A0h]
  unsigned __int64 v100; // [rsp+40h] [rbp-A0h]
  unsigned int *v101; // [rsp+40h] [rbp-A0h]
  __int64 *v102; // [rsp+40h] [rbp-A0h]
  __int64 *v103; // [rsp+48h] [rbp-98h]
  __int64 v104; // [rsp+48h] [rbp-98h]
  __int64 *v105; // [rsp+48h] [rbp-98h]
  unsigned __int64 v106; // [rsp+48h] [rbp-98h]
  bool v107; // [rsp+48h] [rbp-98h]
  unsigned __int64 v108; // [rsp+48h] [rbp-98h]
  unsigned __int64 v109; // [rsp+48h] [rbp-98h]
  unsigned __int64 v112; // [rsp+58h] [rbp-88h]
  __int64 v113; // [rsp+58h] [rbp-88h]
  __int64 v114; // [rsp+58h] [rbp-88h]
  __int64 v115; // [rsp+58h] [rbp-88h]
  __int64 v116; // [rsp+58h] [rbp-88h]
  __int64 v117; // [rsp+58h] [rbp-88h]
  unsigned int *v118; // [rsp+58h] [rbp-88h]
  __int64 v119; // [rsp+58h] [rbp-88h]
  __int64 v120; // [rsp+58h] [rbp-88h]
  __int64 v121; // [rsp+58h] [rbp-88h]
  __int64 v122; // [rsp+58h] [rbp-88h]
  __int64 v123; // [rsp+58h] [rbp-88h]
  __int64 v124; // [rsp+58h] [rbp-88h]
  __int64 v125; // [rsp+58h] [rbp-88h]
  __int64 v126; // [rsp+58h] [rbp-88h]
  char v127; // [rsp+58h] [rbp-88h]
  __int64 v128; // [rsp+58h] [rbp-88h]
  __int64 v129; // [rsp+58h] [rbp-88h]
  __int64 v130; // [rsp+68h] [rbp-78h] BYREF
  _QWORD v131[4]; // [rsp+70h] [rbp-70h] BYREF
  _QWORD v132[10]; // [rsp+90h] [rbp-50h] BYREF

  v4 = a2;
  v6 = *(__int64 **)(a2 + 64);
  v103 = &v6[*(unsigned int *)(a2 + 72)];
  if ( v103 != v6 )
  {
    for ( i = *(__int64 **)(a2 + 64); v103 != i; ++i )
    {
      v8 = *i;
      v112 = *(_QWORD *)(*i + 8) & 0xFFFFFFFFFFFFFFF8LL;
      if ( !v112 )
        continue;
      v97 = *(_QWORD *)(*i + 8);
      v9 = (__int64 *)sub_1DB3C70((__int64 *)v4, v97);
      if ( v9 == (__int64 *)(*(_QWORD *)v4 + 24LL * *(unsigned int *)(v4 + 8))
        || (*(_DWORD *)((*v9 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v9 >> 1) & 3) > (*(_DWORD *)(v112 + 24)
                                                                                            | (unsigned int)(v97 >> 1)
                                                                                            & 3)
        || (v10 = v9[2]) == 0 )
      {
        sub_1E857B0(a1, "Value not live at VNInfo def and not marked unused", *(__int64 **)(a1 + 16));
      }
      else
      {
        if ( v8 == v10 )
        {
          v98 = *(_QWORD *)(v8 + 8);
          v113 = *(_QWORD *)(*(_QWORD *)(a1 + 568) + 272LL);
          v11 = sub_1DA9310(v113, v98);
          v12 = v113;
          v13 = v11;
          if ( !v11 )
          {
            sub_1E857B0(a1, "Invalid VNInfo definition index", *(__int64 **)(a1 + 16));
            goto LABEL_4;
          }
          v114 = (v98 >> 1) & 3;
          if ( ((v98 >> 1) & 3) == 0 )
          {
            if ( *(_QWORD *)(*(_QWORD *)(v12 + 392) + 16LL * *(unsigned int *)(v11 + 48)) == v98 )
              continue;
            v84 = v11;
            v85 = "PHIDef VNInfo is not defined at MBB start";
            goto LABEL_155;
          }
          if ( (v98 & 0xFFFFFFFFFFFFFFF8LL) == 0
            || (v14 = v98 & 0xFFFFFFFFFFFFFFF8LL, (v99 = *(_QWORD *)((v98 & 0xFFFFFFFFFFFFFFF8LL) + 16)) == 0) )
          {
            v84 = v11;
            v85 = "No instruction at VNInfo def index";
LABEL_155:
            sub_1E869F0(a1, v85, v84);
            goto LABEL_4;
          }
          if ( !a3 )
            continue;
          v15 = *(_QWORD *)(v14 + 16);
          if ( (*(_BYTE *)(v99 + 46) & 4) != 0 )
          {
            do
              v15 = *(_QWORD *)v15 & 0xFFFFFFFFFFFFFFF8LL;
            while ( (*(_BYTE *)(v15 + 46) & 4) != 0 );
          }
          v16 = *(_QWORD *)(v99 + 24) + 24LL;
          do
          {
            v17 = *(_QWORD *)(v15 + 32);
            v18 = v17 + 40LL * *(unsigned int *)(v15 + 40);
            if ( v17 != v18 )
              goto LABEL_23;
            v15 = *(_QWORD *)(v15 + 8);
          }
          while ( v16 != v15 && (*(_BYTE *)(v15 + 46) & 4) != 0 );
          if ( v17 == v18 )
          {
            v94 = 0;
LABEL_174:
            v93 = v13;
            sub_1E86C30(a1, "Defining instruction does not modify register", v99);
            sub_1E85D70(a1, v4, a3, a4);
            sub_1E85BF0((unsigned int *)v8);
            v13 = v93;
            v114 = (*(__int64 *)(v8 + 8) >> 1) & 3;
LABEL_168:
            if ( v94 )
            {
              v84 = v13;
              v85 = "Early clobber def must be at an early-clobber slot";
              if ( v114 == 1 )
                continue;
            }
            else
            {
              if ( v114 == 2 )
                continue;
              v84 = v13;
              v85 = "Non-PHI, non-early clobber def must be at a register slot";
            }
            goto LABEL_155;
          }
LABEL_23:
          v94 = 0;
          v90 = 0;
          while ( 2 )
          {
            if ( !*(_BYTE *)v17 && (*(_BYTE *)(v17 + 3) & 0x10) != 0 )
            {
              v19 = *(_DWORD *)(v17 + 8);
              if ( a3 < 0 )
              {
                if ( a3 == v19 )
                  goto LABEL_162;
              }
              else if ( v19 > 0 )
              {
                v20 = *(_QWORD *)(a1 + 40);
                v21 = *(_DWORD *)(*(_QWORD *)(v20 + 8) + 24LL * (unsigned int)v19 + 16);
                v22 = v19 * (v21 & 0xF);
                v23 = (_WORD *)(*(_QWORD *)(v20 + 56) + 2LL * (v21 >> 4));
                v24 = v23 + 1;
                v25 = *v23 + v22;
LABEL_29:
                v26 = v24;
                if ( v24 )
                {
                  while ( a3 != v25 )
                  {
                    v27 = *v26;
                    v24 = 0;
                    ++v26;
                    if ( !v27 )
                      goto LABEL_29;
                    v25 += v27;
                    if ( !v26 )
                      goto LABEL_33;
                  }
LABEL_162:
                  if ( !a4
                    || (*(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 40) + 248LL) + 4LL * ((*(_DWORD *)v17 >> 8) & 0xFFF))
                      & a4) != 0 )
                  {
                    v86 = v94;
                    v90 = 1;
                    if ( (*(_BYTE *)(v17 + 4) & 4) != 0 )
                      v86 = 1;
                    v94 = v86;
                  }
                }
              }
            }
LABEL_33:
            v28 = v17 + 40;
            v29 = v18;
            if ( v28 == v18 )
            {
              while ( 1 )
              {
                v15 = *(_QWORD *)(v15 + 8);
                if ( v16 == v15 || (*(_BYTE *)(v15 + 46) & 4) == 0 )
                  break;
                v18 = *(_QWORD *)(v15 + 32);
                v29 = v18 + 40LL * *(unsigned int *)(v15 + 40);
                if ( v18 != v29 )
                  goto LABEL_39;
              }
              if ( v29 == v18 )
              {
                if ( v90 )
                  goto LABEL_168;
                goto LABEL_174;
              }
            }
            else
            {
              v18 = v28;
            }
LABEL_39:
            v17 = v18;
            v18 = v29;
            continue;
          }
        }
        sub_1E857B0(a1, "Live segment at def has different VNInfo", *(__int64 **)(a1 + 16));
      }
LABEL_4:
      sub_1E85D70(a1, v4, a3, a4);
      sub_1E85BF0((unsigned int *)v8);
    }
  }
  v30 = *(__int64 **)v4;
  result = *(_QWORD *)v4 + 24LL * *(unsigned int *)(v4 + 8);
  v87 = (__int64 *)result;
  if ( result == *(_QWORD *)v4 )
    return result;
  do
  {
    while ( 1 )
    {
      while ( 1 )
      {
        v32 = (unsigned int *)v30[2];
        v33 = *v32;
        if ( (unsigned int)v33 >= *(_DWORD *)(v4 + 72) || v32 != *(unsigned int **)(*(_QWORD *)(v4 + 64) + 8 * v33) )
        {
          sub_1E857B0(a1, "Foreign valno in live segment", *(__int64 **)(a1 + 16));
          sub_1E85D70(a1, v4, a3, a4);
          sub_1E85B90((__int64)v30);
          sub_1E85BF0(v32);
        }
        if ( (*((_QWORD *)v32 + 1) & 0xFFFFFFFFFFFFFFF8LL) == 0 )
        {
          sub_1E857B0(a1, "Live segment valno is marked unused", *(__int64 **)(a1 + 16));
          sub_1E85D70(a1, v4, a3, a4);
          sub_1E85B90((__int64)v30);
        }
        v34 = *v30;
        v115 = *(_QWORD *)(*(_QWORD *)(a1 + 568) + 272LL);
        v35 = v115;
        v36 = sub_1DA9310(v115, *v30);
        v89 = v30 + 3;
        if ( !v36 )
        {
          v78 = *(__int64 **)(a1 + 16);
          v79 = "Bad start of live segment, no basic block";
LABEL_132:
          sub_1E857B0(a1, v79, v78);
          goto LABEL_94;
        }
        if ( v34 != *(_QWORD *)(*(_QWORD *)(v115 + 392) + 16LL * *(unsigned int *)(v36 + 48))
          && v34 != *((_QWORD *)v32 + 1) )
        {
          v116 = v36;
          sub_1E869F0(a1, "Live segment must begin at MBB entry or valno def", v36);
          sub_1E85D70(a1, v4, a3, a4);
          sub_1E85B90((__int64)v30);
          v36 = v116;
          v35 = *(_QWORD *)(*(_QWORD *)(a1 + 568) + 272LL);
        }
        v37 = v30[1];
        v38 = v37 & 0xFFFFFFFFFFFFFFF8LL;
        v39 = (v37 >> 1) & 3;
        if ( ((v37 >> 1) & 3) != 0 )
          v40 = v38 | (2LL * ((int)v39 - 1));
        else
          v40 = *(_QWORD *)v38 & 0xFFFFFFFFFFFFFFF8LL | 6;
        v95 = v36;
        v100 = v30[1] & 0xFFFFFFFFFFFFFFF8LL;
        v104 = v30[1];
        v41 = sub_1DA9310(v35, v40);
        v42 = v104;
        v88 = v41;
        v43 = v95;
        if ( !v41 )
        {
          v78 = *(__int64 **)(a1 + 16);
          v79 = "Bad end of live segment, no basic block";
          goto LABEL_132;
        }
        result = *(_QWORD *)(v35 + 392) + 16LL * *(unsigned int *)(v41 + 48);
        if ( *(_QWORD *)(result + 8) == v104 )
          goto LABEL_88;
        if ( a3 < 0 )
          break;
        v44 = *((_QWORD *)v32 + 1);
        if ( (v44 & 6) != 0 )
          break;
        if ( v44 != *v30 )
          break;
        result = v44 & 0xFFFFFFFFFFFFFFF8LL | 6;
        if ( result != v104 )
          break;
        v30 += 3;
        if ( v87 == v89 )
          return result;
      }
      if ( v39 )
        break;
      v45 = *(_QWORD *)v100 & 0xFFFFFFFFFFFFFFF8LL;
      if ( v45 )
        goto LABEL_59;
LABEL_93:
      sub_1E869F0(a1, "Live segment doesn't end at a valid instruction", v88);
LABEL_94:
      sub_1E85D70(a1, v4, a3, a4);
      result = (__int64)sub_1E85B90((__int64)v30);
      v30 += 3;
      if ( v87 == v89 )
        return result;
    }
    v45 = ((2LL * ((int)v39 - 1)) | v100) & 0xFFFFFFFFFFFFFFF8LL;
    if ( !v45 )
      goto LABEL_93;
LABEL_59:
    v46 = *(_QWORD *)(v45 + 16);
    if ( !v46 )
      goto LABEL_93;
    if ( !v39 )
    {
      v109 = *(_QWORD *)(v45 + 16);
      sub_1E869F0(a1, "Live segment ends at B slot of an instruction", v88);
      sub_1E85D70(a1, v4, a3, a4);
      sub_1E85B90((__int64)v30);
      v46 = v109;
      v43 = v95;
      v42 = v30[1];
      v39 = (v42 >> 1) & 3;
    }
    if ( v39 == 3 )
    {
      if ( (*v30 & 0xFFFFFFFFFFFFFFF8LL) == (v42 & 0xFFFFFFFFFFFFFFF8LL) )
        goto LABEL_64;
      v108 = v46;
      v128 = v43;
      sub_1E869F0(a1, "Live segment ending at dead slot spans instructions", v88);
      sub_1E85D70(a1, v4, a3, a4);
      sub_1E85B90((__int64)v30);
      v42 = v30[1];
      v46 = v108;
      v43 = v128;
      v39 = (v42 >> 1) & 3;
    }
    if ( v39 == 1 && (v89 == (__int64 *)(*(_QWORD *)v4 + 24LL * *(unsigned int *)(v4 + 8)) || *v89 != v42) )
    {
      v106 = v46;
      v126 = v43;
      sub_1E869F0(
        a1,
        "Live segment ending at early clobber slot must be redefined by an EC def in the same instruction",
        v88);
      sub_1E85D70(a1, v4, a3, a4);
      sub_1E85B90((__int64)v30);
      v46 = v106;
      v43 = v126;
      if ( a3 >= 0 )
        goto LABEL_65;
      goto LABEL_103;
    }
LABEL_64:
    if ( a3 >= 0 )
      goto LABEL_65;
LABEL_103:
    for ( j = v46; (*(_BYTE *)(j + 46) & 4) != 0; j = *(_QWORD *)j & 0xFFFFFFFFFFFFFFF8LL )
      ;
    v69 = *(_QWORD *)(v46 + 24) + 24LL;
    while ( 1 )
    {
      v70 = *(_QWORD *)(j + 32);
      v71 = v70 + 40LL * *(unsigned int *)(j + 40);
      if ( v70 != v71 )
        break;
      j = *(_QWORD *)(j + 8);
      if ( v69 == j || (*(_BYTE *)(j + 46) & 4) == 0 )
      {
        if ( v70 == v71 )
        {
          if ( (((unsigned __int8)v30[1] ^ 6) & 6) != 0 )
            goto LABEL_158;
          v127 = 0;
          goto LABEL_150;
        }
        break;
      }
    }
    v127 = 0;
    v92 = 0;
    v107 = 0;
    v102 = v30;
    while ( 1 )
    {
      if ( *(_BYTE *)v70 || a3 != *(_DWORD *)(v70 + 8) )
      {
LABEL_122:
        v76 = v70 + 40;
        v77 = v71;
        if ( v76 == v71 )
          goto LABEL_126;
LABEL_148:
        v71 = v76;
        goto LABEL_128;
      }
      v72 = *(_BYTE *)(v70 + 3) & 0x10;
      v73 = (*(_DWORD *)v70 >> 8) & 0xFFF;
      if ( v73 )
      {
        v74 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 40) + 248LL) + 4LL * v73);
        if ( !v72 )
          goto LABEL_118;
        v92 = 1;
        v74 = ~v74;
      }
      else
      {
        v74 = -1;
        if ( !v72 )
          goto LABEL_120;
      }
      v75 = v127;
      if ( ((*(_BYTE *)(v70 + 3) >> 6) & ((*(_BYTE *)(v70 + 3) & 0x10) != 0)) != 0 )
        v75 = (*(_BYTE *)(v70 + 3) >> 6) & ((*(_BYTE *)(v70 + 3) & 0x10) != 0);
      v127 = v75;
LABEL_118:
      if ( a4 && (v74 & a4) == 0 )
        goto LABEL_122;
LABEL_120:
      if ( (*(_BYTE *)(v70 + 4) & 1) != 0 || (*(_BYTE *)(v70 + 4) & 2) != 0 )
        goto LABEL_122;
      v81 = v107;
      v82 = v72 == 0 || v73 != 0;
      if ( v82 )
        v81 = v82;
      v76 = v70 + 40;
      v77 = v71;
      v107 = v81;
      if ( v76 != v71 )
        goto LABEL_148;
LABEL_126:
      while ( 1 )
      {
        j = *(_QWORD *)(j + 8);
        if ( v69 == j || (*(_BYTE *)(j + 46) & 4) == 0 )
          break;
        v71 = *(_QWORD *)(j + 32);
        v77 = v71 + 40LL * *(unsigned int *)(j + 40);
        if ( v71 != v77 )
          goto LABEL_128;
      }
      if ( v71 == v77 )
        break;
LABEL_128:
      v70 = v71;
      v71 = v77;
    }
    v30 = v102;
    if ( (((unsigned __int8)*(v89 - 2) ^ 6) & 6) != 0 )
    {
      if ( v107 )
        goto LABEL_65;
      v80 = *(_QWORD *)(a1 + 48);
      if ( *(_BYTE *)(v80 + 16) )
      {
        if ( (*(_BYTE *)((*(_QWORD *)(*(_QWORD *)(v80 + 24) + 16LL * (a3 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL) + 29)
            & (a4 == 0)) != 0
          && v92 )
        {
          goto LABEL_65;
        }
      }
LABEL_158:
      v129 = v43;
      sub_1E86C30(a1, "Instruction ending live segment doesn't read the register", v46);
      v83 = a4;
LABEL_159:
      sub_1E85D70(a1, v4, a3, v83);
      sub_1E85B90((__int64)v30);
      v43 = v129;
      goto LABEL_65;
    }
LABEL_150:
    if ( !a4 && !v127 )
    {
      v129 = v43;
      sub_1E86C30(a1, "Instruction ending live segment on dead slot has no dead flag", v46);
      v83 = 0;
      goto LABEL_159;
    }
LABEL_65:
    result = *((_QWORD *)v32 + 1);
    if ( result != *(v89 - 3) || (result & 6) == 0 )
    {
      v101 = v32;
      v47 = v43;
      goto LABEL_67;
    }
    if ( v43 != v88 )
    {
      v101 = v32;
      v47 = *(_QWORD *)(v43 + 8);
LABEL_67:
      if ( a3 < 0 )
        goto LABEL_71;
LABEL_68:
      if ( *(_BYTE *)(v47 + 180) )
      {
LABEL_69:
        if ( v88 == v47 )
          goto LABEL_88;
        goto LABEL_70;
      }
LABEL_71:
      v48 = *((_QWORD *)v101 + 1);
      v91 = (v48 & 6) == 0
         && *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 568) + 272LL) + 392LL)
                      + 16LL * *(unsigned int *)(v47 + 48)) == v48;
      result = *(_QWORD *)(v47 + 72);
      v49 = *(__int64 **)(v47 + 64);
      v105 = (__int64 *)result;
      if ( v49 == (__int64 *)result )
        goto LABEL_69;
      v96 = v47;
      v50 = v4;
      while ( 2 )
      {
        while ( 1 )
        {
          v53 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 568) + 272LL) + 392LL)
                          + 16LL * *(unsigned int *)(*v49 + 48)
                          + 8);
          v54 = v53 & 0xFFFFFFFFFFFFFFF8LL;
          v51 = ((v53 >> 1) & 3) != 0
              ? (2LL * (int)(((v53 >> 1) & 3) - 1)) | v54
              : *(_QWORD *)v54 & 0xFFFFFFFFFFFFFFF8LL | 6;
          v117 = v51;
          v52 = (__int64 *)sub_1DB3C70((__int64 *)v50, v51);
          if ( v52 == (__int64 *)(*(_QWORD *)v50 + 24LL * *(unsigned int *)(v50 + 8)) )
            break;
          result = *(_DWORD *)((*v52 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v52 >> 1) & 3;
          if ( (unsigned int)result > (*(_DWORD *)((v117 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v117 >> 1) & 3) )
            break;
          v55 = (unsigned int *)v52[2];
          if ( !v55 )
            break;
          if ( v91 || v101 == v55 )
            goto LABEL_80;
          v56 = *v49++;
          v118 = (unsigned int *)v52[2];
          sub_1E869F0(a1, "Different value live out of predecessor", v56);
          sub_1E85D70(a1, v50, a3, a4);
          v57 = sub_16E8CB0();
          v58 = sub_1263B40((__int64)v57, "Valno #");
          v59 = sub_16E7A90(v58, *v118);
          v119 = sub_1263B40(v59, " live out of ");
          sub_1DD5B60(v132, *(v49 - 1));
          sub_1E869D0((__int64)v132, v119, v60);
          v61 = sub_1549FC0(v119, 0x40u);
          v131[0] = v53;
          v120 = v61;
          sub_1F10810(v131, v61);
          v62 = sub_1263B40(v120, "\nValno #");
          v63 = sub_16E7A90(v62, *v101);
          v121 = sub_1263B40(v63, " live into ");
          sub_1DD5B60(v131, v96);
          sub_1E869D0((__int64)v131, v121, v64);
          v122 = sub_1549FC0(v121, 0x40u);
          v130 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 568) + 272LL) + 392LL)
                           + 16LL * *(unsigned int *)(v96 + 48));
          sub_1F10810(&v130, v122);
          sub_1549FC0(v122, 0xAu);
          sub_A17130((__int64)v131);
          result = (__int64)sub_A17130((__int64)v132);
          if ( v105 == v49 )
          {
LABEL_87:
            v4 = v50;
            v47 = v96;
            if ( v88 == v96 )
              goto LABEL_88;
LABEL_70:
            v47 = *(_QWORD *)(v47 + 8);
            if ( a3 >= 0 )
              goto LABEL_68;
            goto LABEL_71;
          }
        }
        result = a4;
        if ( !a4 || !v91 )
        {
          sub_1E869F0(a1, "Register not marked live out of predecessor", *v49);
          sub_1E85D70(a1, v50, a3, a4);
          sub_1E85BF0(v101);
          v65 = sub_16E8CB0();
          v123 = sub_1263B40((__int64)v65, " live into ");
          sub_1DD5B60(v132, v96);
          sub_1E869D0((__int64)v132, v123, v66);
          v124 = sub_1549FC0(v123, 0x40u);
          v131[0] = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 568) + 272LL) + 392LL)
                              + 16LL * *(unsigned int *)(v96 + 48));
          sub_1F10810(v131, v124);
          v67 = sub_1263B40(v124, ", not live before ");
          v131[0] = v53;
          v125 = v67;
          sub_1F10810(v131, v67);
          sub_1549FC0(v125, 0xAu);
          result = (__int64)sub_A17130((__int64)v132);
        }
LABEL_80:
        if ( v105 == ++v49 )
          goto LABEL_87;
        continue;
      }
    }
LABEL_88:
    v30 = v89;
  }
  while ( v87 != v89 );
  return result;
}
