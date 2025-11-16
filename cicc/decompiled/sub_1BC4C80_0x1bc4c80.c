// Function: sub_1BC4C80
// Address: 0x1bc4c80
//
void __fastcall sub_1BC4C80(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 *a6)
{
  __int64 v6; // rdx
  __int64 *v7; // rbx
  __int64 **v8; // rsi
  __int64 *v9; // r14
  __int64 v10; // rax
  __int64 *v11; // r12
  __int64 *v12; // rax
  __int64 *v13; // rdi
  __int64 *v14; // rdx
  __int64 *v15; // rsi
  __int64 *v16; // r15
  __int64 *v17; // r12
  __int64 v18; // r14
  unsigned __int64 v19; // rdi
  __int64 v20; // rdx
  int v21; // eax
  __int64 v22; // r13
  __int64 v23; // r15
  __int64 *v24; // rbx
  __int64 v25; // r12
  __int64 *v26; // r10
  __int64 v27; // rsi
  __int64 *v28; // rdi
  __int64 *v29; // rax
  __int64 *v30; // rcx
  __int64 *v31; // r14
  __int64 v32; // r12
  __int64 v33; // rdi
  _QWORD *v34; // rax
  __int64 *v35; // r13
  __int64 v36; // r12
  __int64 v37; // r15
  unsigned int v38; // r12d
  __int64 v39; // rax
  unsigned int v40; // edx
  unsigned __int64 v41; // r14
  int v42; // r9d
  char v43; // r13
  int v44; // eax
  __int64 **v45; // r8
  __int64 v46; // rdi
  unsigned __int64 *v47; // r15
  char v48; // r13
  __int64 v49; // r14
  unsigned __int64 *v50; // rbx
  unsigned __int64 *v51; // rdx
  unsigned __int64 v52; // rax
  __int64 v53; // rax
  __int64 **v54; // r13
  __int64 v55; // rax
  __int64 **v56; // rsi
  __int64 v57; // rax
  __int64 *v58; // rbx
  __int64 *v59; // r15
  __int64 *v60; // rdx
  unsigned __int64 v61; // r12
  __int64 v62; // rdi
  __int64 v63; // r12
  unsigned __int64 v64; // r12
  __int64 v65; // rdi
  __int64 v66; // r12
  unsigned __int64 v67; // r12
  __int64 v68; // rdi
  __int64 v69; // r12
  unsigned __int64 v70; // r12
  __int64 v71; // rdi
  __int64 v72; // r12
  __int64 *v73; // rdx
  __int64 *v74; // rdx
  __int64 *v75; // rdx
  signed __int64 v76; // rax
  signed __int64 v77; // rax
  unsigned __int64 v78; // rax
  __int64 *v79; // r14
  __int64 *v80; // r15
  unsigned int v81; // eax
  unsigned __int64 v82; // rax
  __int64 v83; // r12
  unsigned __int64 v84; // rax
  __int64 v85; // r12
  unsigned __int64 v86; // rax
  __int64 v87; // r12
  int v88; // [rsp-308h] [rbp-308h]
  int v89; // [rsp-300h] [rbp-300h]
  __int64 v90; // [rsp-2E8h] [rbp-2E8h]
  __int64 *v91; // [rsp-2E8h] [rbp-2E8h]
  __int64 *v92; // [rsp-2E0h] [rbp-2E0h]
  unsigned int v93; // [rsp-2E0h] [rbp-2E0h]
  __int64 *v94; // [rsp-2E0h] [rbp-2E0h]
  int v95; // [rsp-2E0h] [rbp-2E0h]
  __int64 v96; // [rsp-2D8h] [rbp-2D8h]
  __int64 **v97; // [rsp-2D0h] [rbp-2D0h]
  __int64 **v98; // [rsp-2D0h] [rbp-2D0h]
  unsigned __int64 *v99; // [rsp-2D0h] [rbp-2D0h]
  unsigned __int64 v100; // [rsp-2C8h] [rbp-2C8h] BYREF
  unsigned int v101; // [rsp-2C0h] [rbp-2C0h]
  __int64 v102; // [rsp-2B8h] [rbp-2B8h]
  unsigned int v103; // [rsp-2B0h] [rbp-2B0h]
  _BYTE *v104; // [rsp-2A8h] [rbp-2A8h] BYREF
  __int64 v105; // [rsp-2A0h] [rbp-2A0h]
  _BYTE v106[32]; // [rsp-298h] [rbp-298h] BYREF
  __int64 *v107; // [rsp-278h] [rbp-278h] BYREF
  __int64 v108; // [rsp-270h] [rbp-270h]
  _BYTE v109[256]; // [rsp-268h] [rbp-268h] BYREF
  __int64 v110; // [rsp-168h] [rbp-168h] BYREF
  __int64 *v111; // [rsp-160h] [rbp-160h]
  __int64 *v112; // [rsp-158h] [rbp-158h]
  __int64 v113; // [rsp-150h] [rbp-150h]
  int v114; // [rsp-148h] [rbp-148h]
  _BYTE v115[320]; // [rsp-140h] [rbp-140h] BYREF

  v6 = *((unsigned int *)a1 + 98);
  if ( !(_DWORD)v6 )
    return;
  v7 = a1;
  v8 = (__int64 **)*a1;
  v9 = *(__int64 **)*a1;
  v97 = (__int64 **)*a1;
  v96 = *(_QWORD *)*v9;
  if ( *(_BYTE *)(v96 + 8) != 11 )
    return;
  v110 = 0;
  v10 = *((unsigned int *)v8 + 2);
  v113 = 32;
  v114 = 0;
  v11 = &v9[v10];
  v12 = (__int64 *)v115;
  v111 = (__int64 *)v115;
  v112 = (__int64 *)v115;
  if ( v11 == v9 )
  {
    v16 = (__int64 *)a1[48];
    v13 = (__int64 *)v115;
    v17 = &v16[3 * v6];
    goto LABEL_17;
  }
  v13 = (__int64 *)v115;
  do
  {
LABEL_7:
    a5 = *v9;
    if ( v13 != v12 )
    {
LABEL_5:
      sub_16CCBA0((__int64)&v110, *v9);
      v13 = v112;
      v12 = v111;
      goto LABEL_6;
    }
    a6 = &v13[HIDWORD(v113)];
    if ( a6 == v13 )
    {
LABEL_35:
      if ( HIDWORD(v113) >= (unsigned int)v113 )
        goto LABEL_5;
      ++HIDWORD(v113);
      *a6 = a5;
      v12 = v111;
      ++v110;
      v13 = v112;
    }
    else
    {
      v14 = v13;
      v15 = 0;
      while ( a5 != *v14 )
      {
        if ( *v14 == -2 )
          v15 = v14;
        if ( a6 == ++v14 )
        {
          if ( !v15 )
            goto LABEL_35;
          ++v9;
          *v15 = a5;
          v13 = v112;
          --v114;
          v12 = v111;
          ++v110;
          if ( v11 != v9 )
            goto LABEL_7;
          goto LABEL_16;
        }
      }
    }
LABEL_6:
    ++v9;
  }
  while ( v11 != v9 );
LABEL_16:
  v16 = (__int64 *)v7[48];
  v17 = &v16[3 * *((unsigned int *)v7 + 98)];
  if ( v16 == v17 )
  {
    v21 = v114;
  }
  else
  {
    while ( 1 )
    {
LABEL_17:
      v18 = *v16;
      if ( v13 == v12 )
      {
        v20 = (__int64)&v12[HIDWORD(v113)];
        if ( (__int64 *)v20 == v12 )
        {
LABEL_33:
          v12 = (__int64 *)v20;
        }
        else
        {
          while ( v18 != *v12 )
          {
            if ( (__int64 *)v20 == ++v12 )
              goto LABEL_33;
          }
        }
      }
      else
      {
        v12 = sub_16CC9F0((__int64)&v110, *v16);
        v19 = (unsigned __int64)v112;
        if ( v18 == *v12 )
        {
          v20 = (__int64)(v112 == v111 ? &v112[HIDWORD(v113)] : &v112[(unsigned int)v113]);
        }
        else
        {
          if ( v112 != v111 )
            goto LABEL_20;
          v12 = &v112[HIDWORD(v113)];
          v20 = (__int64)v12;
        }
      }
      if ( v12 == (__int64 *)v20 )
        goto LABEL_69;
      *v12 = -2;
      v16 += 3;
      v21 = ++v114;
      if ( v17 == v16 )
        break;
      v13 = v112;
      v12 = v111;
    }
  }
  if ( HIDWORD(v113) != v21 )
    goto LABEL_69;
  v22 = *v7;
  if ( v7[1] == *v7 )
    goto LABEL_56;
  v92 = v7;
  v23 = v7[1];
  while ( 2 )
  {
    v24 = *(__int64 **)v22;
    v25 = *(_QWORD *)v22 + 8LL * *(unsigned int *)(v22 + 8);
    if ( v25 != *(_QWORD *)v22 )
    {
      a6 = v112;
      v26 = v111;
      do
      {
LABEL_45:
        v27 = *v24;
        if ( a6 != v26 )
          goto LABEL_43;
        v28 = &a6[HIDWORD(v113)];
        a5 = HIDWORD(v113);
        if ( a6 != v28 )
        {
          v29 = a6;
          v30 = 0;
          while ( v27 != *v29 )
          {
            if ( *v29 == -2 )
              v30 = v29;
            if ( v28 == ++v29 )
            {
              if ( !v30 )
                goto LABEL_70;
              ++v24;
              *v30 = v27;
              a6 = v112;
              --v114;
              v26 = v111;
              ++v110;
              if ( (__int64 *)v25 != v24 )
                goto LABEL_45;
              goto LABEL_54;
            }
          }
          goto LABEL_44;
        }
LABEL_70:
        if ( HIDWORD(v113) < (unsigned int)v113 )
        {
          a5 = (unsigned int)++HIDWORD(v113);
          *v28 = v27;
          v26 = v111;
          ++v110;
          a6 = v112;
        }
        else
        {
LABEL_43:
          sub_16CCBA0((__int64)&v110, v27);
          a6 = v112;
          v26 = v111;
        }
LABEL_44:
        ++v24;
      }
      while ( (__int64 *)v25 != v24 );
    }
LABEL_54:
    v22 += 176;
    if ( v23 != v22 )
      continue;
    break;
  }
  v7 = v92;
LABEL_56:
  v31 = *v97;
  v32 = (__int64)&(*v97)[*((unsigned int *)v97 + 2)];
  if ( *v97 == (__int64 *)v32 )
  {
LABEL_61:
    v107 = (__int64 *)v109;
    v108 = 0x2000000000LL;
    v104 = v106;
    v105 = 0x400000000LL;
    v35 = *v97;
    v36 = (__int64)&(*v97)[*((unsigned int *)v97 + 2)];
    if ( *v97 == (__int64 *)v36 )
      goto LABEL_97;
    do
    {
      if ( !(unsigned __int8)sub_1BBF240(*v35, (__int64)&v110, (__int64)&v107, (__int64)&v104, (_QWORD *)a5, (int)a6) )
        goto LABEL_65;
      ++v35;
    }
    while ( (__int64 *)v36 != v35 );
    v35 = *v97;
    v37 = (__int64)&(*v97)[*((unsigned int *)v97 + 2)];
    if ( *v97 == (__int64 *)v37 )
    {
LABEL_97:
      v38 = 8;
    }
    else
    {
      v38 = 8;
      do
      {
        sub_13A29C0((__int64)&v100, v7[171], (__int64 *)*v35);
        if ( v101 <= 0x40 )
        {
          if ( v100 )
          {
            _BitScanReverse64((unsigned __int64 *)&v39, v100);
            if ( v38 < 64 - ((unsigned int)v39 ^ 0x3F) )
              v38 = 64 - (v39 ^ 0x3F);
          }
        }
        else
        {
          v93 = v101;
          v40 = v93 - sub_16A57B0((__int64)&v100);
          if ( v38 < v40 )
            v38 = v40;
          if ( v100 )
            j_j___libc_free_0_0(v100);
        }
        ++v35;
      }
      while ( (__int64 *)v37 != v35 );
      v35 = *v97;
    }
    v41 = v38;
    if ( v38 != sub_127FA20(v7[172], *(_QWORD *)*v35) )
      goto LABEL_86;
    v54 = (__int64 **)*v97;
    v55 = *((unsigned int *)v97 + 2);
    v56 = (__int64 **)&(*v97)[v55];
    v57 = (v55 * 8) >> 5;
    v90 = v57;
    if ( v57 )
    {
      v94 = v7;
      v58 = *v97;
      v59 = (__int64 *)&v54[4 * v57];
      while ( 1 )
      {
        if ( *((_BYTE *)sub_1648700(*(_QWORD *)(*v58 + 8)) + 16) != 56 )
        {
          v60 = v58;
          v7 = v94;
          goto LABEL_103;
        }
        if ( *((_BYTE *)sub_1648700(*(_QWORD *)(v58[1] + 8)) + 16) != 56 )
        {
          v73 = v58;
          v7 = v94;
          v60 = v73 + 1;
          goto LABEL_103;
        }
        if ( *((_BYTE *)sub_1648700(*(_QWORD *)(v58[2] + 8)) + 16) != 56 )
        {
          v74 = v58;
          v7 = v94;
          v60 = v74 + 2;
          goto LABEL_103;
        }
        if ( *((_BYTE *)sub_1648700(*(_QWORD *)(v58[3] + 8)) + 16) != 56 )
          break;
        v58 += 4;
        if ( v58 == v59 )
        {
          v7 = v94;
          goto LABEL_145;
        }
      }
      v75 = v58;
      v7 = v94;
      v60 = v75 + 3;
LABEL_103:
      if ( v56 == (__int64 **)v60 )
        goto LABEL_133;
      goto LABEL_86;
    }
    v59 = *v97;
LABEL_145:
    v76 = (char *)v56 - (char *)v59;
    if ( (char *)v56 - (char *)v59 != 16 )
    {
      if ( v76 != 24 )
      {
        if ( v76 != 8 )
          goto LABEL_149;
        goto LABEL_165;
      }
      if ( *((_BYTE *)sub_1648700(*(_QWORD *)(*v59 + 8)) + 16) != 56 )
        goto LABEL_148;
      ++v59;
    }
    if ( *((_BYTE *)sub_1648700(*(_QWORD *)(*v59 + 8)) + 16) != 56 )
    {
LABEL_148:
      if ( v56 != (__int64 **)v59 )
        goto LABEL_86;
LABEL_149:
      if ( !v90 )
      {
LABEL_150:
        v77 = (char *)v56 - (char *)v54;
        if ( (char *)v56 - (char *)v54 != 16 )
        {
          if ( v77 != 24 )
          {
            if ( v77 != 8 )
              goto LABEL_153;
            goto LABEL_188;
          }
          sub_14C2530((__int64)&v100, *v54, v7[172], 0, 0, 0, 0, 0);
          v42 = v88;
          if ( v101 > 0x40 )
            v82 = *(_QWORD *)(v100 + 8LL * ((v101 - 1) >> 6));
          else
            v82 = v100;
          v83 = v82 & (1LL << ((unsigned __int8)v101 - 1));
          if ( v103 > 0x40 && v102 )
            j_j___libc_free_0_0(v102);
          if ( v101 > 0x40 && v100 )
            j_j___libc_free_0_0(v100);
          if ( !v83 )
            goto LABEL_197;
          ++v54;
        }
        sub_14C2530((__int64)&v100, *v54, v7[172], 0, 0, 0, 0, 0);
        if ( v101 > 0x40 )
          v84 = *(_QWORD *)(v100 + 8LL * ((v101 - 1) >> 6));
        else
          v84 = v100;
        v85 = v84 & (1LL << ((unsigned __int8)v101 - 1));
        if ( v103 > 0x40 && v102 )
          j_j___libc_free_0_0(v102);
        if ( v101 > 0x40 && v100 )
          j_j___libc_free_0_0(v100);
        if ( !v85 )
          goto LABEL_197;
        ++v54;
LABEL_188:
        sub_14C2530((__int64)&v100, *v54, v7[172], 0, 0, 0, 0, 0);
        if ( v101 > 0x40 )
          v86 = *(_QWORD *)(v100 + 8LL * ((v101 - 1) >> 6));
        else
          v86 = v100;
        v87 = v86 & (1LL << ((unsigned __int8)v101 - 1));
        if ( v103 > 0x40 && v102 )
          j_j___libc_free_0_0(v102);
        if ( v101 > 0x40 && v100 )
          j_j___libc_free_0_0(v100);
        if ( v87 )
        {
LABEL_153:
          v78 = (unsigned __int64)v107;
          v79 = &v107[(unsigned int)v108];
          if ( v79 != v107 )
          {
            v54 = v56;
LABEL_155:
            v80 = (__int64 *)v78;
            v38 = 8;
            do
            {
              v91 = (__int64 *)*v80;
              v95 = sub_14C23D0(*v80, v7[172], 0, v7[170], 0, v7[169]);
              v81 = sub_127FA20(v7[172], *v91) - v95;
              if ( v38 < v81 )
                v38 = v81;
              ++v80;
            }
            while ( v79 != v80 );
            v41 = v38;
            if ( v56 != v54 )
            {
LABEL_160:
              v43 = 0;
              v41 = ++v38;
              if ( !v38 )
                goto LABEL_88;
              goto LABEL_87;
            }
LABEL_86:
            v43 = 1;
LABEL_87:
            if ( (v41 & (v41 - 1)) == 0 )
            {
LABEL_89:
              if ( v38 < *(_DWORD *)(v96 + 8) >> 8 )
              {
                v44 = v105;
                if ( (_DWORD)v105 )
                {
                  v45 = &v107;
                  do
                  {
                    v98 = v45;
                    v46 = *(_QWORD *)&v104[8 * v44 - 8];
                    LODWORD(v105) = v44 - 1;
                    sub_1BBF240(v46, (__int64)&v110, (__int64)v45, (__int64)&v104, v45, v42);
                    v44 = v105;
                    v45 = v98;
                  }
                  while ( (_DWORD)v105 );
                }
                v47 = (unsigned __int64 *)v107;
                if ( &v107[(unsigned int)v108] != v107 )
                {
                  v48 = v43 ^ 1;
                  v49 = (__int64)(v7 + 184);
                  v50 = (unsigned __int64 *)&v107[(unsigned int)v108];
                  v51 = &v100;
                  do
                  {
                    v52 = *v47++;
                    v99 = v51;
                    v100 = v52;
                    v53 = sub_1BC4770(v49, v51);
                    v51 = v99;
                    *(_QWORD *)v53 = v38;
                    *(_BYTE *)(v53 + 8) = v48;
                  }
                  while ( v50 != v47 );
                }
              }
LABEL_65:
              if ( v104 != v106 )
                _libc_free((unsigned __int64)v104);
              if ( v107 != (__int64 *)v109 )
                _libc_free((unsigned __int64)v107);
              goto LABEL_69;
            }
LABEL_88:
            v38 = sub_1454B60(v41);
            goto LABEL_89;
          }
LABEL_167:
          v43 = 1;
          v38 = 8;
          goto LABEL_89;
        }
LABEL_197:
        v78 = (unsigned __int64)v107;
        v79 = &v107[(unsigned int)v108];
        if ( v79 != v107 )
          goto LABEL_155;
        if ( v56 != v54 )
        {
          v38 = 8;
          goto LABEL_160;
        }
        goto LABEL_167;
      }
LABEL_133:
      while ( 1 )
      {
        sub_14C2530((__int64)&v100, *v54, v7[172], 0, 0, 0, 0, 0);
        v72 = 1LL << ((unsigned __int8)v101 - 1);
        if ( v101 <= 0x40 )
          break;
        v61 = *(_QWORD *)(v100 + 8LL * ((v101 - 1) >> 6)) & v72;
        if ( v103 > 0x40 )
        {
          v62 = v102;
          if ( v102 )
            goto LABEL_107;
        }
LABEL_108:
        if ( v100 )
          j_j___libc_free_0_0(v100);
LABEL_110:
        if ( !v61 )
          goto LABEL_197;
        sub_14C2530((__int64)&v100, v54[1], v7[172], 0, 0, 0, 0, 0);
        v42 = v89;
        v63 = 1LL << ((unsigned __int8)v101 - 1);
        if ( v101 <= 0x40 )
        {
          v64 = v100 & v63;
          if ( v103 <= 0x40 )
            goto LABEL_117;
          v65 = v102;
          if ( !v102 )
            goto LABEL_117;
LABEL_114:
          j_j___libc_free_0_0(v65);
          if ( v101 <= 0x40 )
            goto LABEL_117;
          goto LABEL_115;
        }
        v64 = *(_QWORD *)(v100 + 8LL * ((v101 - 1) >> 6)) & v63;
        if ( v103 > 0x40 )
        {
          v65 = v102;
          if ( v102 )
            goto LABEL_114;
        }
LABEL_115:
        if ( v100 )
          j_j___libc_free_0_0(v100);
LABEL_117:
        if ( !v64 )
        {
          ++v54;
          goto LABEL_197;
        }
        sub_14C2530((__int64)&v100, v54[2], v7[172], 0, 0, 0, 0, 0);
        v66 = 1LL << ((unsigned __int8)v101 - 1);
        if ( v101 <= 0x40 )
        {
          v67 = v100 & v66;
          if ( v103 <= 0x40 )
            goto LABEL_124;
          v68 = v102;
          if ( !v102 )
            goto LABEL_124;
LABEL_121:
          j_j___libc_free_0_0(v68);
          if ( v101 <= 0x40 )
            goto LABEL_124;
          goto LABEL_122;
        }
        v67 = *(_QWORD *)(v100 + 8LL * ((v101 - 1) >> 6)) & v66;
        if ( v103 > 0x40 )
        {
          v68 = v102;
          if ( v102 )
            goto LABEL_121;
        }
LABEL_122:
        if ( v100 )
          j_j___libc_free_0_0(v100);
LABEL_124:
        if ( !v67 )
        {
          v54 += 2;
          goto LABEL_197;
        }
        sub_14C2530((__int64)&v100, v54[3], v7[172], 0, 0, 0, 0, 0);
        v69 = 1LL << ((unsigned __int8)v101 - 1);
        if ( v101 <= 0x40 )
        {
          v70 = v100 & v69;
          if ( v103 <= 0x40 )
            goto LABEL_131;
          v71 = v102;
          if ( !v102 )
            goto LABEL_131;
LABEL_128:
          j_j___libc_free_0_0(v71);
          if ( v101 <= 0x40 )
            goto LABEL_131;
          goto LABEL_129;
        }
        v70 = *(_QWORD *)(v100 + 8LL * ((v101 - 1) >> 6)) & v69;
        if ( v103 > 0x40 )
        {
          v71 = v102;
          if ( v102 )
            goto LABEL_128;
        }
LABEL_129:
        if ( v100 )
          j_j___libc_free_0_0(v100);
LABEL_131:
        if ( !v70 )
        {
          v54 += 3;
          goto LABEL_197;
        }
        v54 += 4;
        if ( !--v90 )
          goto LABEL_150;
      }
      v61 = v100 & v72;
      if ( v103 <= 0x40 )
        goto LABEL_110;
      v62 = v102;
      if ( !v102 )
        goto LABEL_110;
LABEL_107:
      j_j___libc_free_0_0(v62);
      if ( v101 <= 0x40 )
        goto LABEL_110;
      goto LABEL_108;
    }
    ++v59;
LABEL_165:
    if ( *((_BYTE *)sub_1648700(*(_QWORD *)(*v59 + 8)) + 16) == 56 )
      goto LABEL_149;
    goto LABEL_148;
  }
  while ( 1 )
  {
    v33 = *(_QWORD *)(*v31 + 8);
    if ( !v33 )
      break;
    if ( *(_QWORD *)(v33 + 8) )
      break;
    v34 = sub_1648700(v33);
    if ( sub_1A018F0((__int64)&v110, (__int64)v34) )
      break;
    if ( (__int64 *)v32 == ++v31 )
      goto LABEL_61;
  }
LABEL_69:
  v19 = (unsigned __int64)v112;
LABEL_20:
  if ( (__int64 *)v19 != v111 )
    _libc_free(v19);
}
