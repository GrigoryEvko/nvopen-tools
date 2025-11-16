// Function: sub_1AF8020
// Address: 0x1af8020
//
__int64 __fastcall sub_1AF8020(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __m128 a7,
        __m128i a8,
        __m128i a9,
        double a10,
        double a11,
        double a12,
        double a13,
        __m128 a14,
        __int64 a15,
        char a16,
        int a17,
        char a18)
{
  _QWORD *v18; // r12
  __int64 v19; // r13
  double v20; // xmm4_8
  double v21; // xmm5_8
  __int64 result; // rax
  signed __int64 v23; // rdx
  char *v24; // rdi
  __int64 v25; // r8
  int v26; // eax
  __int64 v27; // rdx
  double v28; // xmm4_8
  double v29; // xmm5_8
  __int64 v30; // rax
  __int64 v31; // rbx
  unsigned __int64 v32; // rax
  unsigned __int64 v33; // r14
  __int64 v34; // rax
  unsigned __int64 v35; // rdi
  unsigned __int64 v36; // rax
  __int64 v37; // rbx
  __int64 v38; // r14
  int v39; // r15d
  unsigned int v40; // r12d
  __int64 v41; // rax
  _QWORD *v42; // r14
  __int64 v43; // rbx
  __int64 v44; // rdi
  __int64 v45; // rax
  _QWORD *v46; // r8
  _QWORD *v47; // r14
  __int64 v48; // rbx
  char v49; // r15
  char v50; // r9
  int v51; // ecx
  _QWORD *v52; // rdi
  __int64 v53; // rax
  _QWORD *v54; // rax
  unsigned __int64 v55; // r9
  unsigned __int64 v56; // rcx
  __int64 v57; // rdx
  __int64 v58; // rdx
  _QWORD *v59; // rax
  __int64 v60; // rsi
  unsigned __int64 v61; // rdi
  __int64 v62; // rsi
  __int64 v63; // r14
  __int64 v64; // rax
  __int64 v65; // rcx
  int v66; // r9d
  unsigned int v67; // edx
  __int64 v68; // rsi
  __int64 *v69; // r15
  char *v70; // rdx
  char *v71; // rdi
  __int64 v72; // rax
  char *v73; // rax
  _QWORD *v74; // rax
  __int64 v75; // rax
  __int64 v76; // rax
  int v77; // ecx
  unsigned int v78; // r9d
  __int64 *v79; // rsi
  __int64 v80; // rdi
  __int64 v81; // rax
  __int64 v82; // r15
  char *v83; // rcx
  __int64 v84; // rdx
  char *v85; // rdx
  __int64 v86; // rsi
  _QWORD *v87; // rcx
  __int64 v88; // rax
  __int64 v89; // rsi
  _QWORD *v90; // rdx
  __int64 v91; // rdx
  __int64 v92; // rcx
  signed __int64 v93; // rcx
  __int64 v94; // rax
  __int64 v95; // r14
  _QWORD *v96; // rax
  unsigned int v97; // r14d
  int v98; // eax
  __int64 *v99; // r15
  __int64 v100; // rsi
  __int64 v101; // r14
  __int64 v102; // rdi
  int v103; // esi
  int v104; // r10d
  __int64 v105; // rsi
  _QWORD *v106; // [rsp+8h] [rbp-B8h]
  _QWORD *v107; // [rsp+10h] [rbp-B0h]
  __int64 v108; // [rsp+18h] [rbp-A8h]
  __int64 v109; // [rsp+20h] [rbp-A0h]
  __int64 v110; // [rsp+28h] [rbp-98h]
  _QWORD *v111; // [rsp+30h] [rbp-90h]
  unsigned __int64 v112; // [rsp+30h] [rbp-90h]
  __int64 v113; // [rsp+38h] [rbp-88h]
  _QWORD *v114; // [rsp+38h] [rbp-88h]
  char v115; // [rsp+38h] [rbp-88h]
  __int64 v116; // [rsp+38h] [rbp-88h]
  int v117; // [rsp+40h] [rbp-80h]
  _QWORD *v118; // [rsp+40h] [rbp-80h]
  __int64 v119; // [rsp+40h] [rbp-80h]
  __int64 v120; // [rsp+48h] [rbp-78h]
  __int64 v121; // [rsp+48h] [rbp-78h]
  __int64 *v122; // [rsp+48h] [rbp-78h]
  __int64 v123; // [rsp+48h] [rbp-78h]
  __int64 v124; // [rsp+48h] [rbp-78h]
  int v125; // [rsp+50h] [rbp-70h] BYREF
  __int64 v126; // [rsp+58h] [rbp-68h]
  __int64 v127; // [rsp+60h] [rbp-60h]
  __int64 v128; // [rsp+68h] [rbp-58h]
  __int64 v129; // [rsp+70h] [rbp-50h]
  __int64 v130; // [rsp+78h] [rbp-48h]
  __int64 v131; // [rsp+80h] [rbp-40h]
  char v132; // [rsp+88h] [rbp-38h]
  char v133; // [rsp+89h] [rbp-37h]

  v18 = (_QWORD *)a1;
  v126 = a2;
  v127 = a3;
  v125 = a17;
  v128 = a4;
  v131 = a15;
  v129 = a5;
  v132 = a16;
  v130 = a6;
  v133 = a18;
  v19 = sub_13FD000(a1);
  if ( a16 )
    goto LABEL_2;
  v30 = sub_13FCB50(a1);
  v31 = v30;
  if ( !v30
    || *(_WORD *)(v30 + 18)
    || (v32 = sub_157EBA0(v30), v33 = v32, *(_BYTE *)(v32 + 16) != 26)
    || (*(_DWORD *)(v32 + 20) & 0xFFFFFFF) != 1
    || (v34 = sub_157F0B0(v31)) == 0
    || (v121 = v34, (v35 = sub_157EBA0(v34)) == 0)
    || (v117 = sub_15F4D60(v35), v36 = sub_157EBA0(v121), !v117) )
  {
LABEL_2:
    result = sub_1AF6260((__int64)&v125, v18, 0, a7, a8, a9, a10, v20, v21, a13, a14);
    if ( !(_BYTE)result )
      return result;
    goto LABEL_12;
  }
  v113 = v31;
  v37 = v36;
  v111 = (_QWORD *)v33;
  v38 = (__int64)(v18 + 7);
  v39 = v117;
  v118 = v18;
  v40 = 0;
  v109 = v38;
  while ( 1 )
  {
    v41 = sub_15F4DF0(v37, v40);
    if ( !sub_1377F70(v38, v41) )
      break;
    if ( v39 == ++v40 )
    {
      v18 = v118;
      goto LABEL_2;
    }
  }
  v42 = v111;
  v43 = v113;
  v18 = v118;
  v112 = sub_157EBA0(v121);
  if ( *(_BYTE *)(v112 + 16) != 26 )
    goto LABEL_2;
  v44 = (__int64)v118;
  v119 = (__int64)(v42 + 3);
  v114 = *(_QWORD **)(v113 + 48);
  v45 = sub_13F9E70(v44);
  v46 = v114;
  v110 = v45;
  if ( v42 + 3 != v114 )
  {
    v108 = v43;
    v115 = 0;
    v107 = v42;
    v47 = v46;
    do
    {
      v48 = 0;
      if ( v47 )
        v48 = (__int64)(v47 - 3);
      v49 = sub_14AF470(v48, 0, 0, 0);
      if ( !v49 )
        goto LABEL_2;
      if ( *(_BYTE *)(v48 + 16) == 78 )
      {
        v94 = *(_QWORD *)(v48 - 24);
        if ( *(_BYTE *)(v94 + 16)
          || (*(_BYTE *)(v94 + 33) & 0x20) == 0
          || (unsigned int)(*(_DWORD *)(v94 + 36) - 35) > 3 )
        {
          goto LABEL_2;
        }
      }
      else
      {
        switch ( *(_BYTE *)(v48 + 16) )
        {
          case '#':
          case '%':
          case '/':
          case '0':
          case '1':
          case '2':
          case '3':
          case '4':
            v50 = *(_BYTE *)(v48 + 23) & 0x40;
            goto LABEL_33;
          case '8':
            v51 = *(_DWORD *)(v48 + 20);
            v86 = 3LL * (v51 & 0xFFFFFFF);
            v50 = *(_BYTE *)(v48 + 23) & 0x40;
            if ( v50 )
            {
              v87 = *(_QWORD **)(v48 - 8);
              v88 = (__int64)(v87 + 3);
              v89 = (__int64)&v87[v86];
              if ( (_QWORD *)v89 != v87 + 3 )
                goto LABEL_91;
            }
            else
            {
              v88 = v48 + 24 - v86 * 8;
              if ( v48 == v88 )
                goto LABEL_35;
              v89 = v48;
              do
              {
LABEL_91:
                if ( *(_BYTE *)(*(_QWORD *)v88 + 16LL) != 13 )
                  goto LABEL_2;
                v88 += 24;
              }
              while ( v88 != v89 );
LABEL_33:
              if ( !v50 )
              {
                v51 = *(_DWORD *)(v48 + 20);
LABEL_35:
                v52 = (_QWORD *)(v48 - 24LL * (v51 & 0xFFFFFFF));
                goto LABEL_36;
              }
              v87 = *(_QWORD **)(v48 - 8);
            }
            v52 = v87;
LABEL_36:
            v53 = *v52;
            if ( *(_BYTE *)(*v52 + 16LL) <= 0x10u )
            {
              v53 = v52[3];
              if ( *(_BYTE *)(v53 + 16) <= 0x10u )
                goto LABEL_2;
            }
            if ( !v110 && *(_QWORD *)(v53 + 8) )
            {
              v106 = v47;
              v95 = *(_QWORD *)(v53 + 8);
              do
              {
                v96 = sub_1648700(v95);
                if ( !sub_1377F70(v109, v96[5]) )
                  goto LABEL_2;
                v95 = *(_QWORD *)(v95 + 8);
              }
              while ( v95 );
              v47 = v106;
            }
            if ( v115 )
              goto LABEL_2;
            v115 = v49;
            break;
          case '<':
          case '=':
          case '>':
            break;
          default:
            goto LABEL_2;
        }
      }
      v47 = (_QWORD *)v47[1];
    }
    while ( (_QWORD *)v119 != v47 );
    v43 = v108;
    v42 = v107;
  }
  v54 = *(_QWORD **)(v43 + 48);
  if ( (_QWORD *)v119 != v54 )
  {
    v55 = v112 + 24;
    if ( v119 != v112 + 24 )
    {
      if ( v121 + 40 != v43 + 40 )
      {
        v116 = *(_QWORD *)(v43 + 48);
        sub_157EA80(v121 + 40, v43 + 40, v116, v119);
        v55 = v112 + 24;
        v54 = (_QWORD *)v116;
      }
      if ( (_QWORD *)v119 != v54 )
      {
        v56 = v42[3] & 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)((*v54 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v119;
        v42[3] = v42[3] & 7LL | *v54 & 0xFFFFFFFFFFFFFFF8LL;
        v57 = *(_QWORD *)(v112 + 24);
        *(_QWORD *)(v56 + 8) = v55;
        *v54 = v57 & 0xFFFFFFFFFFFFFFF8LL | *v54 & 7LL;
        *(_QWORD *)((v57 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v54;
        *(_QWORD *)(v112 + 24) = v56 | *(_QWORD *)(v112 + 24) & 7LL;
      }
    }
  }
  v58 = *(v42 - 3);
  v59 = (_QWORD *)(v112 - 24LL * (*(_QWORD *)(v112 - 24) != v43) - 24);
  if ( *v59 )
  {
    v60 = v59[1];
    v61 = v59[2] & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v61 = v60;
    if ( v60 )
      *(_QWORD *)(v60 + 16) = v61 | *(_QWORD *)(v60 + 16) & 3LL;
  }
  *v59 = v58;
  if ( v58 )
  {
    v62 = *(_QWORD *)(v58 + 8);
    v59[1] = v62;
    if ( v62 )
      *(_QWORD *)(v62 + 16) = (unsigned __int64)(v59 + 1) | *(_QWORD *)(v62 + 16) & 3LL;
    v59[2] = (v58 + 8) | v59[2] & 3LL;
    *(_QWORD *)(v58 + 8) = v59;
  }
  sub_157F670(v43, v121);
  sub_15F20C0(v42);
  v63 = v126;
  v64 = *(unsigned int *)(v126 + 24);
  if ( (_DWORD)v64 )
  {
    v65 = *(_QWORD *)(v126 + 8);
    v66 = 1;
    v67 = (v64 - 1) & (((unsigned int)v43 >> 9) ^ ((unsigned int)v43 >> 4));
    v122 = (__int64 *)(v65 + 16LL * v67);
    v68 = *v122;
    if ( v43 == *v122 )
    {
LABEL_58:
      if ( v122 != (__int64 *)(v65 + 16 * v64) )
      {
        v69 = (__int64 *)v122[1];
        if ( v69 )
        {
          while ( 1 )
          {
            v70 = (char *)v69[5];
            v71 = (char *)v69[4];
            v72 = (v70 - v71) >> 5;
            if ( v72 <= 0 )
              break;
            v73 = &v71[32 * v72];
            while ( v43 != *(_QWORD *)v71 )
            {
              if ( v43 == *((_QWORD *)v71 + 1) )
              {
                v71 += 8;
                break;
              }
              if ( v43 == *((_QWORD *)v71 + 2) )
              {
                v71 += 16;
                break;
              }
              if ( v43 == *((_QWORD *)v71 + 3) )
              {
                v71 += 24;
                break;
              }
              v71 += 32;
              if ( v71 == v73 )
                goto LABEL_105;
            }
LABEL_67:
            if ( v71 + 8 != v70 )
            {
              memmove(v71, v71 + 8, v70 - (v71 + 8));
              v70 = (char *)v69[5];
            }
            v74 = (_QWORD *)v69[8];
            v69[5] = (__int64)(v70 - 8);
            if ( (_QWORD *)v69[9] == v74 )
            {
              v90 = &v74[*((unsigned int *)v69 + 21)];
              if ( v74 == v90 )
              {
LABEL_103:
                v74 = v90;
              }
              else
              {
                while ( v43 != *v74 )
                {
                  if ( v90 == ++v74 )
                    goto LABEL_103;
                }
              }
              goto LABEL_97;
            }
            v74 = sub_16CC9F0((__int64)(v69 + 7), v43);
            if ( v43 == *v74 )
            {
              v91 = v69[9];
              if ( v91 == v69[8] )
                v92 = *((unsigned int *)v69 + 21);
              else
                v92 = *((unsigned int *)v69 + 20);
              v90 = (_QWORD *)(v91 + 8 * v92);
LABEL_97:
              if ( v90 != v74 )
              {
                *v74 = -2;
                ++*((_DWORD *)v69 + 22);
              }
              goto LABEL_72;
            }
            v75 = v69[9];
            if ( v75 == v69[8] )
            {
              v74 = (_QWORD *)(v75 + 8LL * *((unsigned int *)v69 + 21));
              v90 = v74;
              goto LABEL_97;
            }
LABEL_72:
            v69 = (__int64 *)*v69;
            if ( !v69 )
              goto LABEL_73;
          }
          v73 = (char *)v69[4];
LABEL_105:
          v93 = v70 - v73;
          if ( v70 - v73 != 16 )
          {
            if ( v93 != 24 )
            {
              v71 = (char *)v69[5];
              if ( v93 != 8 )
                goto LABEL_67;
LABEL_108:
              if ( v43 != *(_QWORD *)v73 )
                v73 = (char *)v69[5];
              v71 = v73;
              goto LABEL_67;
            }
            v71 = v73;
            if ( v43 == *(_QWORD *)v73 )
              goto LABEL_67;
            v73 += 8;
          }
          v71 = v73;
          if ( v43 == *(_QWORD *)v73 )
            goto LABEL_67;
          v73 += 8;
          goto LABEL_108;
        }
LABEL_73:
        *v122 = -16;
        --*(_DWORD *)(v63 + 16);
        ++*(_DWORD *)(v63 + 20);
      }
    }
    else
    {
      while ( v68 != -8 )
      {
        v67 = (v64 - 1) & (v66 + v67);
        v122 = (__int64 *)(v65 + 16LL * v67);
        v68 = *v122;
        if ( v43 == *v122 )
          goto LABEL_58;
        ++v66;
      }
    }
  }
  v25 = v129;
  if ( v129 )
  {
    v76 = *(unsigned int *)(v129 + 48);
    if ( (_DWORD)v76 )
    {
      v27 = *(_QWORD *)(v129 + 32);
      v77 = v76 - 1;
      v78 = (v76 - 1) & (((unsigned int)v43 >> 9) ^ ((unsigned int)v43 >> 4));
      v79 = (__int64 *)(v27 + 16LL * v78);
      v80 = *v79;
      if ( v43 == *v79 )
      {
LABEL_77:
        if ( v79 != (__int64 *)(v27 + 16 * v76) )
        {
          v81 = v79[1];
          *(_BYTE *)(v129 + 72) = 0;
          v82 = *(_QWORD *)(v81 + 8);
          if ( !v82 )
            goto LABEL_133;
          v83 = *(char **)(v82 + 32);
          v24 = *(char **)(v82 + 24);
          v84 = (v83 - v24) >> 5;
          if ( v84 > 0 )
          {
            v85 = &v24[32 * v84];
            while ( v81 != *(_QWORD *)v24 )
            {
              if ( v81 == *((_QWORD *)v24 + 1) )
              {
                v24 += 8;
                goto LABEL_8;
              }
              if ( v81 == *((_QWORD *)v24 + 2) )
              {
                v24 += 16;
                goto LABEL_8;
              }
              if ( v81 == *((_QWORD *)v24 + 3) )
              {
                v24 += 24;
                goto LABEL_8;
              }
              v24 += 32;
              if ( v24 == v85 )
                goto LABEL_4;
            }
            goto LABEL_8;
          }
LABEL_4:
          v23 = v83 - v24;
          if ( v83 - v24 != 16 )
          {
            if ( v23 != 24 )
            {
              if ( v23 != 8 )
              {
LABEL_7:
                v24 = *(char **)(v82 + 32);
LABEL_8:
                if ( v24 + 8 != v83 )
                {
                  v120 = v25;
                  memmove(v24, v24 + 8, v83 - (v24 + 8));
                  v25 = v120;
                }
                *(_QWORD *)(v82 + 32) -= 8LL;
                v26 = *(_DWORD *)(v25 + 48);
                v27 = *(_QWORD *)(v25 + 32);
                if ( !v26 )
                  goto LABEL_11;
                v77 = v26 - 1;
LABEL_133:
                v97 = v77 & (((unsigned int)v43 >> 9) ^ ((unsigned int)v43 >> 4));
                v98 = 1;
                v99 = (__int64 *)(v27 + 16LL * v97);
                v100 = *v99;
                if ( v43 == *v99 )
                {
LABEL_134:
                  v101 = v99[1];
                  if ( v101 )
                  {
                    v102 = *(_QWORD *)(v101 + 24);
                    if ( v102 )
                    {
                      v123 = v25;
                      j_j___libc_free_0(v102, *(_QWORD *)(v101 + 40) - v102);
                      v25 = v123;
                    }
                    v124 = v25;
                    j_j___libc_free_0(v101, 56);
                    v25 = v124;
                  }
                  *v99 = -16;
                  --*(_DWORD *)(v25 + 40);
                  ++*(_DWORD *)(v25 + 44);
                }
                else
                {
                  while ( v100 != -8 )
                  {
                    v97 = v77 & (v98 + v97);
                    v99 = (__int64 *)(v27 + 16LL * v97);
                    v100 = *v99;
                    if ( v43 == *v99 )
                      goto LABEL_134;
                    ++v98;
                  }
                }
                goto LABEL_11;
              }
LABEL_156:
              if ( v81 == *(_QWORD *)v24 )
                goto LABEL_8;
              goto LABEL_7;
            }
            if ( v81 == *(_QWORD *)v24 )
              goto LABEL_8;
            v24 += 8;
          }
          if ( v81 == *(_QWORD *)v24 )
            goto LABEL_8;
          v24 += 8;
          goto LABEL_156;
        }
      }
      else
      {
        v103 = 1;
        while ( v80 != -8 )
        {
          v104 = v103 + 1;
          v105 = v77 & (v78 + v103);
          v78 = v105;
          v79 = (__int64 *)(v27 + 16 * v105);
          v80 = *v79;
          if ( v43 == *v79 )
            goto LABEL_77;
          v103 = v104;
        }
      }
    }
    *(_BYTE *)(v129 + 72) = 0;
    BUG();
  }
LABEL_11:
  sub_157F980(v43);
  sub_1AF6260((__int64)&v125, v18, 1, a7, a8, a9, a10, v28, v29, a13, a14);
LABEL_12:
  result = 1;
  if ( v19 )
  {
    sub_13FCC30((__int64)v18, v19);
    return 1;
  }
  return result;
}
