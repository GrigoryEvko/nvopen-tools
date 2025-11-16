// Function: sub_A12E50
// Address: 0xa12e50
//
_QWORD *__fastcall sub_A12E50(__int64 a1)
{
  unsigned int i; // r13d
  __int64 v2; // rsi
  _BYTE *v3; // rax
  _BYTE *v4; // r14
  _BYTE *v5; // r11
  unsigned __int8 v6; // al
  unsigned __int8 v7; // al
  _BYTE *v8; // rdx
  __int64 v9; // rcx
  unsigned __int8 v10; // al
  __int64 *v11; // rbx
  __int64 v12; // rdx
  unsigned int v13; // r8d
  __int64 *v14; // r13
  _QWORD *v15; // rdx
  unsigned int v16; // ecx
  _QWORD *v17; // rdx
  __int64 v18; // rax
  __int64 v19; // r12
  unsigned __int8 v20; // al
  int v21; // eax
  _QWORD *result; // rax
  __int64 v23; // rdx
  _QWORD *v24; // rax
  _QWORD *j; // rdx
  unsigned int v26; // ecx
  unsigned int v27; // eax
  int v28; // ebx
  _QWORD *v29; // rdi
  __int64 v30; // rdx
  _QWORD *v31; // r10
  int v32; // eax
  __int64 v33; // rax
  unsigned __int64 v34; // rdx
  unsigned __int8 v35; // al
  _BYTE *v36; // rax
  __int64 v37; // rcx
  unsigned __int8 v38; // al
  __int64 *v39; // rbx
  __int64 v40; // rsi
  __int64 *v41; // r15
  _QWORD *v42; // r10
  unsigned int v43; // edx
  __int64 *v44; // rsi
  __int64 v45; // rdi
  __int64 v46; // r12
  __int64 v47; // rax
  unsigned __int64 v48; // rdx
  __int64 *v49; // r13
  __int64 v50; // r15
  unsigned __int8 v51; // al
  _BYTE **v52; // rdx
  _BYTE *v53; // r12
  unsigned __int64 *v54; // rax
  unsigned __int64 *v55; // r14
  unsigned __int64 v56; // rcx
  unsigned __int64 v57; // rdx
  unsigned __int64 *v58; // rax
  unsigned __int64 *v59; // rax
  __int64 v60; // rdx
  _BOOL8 v61; // rdi
  __int64 v62; // rax
  __int64 v63; // r13
  __int64 v64; // r15
  unsigned __int8 v65; // al
  __int64 v66; // rax
  unsigned __int8 v67; // dl
  __int64 v68; // r14
  __int64 v69; // rax
  _BYTE *v70; // rcx
  signed __int64 v71; // r11
  __int64 v72; // rsi
  __int64 v73; // rax
  unsigned __int64 v74; // rsi
  int v75; // eax
  __int64 v76; // rcx
  __int64 v77; // r8
  const void *v78; // r9
  size_t v79; // r11
  int v80; // r14d
  _BYTE *v81; // rsi
  __int64 v82; // rdi
  __int64 v83; // rax
  __int64 v84; // rax
  __int64 v85; // rdx
  int v86; // esi
  int v87; // r9d
  unsigned int v88; // edx
  int v89; // r15d
  _QWORD *v90; // r9
  _QWORD *v91; // rdi
  unsigned int v92; // r15d
  int v93; // r9d
  __int64 v94; // rcx
  unsigned int v95; // kr00_4
  unsigned __int64 v96; // rax
  unsigned __int64 v97; // rdi
  __int64 v98; // rdx
  _QWORD *k; // rdx
  _BYTE *v100; // [rsp+8h] [rbp-158h]
  unsigned __int64 *v101; // [rsp+18h] [rbp-148h]
  __int64 v102; // [rsp+18h] [rbp-148h]
  __int64 v103; // [rsp+18h] [rbp-148h]
  unsigned __int64 *v104; // [rsp+18h] [rbp-148h]
  unsigned int v105; // [rsp+18h] [rbp-148h]
  unsigned int v106; // [rsp+18h] [rbp-148h]
  unsigned int v107; // [rsp+18h] [rbp-148h]
  int v108; // [rsp+20h] [rbp-140h]
  __int64 *v109; // [rsp+20h] [rbp-140h]
  unsigned __int64 v110; // [rsp+20h] [rbp-140h]
  const void *v111; // [rsp+20h] [rbp-140h]
  _BYTE *v112; // [rsp+20h] [rbp-140h]
  _BYTE *v113; // [rsp+20h] [rbp-140h]
  _BYTE *v114; // [rsp+20h] [rbp-140h]
  _QWORD *v115; // [rsp+30h] [rbp-130h]
  unsigned int v116; // [rsp+38h] [rbp-128h]
  int v117; // [rsp+3Ch] [rbp-124h]
  __int64 v119; // [rsp+48h] [rbp-118h]
  __int64 v120; // [rsp+50h] [rbp-110h] BYREF
  __int64 v121; // [rsp+58h] [rbp-108h]
  __int64 v122; // [rsp+60h] [rbp-100h]
  __int64 v123; // [rsp+68h] [rbp-F8h]
  __int64 *v124; // [rsp+70h] [rbp-F0h] BYREF
  __int64 v125; // [rsp+78h] [rbp-E8h]
  __int64 v126; // [rsp+80h] [rbp-E0h] BYREF
  int v127; // [rsp+88h] [rbp-D8h] BYREF
  unsigned __int64 *v128; // [rsp+90h] [rbp-D0h]
  int *v129; // [rsp+98h] [rbp-C8h]
  int *v130; // [rsp+A0h] [rbp-C0h]
  __int64 v131; // [rsp+A8h] [rbp-B8h]
  _BYTE *v132; // [rsp+B0h] [rbp-B0h] BYREF
  __int64 v133; // [rsp+B8h] [rbp-A8h]
  _BYTE v134[48]; // [rsp+C0h] [rbp-A0h] BYREF
  _BYTE *v135; // [rsp+F0h] [rbp-70h] BYREF
  __int64 v136; // [rsp+F8h] [rbp-68h]
  _BYTE v137[96]; // [rsp+100h] [rbp-60h] BYREF

  v119 = sub_BA8DC0(*(_QWORD *)(a1 + 256), "llvm.dbg.cu", 11);
  if ( v119 )
  {
    v117 = sub_B91A00(v119);
    if ( v117 )
    {
      for ( i = 0; v117 != i; ++i )
      {
        v2 = i;
        v3 = (_BYTE *)sub_B91A10(v119, i);
        v4 = v3;
        if ( *v3 != 17 )
          continue;
        v5 = v3 - 16;
        v6 = *(v3 - 16);
        if ( (v6 & 2) != 0 )
        {
          if ( !*(_QWORD *)(*((_QWORD *)v4 - 4) + 56LL) )
            continue;
        }
        else if ( !*(_QWORD *)&v5[-8 * ((v6 >> 2) & 0xF) + 56] )
        {
          continue;
        }
        v120 = 0;
        v121 = 0;
        v122 = 0;
        v123 = 0;
        v124 = &v126;
        v125 = 0;
        v7 = *(v4 - 16);
        if ( (v7 & 2) != 0 )
          v8 = (_BYTE *)*((_QWORD *)v4 - 4);
        else
          v8 = &v5[-8 * ((v7 >> 2) & 0xF)];
        v9 = *((_QWORD *)v8 + 7);
        v10 = *(_BYTE *)(v9 - 16);
        if ( (v10 & 2) != 0 )
        {
          v11 = *(__int64 **)(v9 - 32);
          v12 = *(unsigned int *)(v9 - 24);
        }
        else
        {
          v12 = (*(_WORD *)(v9 - 16) >> 6) & 0xF;
          v11 = (__int64 *)(v9 - 16 - 8LL * ((v10 >> 2) & 0xF));
        }
        if ( &v11[v12] == v11 )
          goto LABEL_45;
        v13 = i;
        v14 = &v11[v12];
        do
        {
          v19 = *v11;
          v20 = *(_BYTE *)(*v11 - 16);
          if ( (v20 & 2) != 0 )
            v15 = *(_QWORD **)(v19 - 32);
          else
            v15 = (_QWORD *)(v19 - 16 - 8LL * ((v20 >> 2) & 0xF));
          if ( !*v15 || (unsigned __int8)(*(_BYTE *)*v15 - 18) > 2u )
            goto LABEL_20;
          v2 = (unsigned int)v123;
          if ( (_DWORD)v123 )
          {
            v16 = (v123 - 1) & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
            v17 = (_QWORD *)(v121 + 8LL * v16);
            v18 = *v17;
            if ( v19 == *v17 )
              goto LABEL_20;
            v108 = 1;
            v31 = 0;
            while ( v18 != -4096 )
            {
              if ( v18 != -8192 || v31 )
                v17 = v31;
              v16 = (v123 - 1) & (v108 + v16);
              v18 = *(_QWORD *)(v121 + 8LL * v16);
              if ( v19 == v18 )
                goto LABEL_20;
              ++v108;
              v31 = v17;
              v17 = (_QWORD *)(v121 + 8LL * v16);
            }
            if ( !v31 )
              v31 = v17;
            ++v120;
            v32 = v122 + 1;
            if ( 4 * ((int)v122 + 1) < (unsigned int)(3 * v123) )
            {
              if ( (int)v123 - HIDWORD(v122) - v32 <= (unsigned int)v123 >> 3 )
              {
                v107 = v13;
                v114 = v5;
                sub_9C0C30((__int64)&v120, v123);
                if ( !(_DWORD)v123 )
                {
LABEL_165:
                  LODWORD(v122) = v122 + 1;
                  BUG();
                }
                v2 = v121;
                v91 = 0;
                v5 = v114;
                v92 = (v123 - 1) & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
                v13 = v107;
                v93 = 1;
                v31 = (_QWORD *)(v121 + 8LL * v92);
                v94 = *v31;
                v32 = v122 + 1;
                if ( v19 != *v31 )
                {
                  while ( v94 != -4096 )
                  {
                    if ( !v91 && v94 == -8192 )
                      v91 = v31;
                    v92 = (v123 - 1) & (v93 + v92);
                    v31 = (_QWORD *)(v121 + 8LL * v92);
                    v94 = *v31;
                    if ( v19 == *v31 )
                      goto LABEL_54;
                    ++v93;
                  }
                  if ( v91 )
                    v31 = v91;
                }
              }
              goto LABEL_54;
            }
          }
          else
          {
            ++v120;
          }
          v105 = v13;
          v112 = v5;
          sub_9C0C30((__int64)&v120, 2 * v123);
          if ( !(_DWORD)v123 )
            goto LABEL_165;
          v5 = v112;
          v13 = v105;
          v88 = (v123 - 1) & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
          v31 = (_QWORD *)(v121 + 8LL * v88);
          v2 = *v31;
          v32 = v122 + 1;
          if ( v19 != *v31 )
          {
            v89 = 1;
            v90 = 0;
            while ( v2 != -4096 )
            {
              if ( !v90 && v2 == -8192 )
                v90 = v31;
              v88 = (v123 - 1) & (v89 + v88);
              v31 = (_QWORD *)(v121 + 8LL * v88);
              v2 = *v31;
              if ( v19 == *v31 )
                goto LABEL_54;
              ++v89;
            }
            if ( v90 )
              v31 = v90;
          }
LABEL_54:
          LODWORD(v122) = v32;
          if ( *v31 != -4096 )
            --HIDWORD(v122);
          *v31 = v19;
          v33 = (unsigned int)v125;
          v34 = (unsigned int)v125 + 1LL;
          if ( v34 > HIDWORD(v125) )
          {
            v2 = (__int64)&v126;
            v106 = v13;
            v113 = v5;
            sub_C8D5F0(&v124, &v126, v34, 8);
            v33 = (unsigned int)v125;
            v13 = v106;
            v5 = v113;
          }
          v124[v33] = v19;
          LODWORD(v125) = v125 + 1;
LABEL_20:
          ++v11;
        }
        while ( v14 != v11 );
        v30 = (unsigned int)v125;
        i = v13;
        if ( !(_DWORD)v125 )
          goto LABEL_43;
        v132 = v134;
        v133 = 0x600000000LL;
        v35 = *(v4 - 16);
        if ( (v35 & 2) != 0 )
          v36 = (_BYTE *)*((_QWORD *)v4 - 4);
        else
          v36 = &v5[-8 * ((v35 >> 2) & 0xF)];
        v37 = *((_QWORD *)v36 + 7);
        v38 = *(_BYTE *)(v37 - 16);
        if ( (v38 & 2) != 0 )
        {
          v39 = *(__int64 **)(v37 - 32);
          v40 = *(unsigned int *)(v37 - 24);
        }
        else
        {
          v40 = (*(_WORD *)(v37 - 16) >> 6) & 0xF;
          v39 = (__int64 *)(v37 - 16 - 8LL * ((v38 >> 2) & 0xF));
        }
        v41 = &v39[v40];
        if ( v41 == v39 )
          goto LABEL_74;
        v42 = &v132;
        while ( 2 )
        {
          while ( 1 )
          {
            v46 = *v39;
            if ( !(_DWORD)v123 )
              break;
            v43 = (v123 - 1) & (((unsigned int)v46 >> 9) ^ ((unsigned int)v46 >> 4));
            v44 = (__int64 *)(v121 + 8LL * v43);
            v45 = *v44;
            if ( v46 != *v44 )
            {
              v86 = 1;
              while ( v45 != -4096 )
              {
                v87 = v86 + 1;
                v43 = (v123 - 1) & (v86 + v43);
                v44 = (__int64 *)(v121 + 8LL * v43);
                v45 = *v44;
                if ( v46 == *v44 )
                  goto LABEL_67;
                v86 = v87;
              }
              break;
            }
LABEL_67:
            if ( v44 == (__int64 *)(v121 + 8LL * (unsigned int)v123) )
              break;
            if ( v41 == ++v39 )
              goto LABEL_73;
          }
          v47 = (unsigned int)v133;
          v48 = (unsigned int)v133 + 1LL;
          if ( v48 > HIDWORD(v133) )
          {
            v115 = v42;
            sub_C8D5F0(v42, v134, v48, 8);
            v47 = (unsigned int)v133;
            v42 = v115;
          }
          ++v39;
          *(_QWORD *)&v132[8 * v47] = v46;
          LODWORD(v133) = v133 + 1;
          if ( v41 != v39 )
            continue;
          break;
        }
LABEL_73:
        v30 = (unsigned int)v125;
LABEL_74:
        v127 = 0;
        v129 = &v127;
        v130 = &v127;
        v128 = 0;
        v131 = 0;
        v109 = &v124[v30];
        if ( v109 != v124 )
        {
          v100 = v4;
          v116 = i;
          v49 = v124;
          do
          {
            v50 = *v49;
            v51 = *(_BYTE *)(*v49 - 16);
            if ( (v51 & 2) != 0 )
              v52 = *(_BYTE ***)(v50 - 32);
            else
              v52 = (_BYTE **)(v50 - 16 - 8LL * ((v51 >> 2) & 0xF));
            v53 = sub_A08FE0(a1, *v52);
            if ( v53 )
            {
              v54 = v128;
              v55 = (unsigned __int64 *)&v127;
              if ( !v128 )
                goto LABEL_86;
              do
              {
                while ( 1 )
                {
                  v56 = v54[2];
                  v57 = v54[3];
                  if ( v54[4] >= (unsigned __int64)v53 )
                    break;
                  v54 = (unsigned __int64 *)v54[3];
                  if ( !v57 )
                    goto LABEL_84;
                }
                v55 = v54;
                v54 = (unsigned __int64 *)v54[2];
              }
              while ( v56 );
LABEL_84:
              if ( v55 == (unsigned __int64 *)&v127 || v55[4] > (unsigned __int64)v53 )
              {
LABEL_86:
                v101 = v55;
                v58 = (unsigned __int64 *)sub_22077B0(104);
                v58[4] = (unsigned __int64)v53;
                v55 = v58;
                v58[5] = (unsigned __int64)(v58 + 7);
                v58[6] = 0x600000000LL;
                v59 = sub_A059A0(&v126, v101, v58 + 4);
                if ( v60 )
                {
                  v61 = &v127 == (int *)v60 || v59 || (unsigned __int64)v53 < *(_QWORD *)(v60 + 32);
                  sub_220F040(v61, v55, v60, &v127);
                  ++v131;
                }
                else
                {
                  v104 = v59;
                  j_j___libc_free_0(v55, 104);
                  v55 = v104;
                }
              }
              v62 = *((unsigned int *)v55 + 12);
              if ( v62 + 1 > (unsigned __int64)*((unsigned int *)v55 + 13) )
              {
                sub_C8D5F0(v55 + 5, v55 + 7, v62 + 1, 8);
                v62 = *((unsigned int *)v55 + 12);
              }
              *(_QWORD *)(v55[5] + 8 * v62) = v50;
              ++*((_DWORD *)v55 + 12);
            }
            ++v49;
          }
          while ( v109 != v49 );
          v4 = v100;
          i = v116;
          if ( v129 != &v127 )
          {
            v63 = (__int64)v129;
            while ( 1 )
            {
              v64 = *(_QWORD *)(v63 + 32);
              v65 = *(_BYTE *)(v64 - 16);
              if ( (v65 & 2) != 0 )
              {
                v66 = *(_QWORD *)(*(_QWORD *)(v64 - 32) + 56LL);
                if ( !v66 )
                  goto LABEL_118;
              }
              else
              {
                v66 = *(_QWORD *)(v64 - 16 - 8LL * ((v65 >> 2) & 0xF) + 56);
                if ( !v66 )
                {
LABEL_118:
                  v135 = v137;
                  HIDWORD(v136) = 6;
                  goto LABEL_119;
                }
              }
              v67 = *(_BYTE *)(v66 - 16);
              if ( (v67 & 2) != 0 )
              {
                v68 = *(_QWORD *)(v66 - 32);
                v69 = v68 + 8LL * *(unsigned int *)(v66 - 24);
              }
              else
              {
                v85 = (v67 >> 2) & 0xF;
                v68 = v66 - 16 - 8 * v85;
                v69 = v66 - 16 + 8 * (((*(_WORD *)(v66 - 16) >> 6) & 0xF) - v85);
              }
              v135 = v137;
              v136 = 0x600000000LL;
              if ( v69 != v68 )
              {
                v70 = v137;
                v71 = ((unsigned __int64)(v69 - 8 - v68) >> 3) + 1;
                if ( v71 > 6 )
                {
                  v102 = v69;
                  v110 = ((unsigned __int64)(v69 - 8 - v68) >> 3) + 1;
                  sub_C8D5F0(&v135, v137, v110, 8);
                  LODWORD(v71) = v110;
                  v69 = v102;
                  v70 = &v135[8 * (unsigned int)v136];
                }
                v72 = v69 - v68;
                v73 = 0;
                do
                {
                  *(_QWORD *)&v70[v73] = *(_QWORD *)(v68 + v73);
                  v73 += 8;
                }
                while ( v72 != v73 );
                v74 = HIDWORD(v136);
                v75 = v71 + v136;
                v76 = (unsigned int)(v71 + v136);
                goto LABEL_107;
              }
LABEL_119:
              v76 = 0;
              v74 = 6;
              v75 = 0;
LABEL_107:
              v77 = *(unsigned int *)(v63 + 48);
              v78 = *(const void **)(v63 + 40);
              LODWORD(v136) = v75;
              v79 = 8 * v77;
              v80 = v77;
              if ( v77 + v76 > v74 )
              {
                v103 = 8 * v77;
                v111 = v78;
                sub_C8D5F0(&v135, v137, v77 + v76, 8);
                v76 = (unsigned int)v136;
                v79 = v103;
                v78 = v111;
              }
              v81 = v135;
              if ( v79 )
              {
                memcpy(&v135[8 * v76], v78, v79);
                v81 = v135;
                LODWORD(v76) = v136;
              }
              v82 = *(_QWORD *)(a1 + 248);
              LODWORD(v136) = v80 + v76;
              v83 = sub_B9C770(v82, v81, (unsigned int)(v80 + v76), 0, 1);
              sub_BA6610(v64, 7, v83);
              if ( v135 != v137 )
                _libc_free(v135, 7);
              v63 = sub_220EEE0(v63);
              if ( (int *)v63 == &v127 )
              {
                v4 = v100;
                i = v116;
                break;
              }
            }
          }
        }
        v84 = sub_B9C770(*(_QWORD *)(a1 + 248), v132, (unsigned int)v133, 0, 1);
        v2 = 7;
        sub_BA6610(v4, 7, v84);
        sub_A01E10(v128, 7);
        if ( v132 != v134 )
          _libc_free(v132, 7);
LABEL_43:
        if ( v124 != &v126 )
          _libc_free(v124, v2);
LABEL_45:
        sub_C7D6A0(v121, 8LL * (unsigned int)v123, 8);
      }
    }
  }
  ++*(_QWORD *)(a1 + 1104);
  v21 = *(_DWORD *)(a1 + 1120);
  if ( !v21 )
  {
    result = (_QWORD *)a1;
    if ( !*(_DWORD *)(a1 + 1124) )
      return result;
    v23 = *(unsigned int *)(a1 + 1128);
    if ( (unsigned int)v23 > 0x40 )
    {
      result = (_QWORD *)sub_C7D6A0(*(_QWORD *)(a1 + 1112), 16LL * (unsigned int)v23, 8);
      *(_QWORD *)(a1 + 1112) = 0;
      *(_QWORD *)(a1 + 1120) = 0;
      *(_DWORD *)(a1 + 1128) = 0;
      return result;
    }
    goto LABEL_26;
  }
  v26 = 4 * v21;
  v23 = *(unsigned int *)(a1 + 1128);
  if ( (unsigned int)(4 * v21) < 0x40 )
    v26 = 64;
  if ( (unsigned int)v23 <= v26 )
  {
LABEL_26:
    v24 = *(_QWORD **)(a1 + 1112);
    for ( j = &v24[2 * v23]; j != v24; v24 += 2 )
      *v24 = -4096;
    result = (_QWORD *)a1;
    *(_QWORD *)(a1 + 1120) = 0;
    return result;
  }
  v27 = v21 - 1;
  if ( v27 )
  {
    _BitScanReverse(&v27, v27);
    v28 = 1 << (33 - (v27 ^ 0x1F));
    v29 = *(_QWORD **)(a1 + 1112);
    if ( v28 < 64 )
      v28 = 64;
    if ( (_DWORD)v23 == v28 )
    {
      *(_QWORD *)(a1 + 1120) = 0;
      result = &v29[2 * (unsigned int)v23];
      do
      {
        if ( v29 )
          *v29 = -4096;
        v29 += 2;
      }
      while ( result != v29 );
      return result;
    }
  }
  else
  {
    v28 = 64;
    v29 = *(_QWORD **)(a1 + 1112);
  }
  sub_C7D6A0(v29, 16LL * (unsigned int)v23, 8);
  v95 = 4 * v28;
  v96 = ((((((((v95 / 3 + 1) | ((unsigned __int64)(v95 / 3 + 1) >> 1)) >> 2)
           | (v95 / 3 + 1)
           | ((unsigned __int64)(v95 / 3 + 1) >> 1)) >> 4)
         | (((v95 / 3 + 1) | ((unsigned __int64)(v95 / 3 + 1) >> 1)) >> 2)
         | (v95 / 3 + 1)
         | ((unsigned __int64)(v95 / 3 + 1) >> 1)) >> 8)
       | (((((v95 / 3 + 1) | ((unsigned __int64)(v95 / 3 + 1) >> 1)) >> 2)
         | (v95 / 3 + 1)
         | ((unsigned __int64)(v95 / 3 + 1) >> 1)) >> 4)
       | (((v95 / 3 + 1) | ((unsigned __int64)(v95 / 3 + 1) >> 1)) >> 2)
       | (v95 / 3 + 1)
       | ((unsigned __int64)(v95 / 3 + 1) >> 1)) >> 16;
  v97 = (v96
       | (((((((v95 / 3 + 1) | ((unsigned __int64)(v95 / 3 + 1) >> 1)) >> 2)
           | (v95 / 3 + 1)
           | ((unsigned __int64)(v95 / 3 + 1) >> 1)) >> 4)
         | (((v95 / 3 + 1) | ((unsigned __int64)(v95 / 3 + 1) >> 1)) >> 2)
         | (v95 / 3 + 1)
         | ((unsigned __int64)(v95 / 3 + 1) >> 1)) >> 8)
       | (((((v95 / 3 + 1) | ((unsigned __int64)(v95 / 3 + 1) >> 1)) >> 2)
         | (v95 / 3 + 1)
         | ((unsigned __int64)(v95 / 3 + 1) >> 1)) >> 4)
       | (((v95 / 3 + 1) | ((unsigned __int64)(v95 / 3 + 1) >> 1)) >> 2)
       | (v95 / 3 + 1)
       | ((unsigned __int64)(v95 / 3 + 1) >> 1))
      + 1;
  *(_DWORD *)(a1 + 1128) = v97;
  result = (_QWORD *)sub_C7D670(16 * v97, 8);
  v98 = *(unsigned int *)(a1 + 1128);
  *(_QWORD *)(a1 + 1120) = 0;
  *(_QWORD *)(a1 + 1112) = result;
  for ( k = &result[2 * v98]; k != result; result += 2 )
  {
    if ( result )
      *result = -4096;
  }
  return result;
}
