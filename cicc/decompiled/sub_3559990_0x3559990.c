// Function: sub_3559990
// Address: 0x3559990
//
__int64 __fastcall sub_3559990(__int64 a1, __int64 a2)
{
  __int64 *v2; // rax
  _QWORD *v3; // rax
  __int64 **v4; // rax
  __int64 v5; // rax
  __int64 *v6; // r15
  unsigned __int64 v7; // rsi
  int v8; // eax
  __int64 v9; // rdx
  _QWORD *v10; // rax
  _QWORD *v11; // rdx
  __int64 v12; // rbx
  __int64 *v13; // r12
  unsigned __int64 v14; // rdi
  __int64 v15; // rbx
  __int64 *v16; // r12
  unsigned __int64 v17; // rdi
  int v19; // eax
  __int64 v20; // rdx
  __int64 v21; // rcx
  _QWORD *v22; // r8
  int v23; // eax
  int v24; // eax
  __int64 v25; // rax
  __int64 v26; // r8
  unsigned __int64 v27; // r9
  __int64 v28; // rdx
  __int64 v29; // rax
  __int64 *v30; // rdi
  __int64 v31; // rdi
  unsigned int v32; // edx
  unsigned __int64 v33; // rax
  __int64 v34; // r10
  __int64 *v35; // rax
  __int64 **v36; // r13
  _QWORD *v37; // rdi
  __int64 *v38; // rax
  unsigned int v39; // eax
  __int64 v40; // rdx
  __int64 v41; // rdx
  _BYTE *v42; // r14
  __int64 v43; // r8
  _BYTE *v44; // r13
  unsigned __int64 v45; // r12
  _QWORD *v46; // rax
  _QWORD *v47; // rdx
  __int64 v48; // rax
  __int64 v49; // rcx
  unsigned __int64 v50; // rdx
  unsigned __int64 *v51; // rdx
  _QWORD *v52; // rax
  char v53; // bl
  __int64 v54; // rbx
  __int64 v55; // rcx
  __int64 v56; // rdx
  __int64 *v57; // rax
  int v58; // edx
  __int64 v59; // rcx
  __int64 *v60; // rdx
  int v61; // ecx
  __int64 v62; // rsi
  unsigned __int64 v63; // rsi
  __int64 v64; // r8
  __int64 v65; // rdi
  __int64 v66; // rcx
  __int64 v67; // rdx
  __int64 v68; // rdx
  __int64 v69; // r8
  __int64 v70; // rdi
  __int64 v71; // rsi
  __int64 v72; // rcx
  unsigned __int64 v73; // rax
  unsigned int v74; // ecx
  _QWORD *v75; // rdi
  unsigned int v76; // eax
  int v77; // r12d
  unsigned int v78; // eax
  _QWORD *v79; // rax
  _QWORD *i; // rdx
  __int64 v81; // r8
  __int64 v82; // r9
  __int64 v83; // rdx
  __int64 v84; // rax
  __int64 *v85; // r12
  __int64 v86; // r8
  int v87; // r10d
  __int64 v88; // r9
  unsigned int v89; // edx
  __int64 v90; // r13
  __int64 v91; // rcx
  __int64 v92; // rdx
  __int64 *v93; // rbx
  __int64 v94; // rax
  __int64 v95; // r14
  __int64 *v96; // rbx
  __int64 v97; // rcx
  int v98; // edx
  unsigned __int64 v99; // rcx
  unsigned __int64 v100; // rsi
  __int64 v101; // rsi
  __int64 *v102; // rcx
  __int64 v103; // rdx
  int j; // eax
  int v105; // eax
  __int64 v106; // rax
  __int64 *v107; // [rsp+8h] [rbp-1E8h]
  unsigned __int64 v108; // [rsp+10h] [rbp-1E0h]
  __int64 *v109; // [rsp+18h] [rbp-1D8h]
  __int64 *v110; // [rsp+20h] [rbp-1D0h]
  __int64 **v113; // [rsp+50h] [rbp-1A0h]
  __int64 *v114; // [rsp+60h] [rbp-190h]
  __int64 v115; // [rsp+68h] [rbp-188h]
  __int64 v116; // [rsp+68h] [rbp-188h]
  __int64 **v117; // [rsp+78h] [rbp-178h]
  __int64 *v118; // [rsp+78h] [rbp-178h]
  char v119; // [rsp+86h] [rbp-16Ah] BYREF
  char v120; // [rsp+87h] [rbp-169h] BYREF
  __int64 v121; // [rsp+88h] [rbp-168h] BYREF
  char *v122; // [rsp+90h] [rbp-160h] BYREF
  int v123; // [rsp+98h] [rbp-158h] BYREF
  __int64 v124; // [rsp+A0h] [rbp-150h] BYREF
  int v125; // [rsp+A8h] [rbp-148h]
  __int64 v126; // [rsp+B0h] [rbp-140h] BYREF
  unsigned __int64 v127; // [rsp+B8h] [rbp-138h]
  __int64 v128; // [rsp+C0h] [rbp-130h]
  unsigned int v129; // [rsp+C8h] [rbp-128h]
  __int64 *v130; // [rsp+D0h] [rbp-120h] BYREF
  __int64 v131; // [rsp+D8h] [rbp-118h]
  __int64 *v132; // [rsp+E0h] [rbp-110h] BYREF
  __int64 v133; // [rsp+E8h] [rbp-108h]
  _BYTE v134[32]; // [rsp+F0h] [rbp-100h] BYREF
  _QWORD *v135; // [rsp+110h] [rbp-E0h] BYREF
  unsigned __int64 v136; // [rsp+118h] [rbp-D8h]
  _QWORD v137[8]; // [rsp+120h] [rbp-D0h] BYREF
  __int64 v138; // [rsp+160h] [rbp-90h] BYREF
  unsigned __int64 v139; // [rsp+168h] [rbp-88h]
  __int64 v140; // [rsp+170h] [rbp-80h]
  __int64 v141; // [rsp+178h] [rbp-78h] BYREF
  _QWORD v142[14]; // [rsp+180h] [rbp-70h] BYREF

  v130 = (__int64 *)&v132;
  v2 = *(__int64 **)(a1 + 32);
  v126 = 0;
  v127 = 0;
  v128 = 0;
  v129 = 0;
  v131 = 0;
  v3 = (_QWORD *)sub_B2BE50(*v2);
  v4 = (__int64 **)sub_BCB120(v3);
  v5 = sub_ACA8A0(v4);
  v6 = *(__int64 **)(a1 + 48);
  v7 = (unsigned __int64)v137;
  v108 = v5;
  v109 = *(__int64 **)(a1 + 56);
  while ( v109 != v6 )
  {
    v8 = *(_DWORD *)(*v6 + 44);
    v115 = *v6;
    if ( (v8 & 4) != 0 || (v8 & 8) == 0 )
    {
      if ( (*(_QWORD *)(*(_QWORD *)(v115 + 16) + 24LL) & 0x80u) != 0LL )
        goto LABEL_5;
    }
    else
    {
      v7 = 128;
      if ( sub_2E88A90(*v6, 128, 1) )
        goto LABEL_5;
    }
    v19 = *(_DWORD *)(v115 + 44);
    if ( (v19 & 4) != 0 || (v19 & 8) == 0 )
    {
      if ( (*(_QWORD *)(*(_QWORD *)(v115 + 16) + 24LL) & 0x200000LL) == 0 )
        goto LABEL_29;
    }
    else
    {
      v7 = 0x200000;
      if ( !sub_2E88A90(v115, 0x200000, 1) )
        goto LABEL_29;
    }
    if ( (*(_BYTE *)(v115 + 45) & 0x40) == 0 )
      goto LABEL_5;
LABEL_29:
    if ( sub_2E8B090(v115)
      || (unsigned __int8)sub_2E8B100(v115, v7, v20, v21, v22)
      && (((v7 = v115, (unsigned int)*(unsigned __int16 *)(v115 + 68) - 1 > 1)
        || (*(_BYTE *)(*(_QWORD *)(v115 + 32) + 64LL) & 8) == 0)
       && ((v105 = *(_DWORD *)(v115 + 44), (v105 & 4) != 0) || (v105 & 8) == 0
         ? (v106 = (*(_QWORD *)(*(_QWORD *)(v115 + 16) + 24LL) >> 19) & 1LL)
         : (v7 = 0x80000, LOBYTE(v106) = sub_2E88A90(v115, 0x80000, 1)),
           !(_BYTE)v106)
       || !(unsigned __int8)sub_2E8AED0(v115)) )
    {
LABEL_5:
      ++v126;
      if ( !(_DWORD)v128 )
      {
        if ( !HIDWORD(v128) )
          goto LABEL_11;
        v9 = v129;
        if ( v129 <= 0x40 )
          goto LABEL_8;
        v7 = 16LL * v129;
        sub_C7D6A0(v127, v7, 8);
        v129 = 0;
LABEL_188:
        v127 = 0;
LABEL_10:
        v128 = 0;
        goto LABEL_11;
      }
      v74 = 4 * v128;
      v9 = v129;
      if ( (unsigned int)(4 * v128) < 0x40 )
        v74 = 64;
      if ( v129 <= v74 )
      {
LABEL_8:
        v10 = (_QWORD *)v127;
        v11 = (_QWORD *)(v127 + 16 * v9);
        if ( (_QWORD *)v127 != v11 )
        {
          do
          {
            *v10 = -4096;
            v10 += 2;
          }
          while ( v11 != v10 );
        }
        goto LABEL_10;
      }
      v75 = (_QWORD *)v127;
      v7 = 16LL * v129;
      if ( (_DWORD)v128 == 1 )
      {
        v77 = 64;
      }
      else
      {
        _BitScanReverse(&v76, v128 - 1);
        v77 = 1 << (33 - (v76 ^ 0x1F));
        if ( v77 < 64 )
          v77 = 64;
        if ( v77 == v129 )
        {
          v128 = 0;
          v7 += v127;
          do
          {
            if ( v75 )
              *v75 = -4096;
            v75 += 2;
          }
          while ( (_QWORD *)v7 != v75 );
          goto LABEL_11;
        }
      }
      sub_C7D6A0(v127, v7, 8);
      v78 = sub_3540050(v77);
      v129 = v78;
      if ( !v78 )
        goto LABEL_188;
      v7 = 8;
      v79 = (_QWORD *)sub_C7D670(16LL * v78, 8);
      v128 = 0;
      v127 = (unsigned __int64)v79;
      for ( i = &v79[2 * v129]; i != v79; v79 += 2 )
      {
        if ( v79 )
          *v79 = -4096;
      }
LABEL_11:
      v12 = (__int64)v130;
      v13 = &v130[7 * (unsigned int)v131];
      while ( (__int64 *)v12 != v13 )
      {
        while ( 1 )
        {
          v13 -= 7;
          v14 = v13[1];
          if ( (__int64 *)v14 == v13 + 3 )
            break;
          _libc_free(v14);
          if ( (__int64 *)v12 == v13 )
            goto LABEL_15;
        }
      }
LABEL_15:
      LODWORD(v131) = 0;
      goto LABEL_16;
    }
    if ( (unsigned int)*(unsigned __int16 *)(v115 + 68) - 1 <= 1 && (*(_BYTE *)(*(_QWORD *)(v115 + 32) + 64LL) & 8) != 0 )
    {
LABEL_121:
      v132 = (__int64 *)v134;
      v133 = 0x400000000LL;
      sub_2E864A0(v115);
      if ( v83 == 1 )
        sub_353FFA0(v115, (__int64)&v132);
      v84 = (unsigned int)v133;
      if ( !(_DWORD)v133 )
      {
        if ( !HIDWORD(v133) )
        {
          sub_C8D5F0((__int64)&v132, v134, 1u, 8u, v81, v82);
          v84 = (unsigned int)v133;
        }
        v132[v84] = v108;
        v84 = (unsigned int)(v133 + 1);
        LODWORD(v133) = v133 + 1;
      }
      v85 = v132;
      v118 = &v132[v84];
      if ( v132 == v118 )
      {
LABEL_148:
        v7 = (unsigned __int64)v134;
        if ( v118 != (__int64 *)v134 )
          _libc_free((unsigned __int64)v118);
        goto LABEL_16;
      }
      while ( 2 )
      {
        v95 = *v85;
        v125 = 0;
        v124 = v95;
        if ( !v129 )
        {
          ++v126;
          v138 = 0;
          break;
        }
        v86 = v129 - 1;
        v87 = 1;
        v88 = 0;
        v89 = v86 & (((unsigned int)v95 >> 9) ^ ((unsigned int)v95 >> 4));
        v90 = v127 + 16LL * v89;
        v91 = *(_QWORD *)v90;
        if ( v95 == *(_QWORD *)v90 )
        {
LABEL_127:
          v92 = *(unsigned int *)(v90 + 8);
          goto LABEL_128;
        }
        while ( v91 != -4096 )
        {
          if ( !v88 && v91 == -8192 )
            v88 = v90;
          v89 = v86 & (v87 + v89);
          v90 = v127 + 16LL * v89;
          v91 = *(_QWORD *)v90;
          if ( v95 == *(_QWORD *)v90 )
            goto LABEL_127;
          ++v87;
        }
        if ( v88 )
          v90 = v88;
        ++v126;
        v98 = v128 + 1;
        v138 = v90;
        if ( 4 * ((int)v128 + 1) < 3 * v129 )
        {
          v97 = v95;
          v86 = v129 >> 3;
          v96 = &v138;
          if ( v129 - HIDWORD(v128) - v98 <= (unsigned int)v86 )
          {
            sub_A429D0((__int64)&v126, v129);
LABEL_134:
            sub_A56BF0((__int64)&v126, &v124, &v138);
            v97 = v124;
            v90 = v138;
            v98 = v128 + 1;
          }
          LODWORD(v128) = v98;
          if ( *(_QWORD *)v90 != -4096 )
            --HIDWORD(v128);
          *(_QWORD *)v90 = v97;
          *(_DWORD *)(v90 + 8) = v125;
          v99 = (unsigned int)v131;
          v135 = v137;
          v138 = v95;
          v100 = (unsigned int)v131 + 1LL;
          v92 = (unsigned int)v131;
          v136 = 0x400000000LL;
          v139 = (unsigned __int64)&v141;
          v140 = 0x400000000LL;
          if ( v100 > HIDWORD(v131) )
          {
            if ( v130 > &v138
              || (v116 = (__int64)v130,
                  v99 = (unsigned __int64)&v130[7 * (unsigned int)v131],
                  (unsigned __int64)&v138 >= v99) )
            {
              sub_354D260((__int64)&v130, v100, (__int64)v130, v99, v86, v88);
              v99 = (unsigned int)v131;
              v101 = (__int64)v130;
              v92 = (unsigned int)v131;
            }
            else
            {
              sub_354D260((__int64)&v130, v100, (__int64)v130, v99, v86, v88);
              v101 = (__int64)v130;
              v99 = (unsigned int)v131;
              v96 = (__int64 *)((char *)&v138 + (_QWORD)v130 - v116);
              v92 = (unsigned int)v131;
            }
          }
          else
          {
            v101 = (__int64)v130;
          }
          v102 = (__int64 *)(v101 + 56 * v99);
          if ( v102 )
          {
            v103 = *v96;
            v102[2] = 0x400000000LL;
            *v102 = v103;
            v102[1] = (__int64)(v102 + 3);
            if ( *((_DWORD *)v96 + 4) )
              sub_353DE10((__int64)(v102 + 1), (char **)v96 + 1, (__int64)(v102 + 3), (__int64)v102, v86, v88);
            v92 = (unsigned int)v131;
          }
          LODWORD(v131) = v92 + 1;
          if ( (__int64 *)v139 != &v141 )
          {
            _libc_free(v139);
            v92 = (unsigned int)(v131 - 1);
          }
          *(_DWORD *)(v90 + 8) = v92;
LABEL_128:
          v93 = &v130[7 * v92];
          v94 = *((unsigned int *)v93 + 4);
          if ( v94 + 1 > (unsigned __int64)*((unsigned int *)v93 + 5) )
          {
            sub_C8D5F0((__int64)(v93 + 1), v93 + 3, v94 + 1, 8u, v86, v88);
            v94 = *((unsigned int *)v93 + 4);
          }
          ++v85;
          *(_QWORD *)(v93[1] + 8 * v94) = v6;
          ++*((_DWORD *)v93 + 4);
          if ( v118 == v85 )
          {
            v118 = v132;
            goto LABEL_148;
          }
          continue;
        }
        break;
      }
      v96 = &v138;
      sub_A429D0((__int64)&v126, 2 * v129);
      goto LABEL_134;
    }
    v23 = *(_DWORD *)(v115 + 44);
    if ( (v23 & 4) != 0 || (v23 & 8) == 0 )
    {
      if ( (*(_QWORD *)(*(_QWORD *)(v115 + 16) + 24LL) & 0x80000LL) != 0 )
        goto LABEL_121;
    }
    else if ( sub_2E88A90(v115, 0x80000, 1) )
    {
      goto LABEL_121;
    }
    v7 = v115;
    if ( (unsigned int)*(unsigned __int16 *)(v115 + 68) - 1 <= 1
      && (*(_BYTE *)(*(_QWORD *)(v115 + 32) + 64LL) & 0x10) != 0
      || ((v24 = *(_DWORD *)(v115 + 44), (v24 & 4) != 0) || (v24 & 8) == 0
        ? (v25 = (*(_QWORD *)(*(_QWORD *)(v115 + 16) + 24LL) >> 20) & 1LL)
        : (v7 = 0x100000, LOBYTE(v25) = sub_2E88A90(v115, 0x100000, 1)),
          (_BYTE)v25) )
    {
      v132 = (__int64 *)v134;
      v133 = 0x400000000LL;
      sub_2E864A0(v115);
      if ( v28 == 1 )
      {
        v7 = (unsigned __int64)&v132;
        sub_353FFA0(v115, (__int64)&v132);
      }
      v29 = (unsigned int)v133;
      if ( !(_DWORD)v133 )
      {
        if ( !HIDWORD(v133) )
        {
          sub_C8D5F0((__int64)&v132, v134, 1u, 8u, v26, v27);
          v29 = (unsigned int)v133;
        }
        v7 = v108;
        v132[v29] = v108;
        v29 = (unsigned int)(v133 + 1);
        LODWORD(v133) = v133 + 1;
      }
      v30 = v132;
      v107 = &v132[v29];
      if ( v107 != v132 )
      {
        v110 = v132;
        while ( 1 )
        {
          v7 = v127;
          v31 = *v110;
          if ( !v129 )
            goto LABEL_100;
          v32 = (v129 - 1) & (((unsigned int)v31 >> 9) ^ ((unsigned int)v31 >> 4));
          v33 = v127 + 16LL * v32;
          v34 = *(_QWORD *)v33;
          if ( v31 != *(_QWORD *)v33 )
          {
            for ( j = 1; ; j = v27 )
            {
              if ( v34 == -4096 )
                goto LABEL_100;
              v27 = (unsigned int)(j + 1);
              v32 = (v129 - 1) & (j + v32);
              v33 = v127 + 16LL * v32;
              v34 = *(_QWORD *)v33;
              if ( v31 == *(_QWORD *)v33 )
                break;
            }
          }
          if ( v33 != v127 + 16LL * v129 )
          {
            v7 = (unsigned int)v131;
            v35 = &v130[7 * *(unsigned int *)(v33 + 8)];
            if ( v35 != &v130[7 * (unsigned int)v131] )
            {
              v36 = (__int64 **)v35[1];
              v113 = &v36[*((unsigned int *)v35 + 4)];
              if ( v113 != v36 )
                break;
            }
          }
LABEL_100:
          if ( v107 == ++v110 )
          {
            v30 = v132;
            goto LABEL_102;
          }
        }
        v117 = (__int64 **)v35[1];
LABEL_53:
        v37 = v137;
        v38 = *v117;
        v139 = (unsigned __int64)v142;
        v7 = (unsigned __int64)v137;
        v140 = 8;
        v138 = 0;
        LODWORD(v141) = 0;
        BYTE4(v141) = 1;
        v135 = v137;
        v114 = v38;
        v137[0] = v38;
        v136 = 0x800000001LL;
        v39 = 1;
        while ( 1 )
        {
          v40 = v39--;
          v41 = v37[v40 - 1];
          LODWORD(v136) = v39;
          v42 = *(_BYTE **)(v41 + 120);
          v43 = 16LL * *(unsigned int *)(v41 + 128);
          v44 = &v42[v43];
          if ( v42 != &v42[v43] )
            break;
LABEL_72:
          if ( !v39 )
          {
            v53 = 0;
            goto LABEL_74;
          }
        }
        while ( 1 )
        {
          if ( ((*v42 ^ 6) & 6) != 0 )
            goto LABEL_56;
          v45 = *(_QWORD *)v42 & 0xFFFFFFFFFFFFFFF8LL;
          if ( BYTE4(v141) )
          {
            v46 = (_QWORD *)v139;
            v47 = (_QWORD *)(v139 + 8LL * HIDWORD(v140));
            if ( (_QWORD *)v139 != v47 )
            {
              while ( v45 != *v46 )
              {
                if ( v47 == ++v46 )
                  goto LABEL_62;
              }
              goto LABEL_56;
            }
LABEL_62:
            if ( (__int64 *)v45 == v6 )
              goto LABEL_152;
            goto LABEL_63;
          }
          v7 = *(_QWORD *)v42 & 0xFFFFFFFFFFFFFFF8LL;
          if ( sub_C8CA60((__int64)&v138, v45) )
          {
LABEL_56:
            v42 += 16;
            if ( v44 == v42 )
              goto LABEL_71;
          }
          else
          {
            if ( (__int64 *)v45 == v6 )
            {
LABEL_152:
              v37 = v135;
              v53 = 1;
LABEL_74:
              if ( v37 != v137 )
                _libc_free((unsigned __int64)v37);
              if ( !BYTE4(v141) )
                _libc_free(v139);
              if ( !v53 )
              {
                v54 = *v114;
                if ( (unsigned __int8)sub_2FE0930(
                                        *(__int64 **)(a1 + 16),
                                        *v114,
                                        &v121,
                                        (__int64)&v123,
                                        (__int64)&v119,
                                        *(_QWORD *)(a1 + 24)) )
                {
                  if ( (unsigned __int8)sub_2FE0930(
                                          *(__int64 **)(a1 + 16),
                                          v115,
                                          &v122,
                                          (__int64)&v124,
                                          (__int64)&v120,
                                          *(_QWORD *)(a1 + 24))
                    && (unsigned __int8)sub_2EAB6C0(v121, v122)
                    && v119 == v120
                    && v123 < (int)v124 )
                  {
                    goto LABEL_98;
                  }
                }
                if ( !a2 )
                  goto LABEL_98;
                v56 = *(_QWORD *)(v54 + 48);
                v57 = (__int64 *)(v56 & 0xFFFFFFFFFFFFFFF8LL);
                if ( (v56 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
                  goto LABEL_207;
                v58 = v56 & 7;
                if ( !v58 )
                {
                  *(_QWORD *)(v54 + 48) = v57;
                  goto LABEL_84;
                }
                if ( v58 != 3 )
LABEL_207:
                  BUG();
                v57 = (__int64 *)v57[2];
LABEL_84:
                v59 = *(_QWORD *)(v115 + 48);
                v60 = (__int64 *)(v59 & 0xFFFFFFFFFFFFFFF8LL);
                if ( (v59 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
                  goto LABEL_208;
                v61 = v59 & 7;
                if ( !v61 )
                {
                  *(_QWORD *)(v115 + 48) = v60;
                  goto LABEL_87;
                }
                if ( v61 != 3 )
LABEL_208:
                  BUG();
                v60 = (__int64 *)v60[2];
LABEL_87:
                v55 = *v57;
                if ( !*v57 )
                  goto LABEL_98;
                if ( (v55 & 4) != 0 )
                  goto LABEL_98;
                v55 &= 0xFFFFFFFFFFFFFFF8LL;
                if ( !v55 )
                  goto LABEL_98;
                v62 = *v60;
                if ( !*v60 )
                  goto LABEL_98;
                if ( (v62 & 4) != 0 )
                  goto LABEL_98;
                v63 = v62 & 0xFFFFFFFFFFFFFFF8LL;
                if ( !v63 || v63 == v55 && v60[1] >= v57[1] )
                  goto LABEL_98;
                v64 = v60[5];
                v65 = v60[6];
                v66 = v60[7];
                v67 = v60[8];
                v138 = v63;
                v139 = 0xBFFFFFFFFFFFFFFELL;
                v140 = v64;
                v141 = v65;
                v142[0] = v66;
                v142[1] = v67;
                v68 = *v57;
                v69 = v57[5];
                v70 = v57[6];
                v71 = v57[7];
                v72 = v57[8];
                v73 = 0;
                if ( v68 && (v68 & 4) == 0 )
                  v73 = v68 & 0xFFFFFFFFFFFFFFF8LL;
                v137[0] = v69;
                v137[1] = v70;
                v135 = (_QWORD *)v73;
                v137[2] = v71;
                v7 = (unsigned __int64)&v135;
                v136 = 0xBFFFFFFFFFFFFFFELL;
                v137[3] = v72;
                if ( (unsigned __int8)sub_CF4E00(a2, (__int64)&v135, (__int64)&v138) )
                {
LABEL_98:
                  v7 = (unsigned __int64)&v138;
                  v139 = 0x100000000LL;
                  v138 = (unsigned __int64)v114 | 6;
                  sub_2F8F1B0((__int64)v6, (__int64)&v138, 1u, v55, (unsigned __int64)&v138, v27);
                }
              }
              if ( v113 == ++v117 )
                goto LABEL_100;
              goto LABEL_53;
            }
LABEL_63:
            v48 = (unsigned int)v136;
            v49 = HIDWORD(v136);
            v50 = (unsigned int)v136 + 1LL;
            if ( v50 > HIDWORD(v136) )
            {
              v7 = (unsigned __int64)v137;
              sub_C8D5F0((__int64)&v135, v137, v50, 8u, v43, v27);
              v48 = (unsigned int)v136;
            }
            v51 = v135;
            v135[v48] = v45;
            LODWORD(v136) = v136 + 1;
            if ( !BYTE4(v141) )
              goto LABEL_153;
            v52 = (_QWORD *)v139;
            v49 = HIDWORD(v140);
            v51 = (unsigned __int64 *)(v139 + 8LL * HIDWORD(v140));
            if ( (unsigned __int64 *)v139 != v51 )
            {
              while ( v45 != *v52 )
              {
                if ( v51 == ++v52 )
                  goto LABEL_69;
              }
              goto LABEL_56;
            }
LABEL_69:
            if ( HIDWORD(v140) >= (unsigned int)v140 )
            {
LABEL_153:
              v7 = v45;
              sub_C8CC70((__int64)&v138, v45, (__int64)v51, v49, v43, v27);
              goto LABEL_56;
            }
            v42 += 16;
            ++HIDWORD(v140);
            *v51 = v45;
            ++v138;
            if ( v44 == v42 )
            {
LABEL_71:
              v39 = v136;
              v37 = v135;
              goto LABEL_72;
            }
          }
        }
      }
LABEL_102:
      if ( v30 != (__int64 *)v134 )
        _libc_free((unsigned __int64)v30);
    }
LABEL_16:
    v6 += 32;
  }
  v15 = (__int64)v130;
  v16 = &v130[7 * (unsigned int)v131];
  if ( v130 != v16 )
  {
    do
    {
      v16 -= 7;
      v17 = v16[1];
      if ( (__int64 *)v17 != v16 + 3 )
        _libc_free(v17);
    }
    while ( (__int64 *)v15 != v16 );
    v16 = v130;
  }
  if ( v16 != (__int64 *)&v132 )
    _libc_free((unsigned __int64)v16);
  return sub_C7D6A0(v127, 16LL * v129, 8);
}
