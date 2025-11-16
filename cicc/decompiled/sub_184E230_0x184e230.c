// Function: sub_184E230
// Address: 0x184e230
//
__int64 __fastcall sub_184E230(__int64 a1)
{
  __int64 v1; // rdx
  __int64 *v2; // rax
  __int64 v3; // rsi
  __int64 v4; // r13
  char v5; // al
  __int64 v6; // r14
  __int64 v7; // rbx
  unsigned __int8 v8; // al
  __int64 v9; // r12
  unsigned __int64 v10; // rdx
  unsigned __int64 v11; // r13
  __int64 *v12; // rax
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // r14
  __int64 v16; // r15
  char v17; // bl
  __int64 v18; // rcx
  __int64 v19; // r12
  unsigned __int8 v20; // r12
  unsigned __int64 v21; // rbx
  unsigned __int64 v22; // rdx
  unsigned __int64 v23; // r14
  unsigned int v24; // eax
  char v25; // r11
  __int64 **v26; // r12
  __int64 **v27; // rbx
  __int64 *v28; // rax
  __int64 v29; // rdi
  __int64 *v30; // rax
  __int64 **v31; // rbx
  __int64 *v32; // r8
  __int64 **v33; // rbx
  __int64 **v34; // r12
  __int64 *v35; // r9
  __int64 v36; // rsi
  __int64 *v37; // rdi
  __int64 *v38; // rax
  __int64 *v39; // rcx
  __int64 **v40; // rbx
  __int64 v41; // rax
  __int64 **v42; // r14
  __int64 **v43; // r12
  __int64 v44; // r15
  unsigned __int64 v45; // rdi
  __int64 *v46; // rax
  __int64 *v47; // r13
  __int64 **v48; // rax
  __int64 v49; // rdx
  unsigned __int64 v50; // rbx
  __int64 v51; // r12
  __int64 **v52; // rcx
  __int64 v53; // rax
  unsigned __int64 v54; // r12
  char v55; // r14
  __int64 v56; // r13
  int v57; // eax
  char v58; // bl
  __int64 **v59; // rdx
  __int64 v61; // rbx
  __int64 v62; // rdx
  __int64 i; // r13
  int *v64; // rax
  int *v65; // r9
  __int64 v66; // rcx
  __int64 v67; // rdx
  unsigned __int64 v68; // rax
  _QWORD *v69; // r8
  __int64 v70; // rax
  unsigned __int64 *v71; // rbx
  int *v72; // r14
  char v73; // r13
  unsigned __int64 v74; // r12
  int *v75; // rax
  __int64 v76; // rcx
  __int64 v77; // rdx
  unsigned __int64 v78; // rdx
  _QWORD *v79; // rax
  __int64 v80; // rax
  __int64 v81; // rdx
  __int64 v82; // rdx
  char v83; // r11
  int *v84; // rax
  __int64 *v85; // rdx
  __int64 v86; // rax
  __int64 v87; // r13
  unsigned __int64 j; // r12
  __int64 v89; // r14
  __int64 v90; // [rsp+0h] [rbp-200h]
  __int64 v91; // [rsp+0h] [rbp-200h]
  __int64 v92; // [rsp+28h] [rbp-1D8h]
  __int64 v93; // [rsp+30h] [rbp-1D0h]
  unsigned __int64 v94; // [rsp+30h] [rbp-1D0h]
  __int64 v95; // [rsp+38h] [rbp-1C8h]
  __int64 v96; // [rsp+38h] [rbp-1C8h]
  __int64 v97; // [rsp+38h] [rbp-1C8h]
  _QWORD *v98; // [rsp+38h] [rbp-1C8h]
  unsigned __int8 v99; // [rsp+47h] [rbp-1B9h]
  char v100; // [rsp+50h] [rbp-1B0h]
  unsigned __int64 *v101; // [rsp+50h] [rbp-1B0h]
  int *v102; // [rsp+50h] [rbp-1B0h]
  __int64 *v103; // [rsp+58h] [rbp-1A8h]
  __int64 **v104; // [rsp+58h] [rbp-1A8h]
  __int64 v105; // [rsp+60h] [rbp-1A0h]
  unsigned __int64 v106; // [rsp+60h] [rbp-1A0h]
  char v107; // [rsp+60h] [rbp-1A0h]
  char v108; // [rsp+60h] [rbp-1A0h]
  char v109; // [rsp+60h] [rbp-1A0h]
  __int64 *v110; // [rsp+68h] [rbp-198h]
  unsigned __int64 v111; // [rsp+78h] [rbp-188h] BYREF
  __int64 v112; // [rsp+80h] [rbp-180h] BYREF
  int v113; // [rsp+88h] [rbp-178h] BYREF
  int *v114; // [rsp+90h] [rbp-170h]
  int *v115; // [rsp+98h] [rbp-168h]
  int *v116; // [rsp+A0h] [rbp-160h]
  __int64 v117; // [rsp+A8h] [rbp-158h]
  __int64 v118; // [rsp+B0h] [rbp-150h] BYREF
  _BYTE *v119; // [rsp+B8h] [rbp-148h] BYREF
  __int64 v120; // [rsp+C0h] [rbp-140h]
  _BYTE v121[40]; // [rsp+C8h] [rbp-138h] BYREF
  unsigned __int64 *v122; // [rsp+F0h] [rbp-110h] BYREF
  __int64 *v123; // [rsp+F8h] [rbp-108h]
  __int64 *v124; // [rsp+100h] [rbp-100h]
  __int64 v125; // [rsp+108h] [rbp-F8h]
  int v126; // [rsp+110h] [rbp-F0h]
  _BYTE v127[72]; // [rsp+118h] [rbp-E8h] BYREF
  __int64 v128; // [rsp+160h] [rbp-A0h] BYREF
  unsigned __int64 *v129; // [rsp+168h] [rbp-98h]
  unsigned __int64 *v130; // [rsp+170h] [rbp-90h]
  __int64 v131; // [rsp+178h] [rbp-88h]
  __int64 v132; // [rsp+180h] [rbp-80h] BYREF
  _QWORD v133[2]; // [rsp+188h] [rbp-78h] BYREF
  __int64 v134; // [rsp+198h] [rbp-68h]
  __int64 **v135; // [rsp+1A0h] [rbp-60h]
  __int64 **v136; // [rsp+1A8h] [rbp-58h]
  __int64 v137; // [rsp+1B0h] [rbp-50h]
  __int64 v138; // [rsp+1B8h] [rbp-48h]
  __int64 v139; // [rsp+1C0h] [rbp-40h]
  __int64 v140; // [rsp+1C8h] [rbp-38h]

  v1 = *(unsigned int *)(a1 + 88);
  v115 = &v113;
  v116 = &v113;
  v119 = v121;
  v120 = 0x400000000LL;
  v2 = *(__int64 **)(a1 + 80);
  v113 = 0;
  v3 = (__int64)&v2[v1];
  v114 = 0;
  v117 = 0;
  v118 = 0;
  v103 = (__int64 *)v3;
  if ( v2 != (__int64 *)v3 )
  {
    v110 = v2;
    v99 = 0;
    while ( 1 )
    {
      v4 = *v110;
      if ( sub_15E4F60(*v110) )
        goto LABEL_3;
      sub_15E4B50(v4);
      v100 = v5;
      if ( v5 )
        goto LABEL_3;
      if ( byte_4FAA740 )
      {
        v6 = *(_QWORD *)(v4 + 80);
        if ( !v6 )
          BUG();
        v7 = *(_QWORD *)(v6 + 24);
        v105 = v6 + 16;
        if ( v7 != v6 + 16 )
        {
          v92 = v4;
          while ( 1 )
          {
            if ( !v7 )
              BUG();
            v8 = *(_BYTE *)(v7 - 8);
            v9 = v7 - 24;
            if ( v8 <= 0x17u )
              break;
            v10 = v9 | 4;
            if ( v8 != 78 )
            {
              if ( v8 != 29 )
                break;
              v10 = v9 & 0xFFFFFFFFFFFFFFFBLL;
            }
            v11 = v10 & 0xFFFFFFFFFFFFFFF8LL;
            if ( (v10 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
              break;
            v12 = (__int64 *)(v11 - 72);
            if ( (v10 & 4) != 0 )
              v12 = (__int64 *)(v11 - 24);
            v13 = *v12;
            if ( *(_BYTE *)(v13 + 16) )
              break;
            if ( (*(_BYTE *)(v13 + 18) & 1) != 0 )
            {
              v96 = v13;
              sub_15E08E0(v13, v3);
              v14 = *(_QWORD *)(v96 + 88);
              v15 = v14 + 40LL * *(_QWORD *)(v96 + 96);
              if ( (*(_BYTE *)(v96 + 18) & 1) != 0 )
              {
                sub_15E08E0(v96, v3);
                v14 = *(_QWORD *)(v96 + 88);
              }
            }
            else
            {
              v14 = *(_QWORD *)(v13 + 88);
              v15 = v14 + 40LL * *(_QWORD *)(v13 + 96);
            }
            if ( v14 == v15 )
              break;
            v95 = v7 - 24;
            v16 = v14;
            v93 = v7;
            do
            {
              while ( 1 )
              {
                v17 = sub_15E46C0(v16);
                if ( v17 )
                {
                  v18 = *(_DWORD *)(v11 + 20) & 0xFFFFFFF;
                  v19 = *(_QWORD *)(v11 + 24 * (*(unsigned int *)(v16 + 32) - v18));
                  if ( *(_BYTE *)(v19 + 16) == 17
                    && !(unsigned __int8)sub_15E46C0(*(_QWORD *)(v11 + 24 * (*(unsigned int *)(v16 + 32) - v18))) )
                  {
                    break;
                  }
                }
                v16 += 40;
                if ( v15 == v16 )
                  goto LABEL_29;
              }
              v3 = 32;
              v16 += 40;
              sub_15E0E40(v19, 32);
              v100 = v17;
            }
            while ( v15 != v16 );
LABEL_29:
            v7 = v93;
            if ( !(unsigned __int8)sub_14AE440(v95) )
            {
LABEL_30:
              v99 |= v100;
              v4 = v92;
              goto LABEL_31;
            }
LABEL_12:
            v7 = *(_QWORD *)(v7 + 8);
            if ( v105 == v7 )
              goto LABEL_30;
          }
          if ( !(unsigned __int8)sub_14AE440(v7 - 24) )
            goto LABEL_30;
          goto LABEL_12;
        }
      }
LABEL_31:
      if ( (unsigned __int8)sub_1560180(v4 + 112, 36) || (v3 = 37, (unsigned __int8)sub_1560180(v4 + 112, 37)) )
      {
        v3 = 30;
        v20 = sub_1560180(v4 + 112, 30);
        if ( v20 && !*(_BYTE *)(**(_QWORD **)(*(_QWORD *)(v4 + 24) + 16LL) + 8LL) )
        {
          if ( (*(_BYTE *)(v4 + 18) & 1) != 0 )
          {
            sub_15E08E0(v4, 30);
            v61 = *(_QWORD *)(v4 + 88);
            if ( (*(_BYTE *)(v4 + 18) & 1) != 0 )
              sub_15E08E0(v4, 30);
            v62 = *(_QWORD *)(v4 + 88);
          }
          else
          {
            v61 = *(_QWORD *)(v4 + 88);
            v62 = v61;
          }
          for ( i = v62 + 40LL * *(_QWORD *)(v4 + 96); v61 != i; v61 += 40 )
          {
            if ( *(_BYTE *)(*(_QWORD *)v61 + 8LL) == 15 && !(unsigned __int8)sub_15E04D0(v61) )
            {
              v3 = 22;
              sub_15E0E40(v61, 22);
              v99 = v20;
            }
          }
          goto LABEL_3;
        }
        if ( (*(_BYTE *)(v4 + 18) & 1) == 0 )
        {
LABEL_35:
          v21 = *(_QWORD *)(v4 + 88);
          v22 = v21;
          goto LABEL_36;
        }
      }
      else if ( (*(_BYTE *)(v4 + 18) & 1) == 0 )
      {
        goto LABEL_35;
      }
      sub_15E08E0(v4, v3);
      v21 = *(_QWORD *)(v4 + 88);
      if ( (*(_BYTE *)(v4 + 18) & 1) != 0 )
        sub_15E08E0(v4, v3);
      v22 = *(_QWORD *)(v4 + 88);
LABEL_36:
      v23 = v22 + 40LL * *(_QWORD *)(v4 + 96);
      if ( v21 != v23 )
      {
        while ( *(_BYTE *)(*(_QWORD *)v21 + 8LL) != 15 )
        {
LABEL_38:
          v21 += 40LL;
          if ( v21 == v23 )
            goto LABEL_3;
        }
        if ( (unsigned __int8)sub_15E04D0(v21) )
        {
LABEL_41:
          if ( !(unsigned __int8)sub_15E03C0(v21) )
          {
            v133[0] = v21;
            v3 = (__int64)&v128;
            v129 = v133;
            v130 = v133;
            v131 = 0x100000008LL;
            LODWORD(v132) = 0;
            v128 = 1;
            v24 = sub_184D2E0(v21, (__int64)&v128);
            if ( v24 )
            {
              v3 = v24;
              sub_15E0E40(v21, v24);
              v99 = 1;
            }
            if ( v130 != v129 )
              _libc_free((unsigned __int64)v130);
          }
          goto LABEL_38;
        }
        v131 = 0x400000000LL;
        v3 = (__int64)&v128;
        v128 = (__int64)off_49F10C0;
        LOBYTE(v129) = 0;
        v130 = (unsigned __int64 *)&v132;
        v135 = (__int64 **)a1;
        sub_139C7C0(v21, (__int64)&v128);
        v25 = (char)v129;
        if ( (_BYTE)v129 )
        {
LABEL_50:
          v128 = (__int64)off_49F10C0;
          if ( v130 != (unsigned __int64 *)&v132 )
            _libc_free((unsigned __int64)v130);
        }
        else
        {
          if ( !(_DWORD)v131 )
          {
            v3 = 22;
            sub_15E0E40(v21, 22);
            v99 = 1;
            goto LABEL_50;
          }
          v64 = v114;
          v111 = v21;
          v65 = &v113;
          if ( !v114 )
            goto LABEL_171;
          do
          {
            while ( 1 )
            {
              v66 = *((_QWORD *)v64 + 2);
              v67 = *((_QWORD *)v64 + 3);
              if ( *((_QWORD *)v64 + 4) >= v21 )
                break;
              v64 = (int *)*((_QWORD *)v64 + 3);
              if ( !v67 )
                goto LABEL_146;
            }
            v65 = v64;
            v64 = (int *)*((_QWORD *)v64 + 2);
          }
          while ( v66 );
LABEL_146:
          if ( v65 == &v113 || (v68 = v21, *((_QWORD *)v65 + 4) > v21) )
          {
LABEL_171:
            v3 = (__int64)v65;
            v108 = (char)v129;
            v122 = &v111;
            v84 = (int *)sub_18485C0(&v112, v65, &v122);
            v25 = v108;
            v65 = v84;
            v68 = v111;
          }
          *((_QWORD *)v65 + 5) = v68;
          v69 = v65 + 10;
          v70 = (unsigned int)v120;
          if ( (unsigned int)v120 >= HIDWORD(v120) )
          {
            v3 = (__int64)v121;
            v98 = v65 + 10;
            v102 = v65;
            v109 = v25;
            sub_16CD150((__int64)&v119, v121, 0, 8, (int)v69, (int)v65);
            v70 = (unsigned int)v120;
            v69 = v98;
            v65 = v102;
            v25 = v109;
          }
          *(_QWORD *)&v119[8 * v70] = v69;
          LODWORD(v120) = v120 + 1;
          v101 = &v130[(unsigned int)v131];
          if ( v130 != v101 )
          {
            v97 = (__int64)(v65 + 12);
            v106 = v21;
            v71 = v130;
            v94 = v23;
            v72 = v65;
            v73 = v25;
            do
            {
              v74 = *v71;
              v75 = v114;
              v3 = (__int64)&v113;
              v111 = *v71;
              if ( !v114 )
                goto LABEL_159;
              do
              {
                while ( 1 )
                {
                  v76 = *((_QWORD *)v75 + 2);
                  v77 = *((_QWORD *)v75 + 3);
                  if ( *((_QWORD *)v75 + 4) >= v74 )
                    break;
                  v75 = (int *)*((_QWORD *)v75 + 3);
                  if ( !v77 )
                    goto LABEL_157;
                }
                v3 = (__int64)v75;
                v75 = (int *)*((_QWORD *)v75 + 2);
              }
              while ( v76 );
LABEL_157:
              if ( (int *)v3 == &v113 || (v78 = v74, *(_QWORD *)(v3 + 32) > v74) )
              {
LABEL_159:
                v122 = &v111;
                v79 = sub_18485C0(&v112, (_QWORD *)v3, &v122);
                v78 = v111;
                v3 = (__int64)v79;
              }
              *(_QWORD *)(v3 + 40) = v78;
              v80 = v3 + 40;
              v81 = (unsigned int)v120;
              if ( (unsigned int)v120 >= HIDWORD(v120) )
              {
                v3 = (__int64)v121;
                v90 = v80;
                sub_16CD150((__int64)&v119, v121, 0, 8, (int)v69, (int)v65);
                v81 = (unsigned int)v120;
                v80 = v90;
              }
              *(_QWORD *)&v119[8 * v81] = v80;
              LODWORD(v120) = v120 + 1;
              v82 = (unsigned int)v72[14];
              if ( (unsigned int)v82 >= v72[15] )
              {
                v3 = (__int64)(v72 + 16);
                v91 = v80;
                sub_16CD150(v97, v72 + 16, 0, 8, (int)v69, (int)v65);
                v82 = (unsigned int)v72[14];
                v80 = v91;
              }
              *(_QWORD *)(*((_QWORD *)v72 + 6) + 8 * v82) = v80;
              ++v72[14];
              if ( v74 != v106 )
                v73 = 1;
              ++v71;
            }
            while ( v101 != v71 );
            v83 = v73;
            v128 = (__int64)off_49F10C0;
            v21 = v106;
            v23 = v94;
            if ( v130 != (unsigned __int64 *)&v132 )
            {
              _libc_free((unsigned __int64)v130);
              v83 = v73;
            }
            v107 = v83;
            nullsub_518();
            if ( v107 )
              goto LABEL_38;
            goto LABEL_41;
          }
          v128 = (__int64)off_49F10C0;
          if ( v130 != (unsigned __int64 *)&v132 )
          {
            _libc_free((unsigned __int64)v130);
            nullsub_518();
            goto LABEL_41;
          }
        }
        nullsub_518();
        goto LABEL_41;
      }
LABEL_3:
      if ( v103 == ++v110 )
        goto LABEL_61;
    }
  }
  v99 = 0;
LABEL_61:
  v128 = 0;
  v129 = 0;
  v130 = 0;
  v131 = 0;
  v132 = 0;
  v133[0] = 0;
  v133[1] = 0;
  v134 = 0;
  v135 = 0;
  v136 = 0;
  v137 = 0;
  v138 = 0;
  v139 = 0;
  v140 = 0;
  sub_184A1D0((int *)&v128, (__int64)&v118);
  sub_184A5F0((__int64)&v128);
  v26 = v136;
  v27 = v135;
  if ( v135 != v136 )
  {
LABEL_66:
    if ( (char *)v26 - (char *)v27 == 8 )
    {
      v28 = *v27;
      v29 = **v27;
      if ( v29 && *((_DWORD *)v28 + 4) == 1 && *(__int64 **)v28[1] == v28 )
      {
        sub_15E0E40(v29, 22);
        v99 = 1;
      }
      goto LABEL_65;
    }
    v30 = *v27;
    v31 = v27 + 1;
    if ( !*((_DWORD *)v30 + 4) )
      goto LABEL_70;
    while ( v26 != v31 )
    {
      v30 = *v31++;
      if ( !*((_DWORD *)v30 + 4) )
      {
LABEL_70:
        if ( !(unsigned __int8)sub_15E04D0(*v30) )
          goto LABEL_65;
      }
    }
    v32 = (__int64 *)v127;
    v33 = v136;
    v122 = 0;
    v34 = v135;
    v125 = 8;
    v123 = (__int64 *)v127;
    v35 = (__int64 *)v127;
    v124 = (__int64 *)v127;
    v126 = 0;
    if ( v136 == v135 )
      goto LABEL_105;
    while ( 1 )
    {
LABEL_76:
      v36 = **v34;
      if ( v32 != v35 )
        goto LABEL_74;
      v37 = &v32[HIDWORD(v125)];
      if ( v37 != v32 )
      {
        v38 = v32;
        v39 = 0;
        while ( v36 != *v38 )
        {
          if ( *v38 == -2 )
            v39 = v38;
          if ( v37 == ++v38 )
          {
            if ( !v39 )
              goto LABEL_131;
            ++v34;
            *v39 = v36;
            v32 = v124;
            --v126;
            v35 = v123;
            v122 = (unsigned __int64 *)((char *)v122 + 1);
            if ( v33 != v34 )
              goto LABEL_76;
            goto LABEL_85;
          }
        }
        goto LABEL_75;
      }
LABEL_131:
      if ( HIDWORD(v125) < (unsigned int)v125 )
      {
        ++HIDWORD(v125);
        *v37 = v36;
        v35 = v123;
        v122 = (unsigned __int64 *)((char *)v122 + 1);
        v32 = v124;
      }
      else
      {
LABEL_74:
        sub_16CCBA0((__int64)&v122, v36);
        v32 = v124;
        v35 = v123;
      }
LABEL_75:
      if ( v33 == ++v34 )
      {
LABEL_85:
        v40 = v135;
        v104 = v136;
        if ( v136 == v135 )
          goto LABEL_105;
        while ( 2 )
        {
          v41 = (__int64)*v40++;
          v42 = *(__int64 ***)(v41 + 8);
          v43 = &v42[*(unsigned int *)(v41 + 16)];
          if ( v43 != v42 )
          {
            while ( 2 )
            {
              v44 = **v42;
              if ( (unsigned __int8)sub_15E04D0(v44) )
                goto LABEL_93;
              v45 = (unsigned __int64)v124;
              v46 = v123;
              if ( v124 == v123 )
              {
                v47 = &v124[HIDWORD(v125)];
                if ( v124 == v47 )
                {
                  v85 = v124;
                }
                else
                {
                  do
                  {
                    if ( v44 == *v46 )
                      break;
                    ++v46;
                  }
                  while ( v47 != v46 );
                  v85 = &v124[HIDWORD(v125)];
                }
              }
              else
              {
                v47 = &v124[(unsigned int)v125];
                v46 = sub_16CC9F0((__int64)&v122, v44);
                if ( v44 == *v46 )
                {
                  v45 = (unsigned __int64)v124;
                  if ( v124 == v123 )
                    v85 = &v124[HIDWORD(v125)];
                  else
                    v85 = &v124[(unsigned int)v125];
                }
                else
                {
                  v45 = (unsigned __int64)v124;
                  if ( v124 != v123 )
                  {
                    v46 = &v124[(unsigned int)v125];
LABEL_92:
                    if ( v47 == v46 )
                      goto LABEL_106;
LABEL_93:
                    if ( v43 == ++v42 )
                      goto LABEL_94;
                    continue;
                  }
                  v46 = &v124[HIDWORD(v125)];
                  v85 = v46;
                }
              }
              break;
            }
            for ( ; v85 != v46; ++v46 )
            {
              if ( (unsigned __int64)*v46 < 0xFFFFFFFFFFFFFFFELL )
                break;
            }
            goto LABEL_92;
          }
LABEL_94:
          if ( v104 != v40 )
            continue;
          break;
        }
        v48 = v135;
        v49 = v136 - v135;
        if ( !(_DWORD)v49 )
          goto LABEL_105;
        v50 = 0;
        v51 = 8LL * (unsigned int)(v49 - 1);
        while ( 1 )
        {
          sub_15E0E40(*v48[v50 / 8], 22);
          if ( v50 == v51 )
            break;
          v48 = v135;
          v50 += 8LL;
        }
        v52 = v135;
        v53 = v136 - v135;
        if ( !(_DWORD)v53 )
          goto LABEL_104;
        v54 = 0;
        v55 = 36;
        v56 = 8LL * (unsigned int)v53;
        while ( 2 )
        {
          v57 = sub_184D2E0(*v52[v54 / 8], (__int64)&v122);
          v58 = v57;
          if ( v57 == 36 )
            goto LABEL_173;
          if ( v57 == 37 )
          {
            v55 = 37;
LABEL_173:
            v59 = v135;
            v54 += 8LL;
            v52 = v135;
            if ( v54 == v56 )
            {
              v58 = v55;
              goto LABEL_175;
            }
            continue;
          }
          break;
        }
        v59 = v135;
        if ( !v57 )
          goto LABEL_104;
LABEL_175:
        v86 = v136 - v59;
        if ( (_DWORD)v86 )
        {
          v87 = 8LL * (unsigned int)(v86 - 1);
          for ( j = 0; ; j += 8LL )
          {
            v89 = *v59[j / 8];
            sub_15E0F90(v89, 37);
            sub_15E0F90(v89, 36);
            sub_15E0E40(v89, v58);
            if ( j == v87 )
              break;
            v59 = v135;
          }
        }
LABEL_104:
        v99 = 1;
LABEL_105:
        v45 = (unsigned __int64)v124;
LABEL_106:
        if ( (__int64 *)v45 != v123 )
          _libc_free(v45);
LABEL_65:
        sub_184A5F0((__int64)&v128);
        v26 = v136;
        v27 = v135;
        if ( v136 == v135 )
          break;
        goto LABEL_66;
      }
    }
  }
  if ( v138 )
    j_j___libc_free_0(v138, v140 - v138);
  if ( v135 )
    j_j___libc_free_0(v135, v137 - (_QWORD)v135);
  if ( v133[0] )
    j_j___libc_free_0(v133[0], v134 - v133[0]);
  j___libc_free_0(v130);
  if ( v119 != v121 )
    _libc_free((unsigned __int64)v119);
  sub_1849060(v114);
  return v99;
}
