// Function: sub_14DECB0
// Address: 0x14decb0
//
void __fastcall sub_14DECB0(_QWORD *a1, __int64 a2)
{
  __int64 v2; // r15
  _QWORD *v3; // rbx
  __int64 v4; // r12
  bool v5; // zf
  __int64 v6; // rax
  __int64 v7; // rdx
  unsigned __int64 *v8; // rdi
  int v9; // r14d
  _BYTE ***v10; // r9
  unsigned int v11; // edx
  unsigned __int64 v12; // rbx
  __int64 v13; // rax
  unsigned __int64 v14; // r13
  unsigned __int64 *v15; // rax
  int v16; // r12d
  unsigned __int64 *v17; // rcx
  unsigned __int64 *v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rcx
  _QWORD *v21; // rax
  _QWORD *v22; // r14
  __int64 v23; // rdx
  _QWORD *v24; // r12
  __int64 v25; // rdi
  __int64 v26; // r12
  __int64 v27; // rax
  _QWORD *v28; // rdx
  int v29; // eax
  _QWORD *v30; // r12
  __int64 v31; // r14
  __int64 v32; // rdi
  __int64 v33; // r14
  unsigned int v34; // r13d
  __int64 *v35; // r8
  __int64 v36; // r11
  char v37; // dl
  __int64 v38; // r10
  __int64 v39; // rcx
  _QWORD *v40; // rdx
  _QWORD *v41; // rax
  _QWORD *v42; // rdx
  __int64 v43; // rax
  unsigned int v44; // eax
  _QWORD *v45; // rdx
  __int64 v46; // rcx
  _QWORD *v47; // rdx
  _QWORD *v48; // rax
  _QWORD *v49; // rdx
  __int64 v50; // rax
  __int64 v51; // rcx
  __int64 v52; // rax
  __int64 v53; // rdi
  unsigned int v54; // esi
  __int64 *v55; // rdx
  __int64 v56; // r10
  __int64 **v57; // rax
  __int64 **v58; // rsi
  __int64 **v59; // rcx
  unsigned int v60; // edi
  __int64 v61; // r8
  unsigned int v62; // esi
  __int64 *v63; // rax
  __int64 v64; // r11
  __int64 v65; // r8
  unsigned __int64 v66; // r13
  _QWORD *v67; // rax
  __int64 v68; // rdi
  _QWORD *v69; // rax
  _QWORD *v70; // r13
  _QWORD *v71; // r14
  char v72; // dl
  __int64 v73; // r12
  _QWORD *v74; // rax
  _QWORD *v75; // rsi
  _QWORD *v76; // rcx
  __int64 v77; // rax
  unsigned __int64 v78; // r11
  __int64 v79; // rax
  __int64 **v80; // rax
  __int64 v81; // rdi
  __int64 v82; // rsi
  int v83; // edx
  int v84; // r9d
  _QWORD *v85; // rcx
  __int64 v86; // rsi
  _QWORD *v87; // rdi
  _QWORD *v88; // rsi
  _QWORD *v89; // rcx
  _BYTE *v90; // rax
  __int64 v91; // rcx
  __int64 v92; // r8
  int v93; // eax
  int v94; // r10d
  __int64 *v95; // [rsp+18h] [rbp-6D8h]
  __int64 *v96; // [rsp+18h] [rbp-6D8h]
  unsigned int v97; // [rsp+24h] [rbp-6CCh]
  unsigned int v98; // [rsp+24h] [rbp-6CCh]
  _QWORD *v99; // [rsp+28h] [rbp-6C8h]
  _QWORD *v100; // [rsp+28h] [rbp-6C8h]
  __int64 *v101; // [rsp+28h] [rbp-6C8h]
  unsigned int v102; // [rsp+30h] [rbp-6C0h]
  __int64 v103; // [rsp+30h] [rbp-6C0h]
  __int64 v104; // [rsp+30h] [rbp-6C0h]
  __int64 v105; // [rsp+30h] [rbp-6C0h]
  unsigned __int64 v106; // [rsp+30h] [rbp-6C0h]
  _BYTE ***v107; // [rsp+30h] [rbp-6C0h]
  __int64 *v108; // [rsp+38h] [rbp-6B8h]
  __int64 v109; // [rsp+38h] [rbp-6B8h]
  __int64 v110; // [rsp+38h] [rbp-6B8h]
  unsigned int v111; // [rsp+38h] [rbp-6B8h]
  __int64 *v112; // [rsp+38h] [rbp-6B8h]
  _QWORD *v113; // [rsp+40h] [rbp-6B0h]
  unsigned int v114; // [rsp+40h] [rbp-6B0h]
  __int64 v115; // [rsp+48h] [rbp-6A8h]
  int v116; // [rsp+48h] [rbp-6A8h]
  __int64 v117; // [rsp+48h] [rbp-6A8h]
  _BYTE *v118; // [rsp+50h] [rbp-6A0h] BYREF
  __int64 v119; // [rsp+58h] [rbp-698h]
  _BYTE v120[256]; // [rsp+60h] [rbp-690h] BYREF
  __int64 v121; // [rsp+160h] [rbp-590h] BYREF
  __int64 **v122; // [rsp+168h] [rbp-588h]
  __int64 **v123; // [rsp+170h] [rbp-580h]
  __int64 v124; // [rsp+178h] [rbp-578h]
  int v125; // [rsp+180h] [rbp-570h]
  _BYTE v126[264]; // [rsp+188h] [rbp-568h] BYREF
  _BYTE **v127; // [rsp+290h] [rbp-460h] BYREF
  _BYTE *v128; // [rsp+298h] [rbp-458h]
  _BYTE *v129; // [rsp+2A0h] [rbp-450h] BYREF
  __int64 v130; // [rsp+2A8h] [rbp-448h]
  int v131; // [rsp+2B0h] [rbp-440h]
  _BYTE v132[488]; // [rsp+2B8h] [rbp-438h] BYREF
  _BYTE *v133; // [rsp+4A0h] [rbp-250h] BYREF
  __int64 v134; // [rsp+4A8h] [rbp-248h]
  _BYTE v135[576]; // [rsp+4B0h] [rbp-240h] BYREF

  v2 = a2;
  v3 = a1;
  v4 = *a1;
  v133 = v135;
  v5 = *(_BYTE *)(v4 + 72) == 0;
  v134 = 0x2000000000LL;
  if ( v5 )
  {
    HIDWORD(v128) = 32;
    v127 = &v129;
    v6 = *(_QWORD *)(v4 + 56);
    if ( v6 )
    {
      v7 = *(_QWORD *)(v6 + 24);
      v8 = (unsigned __int64 *)&v129;
      v129 = *(_BYTE **)(v4 + 56);
      v9 = 1;
      LODWORD(v128) = 1;
      v10 = &v127;
      v130 = v7;
      v11 = 1;
      *(_DWORD *)(v6 + 48) = 0;
      v115 = v4;
      v113 = v3;
      do
      {
        while ( 1 )
        {
          v16 = v9++;
          v17 = &v8[2 * v11 - 2];
          v18 = (unsigned __int64 *)v17[1];
          if ( v18 != *(unsigned __int64 **)(*v17 + 32) )
            break;
          --v11;
          *(_DWORD *)(*v17 + 52) = v16;
          LODWORD(v128) = v11;
          if ( !v11 )
            goto LABEL_9;
        }
        v12 = *v18;
        v17[1] = (unsigned __int64)(v18 + 1);
        v13 = (unsigned int)v128;
        v14 = *(_QWORD *)(v12 + 24);
        if ( (unsigned int)v128 >= HIDWORD(v128) )
        {
          v107 = v10;
          sub_16CD150(v10, &v129, 0, 16);
          v8 = (unsigned __int64 *)v127;
          v13 = (unsigned int)v128;
          v10 = v107;
        }
        v15 = &v8[2 * v13];
        *v15 = v12;
        v15[1] = v14;
        LODWORD(v128) = (_DWORD)v128 + 1;
        v11 = (unsigned int)v128;
        *(_DWORD *)(v12 + 48) = v16;
        v8 = (unsigned __int64 *)v127;
      }
      while ( v11 );
LABEL_9:
      v3 = v113;
      v2 = a2;
      *(_DWORD *)(v115 + 76) = 0;
      *(_BYTE *)(v115 + 72) = 1;
      if ( v8 != (unsigned __int64 *)&v129 )
        _libc_free((unsigned __int64)v8);
      v19 = v113[3];
      v20 = (unsigned int)v134;
      v21 = *(_QWORD **)(v19 + 16);
      if ( v21 == *(_QWORD **)(v19 + 8) )
        goto LABEL_12;
      goto LABEL_15;
    }
  }
  else
  {
    *(_DWORD *)(v4 + 76) = 0;
  }
  v19 = a1[3];
  v20 = 0;
  v21 = *(_QWORD **)(v19 + 16);
  if ( v21 == *(_QWORD **)(v19 + 8) )
  {
LABEL_12:
    v22 = &v21[*(unsigned int *)(v19 + 28)];
    goto LABEL_16;
  }
LABEL_15:
  v22 = &v21[*(unsigned int *)(v19 + 24)];
LABEL_16:
  if ( v21 != v22 )
  {
    while ( 1 )
    {
      v23 = *v21;
      v24 = v21;
      if ( *v21 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v22 == ++v21 )
        goto LABEL_19;
    }
    while ( v22 != v24 )
    {
      v60 = *(_DWORD *)(*v3 + 48LL);
      if ( v60 )
      {
        v61 = *(_QWORD *)(*v3 + 32LL);
        v62 = (v60 - 1) & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
        v63 = (__int64 *)(v61 + 16LL * v62);
        v64 = *v63;
        if ( *v63 == v23 )
        {
LABEL_65:
          if ( v63 != (__int64 *)(v61 + 16LL * v60) )
          {
            v65 = v63[1];
            if ( v65 )
            {
              v66 = ((unsigned __int64)*(unsigned int *)(v65 + 48) << 32) | *(unsigned int *)(v65 + 16);
              if ( HIDWORD(v134) <= (unsigned int)v20 )
              {
                v117 = v63[1];
                sub_16CD150(&v133, v135, 0, 16);
                LODWORD(v20) = v134;
                v65 = v117;
              }
              v67 = &v133[16 * (unsigned int)v20];
              *v67 = v65;
              v68 = (__int64)v133;
              v67[1] = v66;
              LODWORD(v134) = v134 + 1;
              sub_14DEB00(
                v68,
                ((16LL * (unsigned int)v134) >> 4) - 1,
                0,
                *(_QWORD *)(v68 + 16LL * (unsigned int)v134 - 16),
                *(_QWORD *)(v68 + 16LL * (unsigned int)v134 - 8));
              v20 = (unsigned int)v134;
            }
          }
        }
        else
        {
          v93 = 1;
          while ( v64 != -8 )
          {
            v94 = v93 + 1;
            v62 = (v60 - 1) & (v93 + v62);
            v63 = (__int64 *)(v61 + 16LL * v62);
            v64 = *v63;
            if ( *v63 == v23 )
              goto LABEL_65;
            v93 = v94;
          }
        }
      }
      v69 = v24 + 1;
      if ( v24 + 1 == v22 )
        break;
      while ( 1 )
      {
        v23 = *v69;
        v24 = v69;
        if ( *v69 < 0xFFFFFFFFFFFFFFFELL )
          break;
        if ( v22 == ++v69 )
          goto LABEL_19;
      }
    }
  }
LABEL_19:
  v121 = 0;
  v118 = v120;
  v119 = 0x2000000000LL;
  v122 = (__int64 **)v126;
  v123 = (__int64 **)v126;
  v124 = 32;
  v125 = 0;
  v127 = 0;
  v128 = v132;
  v129 = v132;
  v130 = 32;
  v131 = 0;
  if ( (_DWORD)v20 )
  {
    do
    {
      v25 = (__int64)v133;
      v26 = *(_QWORD *)v133;
      v114 = *((_DWORD *)v133 + 2);
      if ( v20 != 1 )
      {
        v90 = &v133[16 * v20];
        v91 = *((_QWORD *)v90 - 2);
        v92 = *((_QWORD *)v90 - 1);
        *((_QWORD *)v90 - 2) = v26;
        *((_DWORD *)v90 - 2) = *(_DWORD *)(v25 + 8);
        *((_DWORD *)v90 - 1) = *(_DWORD *)(v25 + 12);
        sub_14DEBB0(v25, 0, (__int64)&v90[-v25 - 16] >> 4, v91, v92);
      }
      LODWORD(v134) = v134 - 1;
      v27 = 0;
      LODWORD(v119) = 0;
      if ( !HIDWORD(v119) )
      {
        sub_16CD150(&v118, v120, 0, 8);
        v27 = 8LL * (unsigned int)v119;
      }
      *(_QWORD *)&v118[v27] = v26;
      v28 = v128;
      v29 = v119 + 1;
      LODWORD(v119) = v119 + 1;
      if ( v129 != v128 )
        goto LABEL_25;
      v87 = &v128[8 * HIDWORD(v130)];
      if ( v128 != (_BYTE *)v87 )
      {
        v88 = 0;
        do
        {
          if ( v26 == *v28 )
            goto LABEL_27;
          if ( *v28 == -2 )
            v88 = v28;
          ++v28;
        }
        while ( v87 != v28 );
        if ( v88 )
        {
          *v88 = v26;
          v29 = v119;
          --v131;
          v127 = (_BYTE **)((char *)v127 + 1);
          goto LABEL_27;
        }
      }
      if ( HIDWORD(v130) >= (unsigned int)v130 )
      {
LABEL_25:
        sub_16CCBA0(&v127, v26);
        goto LABEL_26;
      }
      ++HIDWORD(v130);
      *v87 = v26;
      v29 = v119;
      v127 = (_BYTE **)((char *)v127 + 1);
LABEL_27:
      while ( v29 )
      {
        v30 = *(_QWORD **)&v118[8 * v29 - 8];
        LODWORD(v119) = v29 - 1;
        v31 = *v30;
        v32 = sub_157EBA0(*v30);
        if ( v32 )
        {
          v116 = sub_15F4D60(v32);
          v33 = sub_157EBA0(v31);
          if ( v116 )
          {
            v34 = 0;
            while ( 1 )
            {
              v51 = sub_15F4DF0(v33, v34);
              v52 = *(unsigned int *)(*v3 + 48LL);
              if ( !(_DWORD)v52 )
                goto LABEL_151;
              v53 = *(_QWORD *)(*v3 + 32LL);
              v54 = (v52 - 1) & (((unsigned int)v51 >> 9) ^ ((unsigned int)v51 >> 4));
              v55 = (__int64 *)(v53 + 16LL * v54);
              v56 = *v55;
              if ( v51 != *v55 )
              {
                v83 = 1;
                while ( v56 != -8 )
                {
                  v84 = v83 + 1;
                  v54 = (v52 - 1) & (v83 + v54);
                  v55 = (__int64 *)(v53 + 16LL * v54);
                  v56 = *v55;
                  if ( v51 == *v55 )
                    goto LABEL_50;
                  v83 = v84;
                }
LABEL_151:
                BUG();
              }
LABEL_50:
              if ( v55 == (__int64 *)(v53 + 16 * v52) )
                goto LABEL_151;
              v35 = (__int64 *)v55[1];
              if ( v30 == (_QWORD *)v35[1] )
                goto LABEL_47;
              v36 = *((unsigned int *)v35 + 4);
              if ( v114 < (unsigned int)v36 )
                goto LABEL_47;
              v57 = v122;
              if ( v123 != v122 )
                goto LABEL_31;
              v58 = &v122[HIDWORD(v124)];
              if ( v122 != v58 )
              {
                v59 = 0;
                while ( v35 != *v57 )
                {
                  if ( *v57 == (__int64 *)-2LL )
                    v59 = v57;
                  if ( v58 == ++v57 )
                  {
                    if ( !v59 )
                      goto LABEL_122;
                    *v59 = v35;
                    --v125;
                    ++v121;
                    goto LABEL_32;
                  }
                }
                goto LABEL_47;
              }
LABEL_122:
              if ( HIDWORD(v124) < (unsigned int)v124 )
              {
                ++HIDWORD(v124);
                *v58 = v35;
                ++v121;
              }
              else
              {
LABEL_31:
                v102 = *((_DWORD *)v35 + 4);
                v108 = (__int64 *)v55[1];
                sub_16CCBA0(&v121, v35);
                v35 = v108;
                v36 = v102;
                if ( !v37 )
                  goto LABEL_47;
              }
LABEL_32:
              v38 = *v35;
              if ( *((_BYTE *)v3 + 8) )
                break;
LABEL_38:
              v44 = *(_DWORD *)(v2 + 8);
              if ( v44 >= *(_DWORD *)(v2 + 12) )
              {
                v101 = v35;
                v105 = v38;
                v111 = v36;
                sub_16CD150(v2, v2 + 16, 0, 8);
                v44 = *(_DWORD *)(v2 + 8);
                v35 = v101;
                v38 = v105;
                v36 = v111;
              }
              v45 = (_QWORD *)(*(_QWORD *)v2 + 8LL * v44);
              if ( v45 )
              {
                *v45 = v38;
                v44 = *(_DWORD *)(v2 + 8);
              }
              *(_DWORD *)(v2 + 8) = v44 + 1;
              v46 = v3[3];
              v47 = *(_QWORD **)(v46 + 16);
              v48 = *(_QWORD **)(v46 + 8);
              if ( v47 == v48 )
              {
                v49 = &v48[*(unsigned int *)(v46 + 28)];
                if ( v48 == v49 )
                {
                  v89 = *(_QWORD **)(v46 + 8);
                }
                else
                {
                  do
                  {
                    if ( v38 == *v48 )
                      break;
                    ++v48;
                  }
                  while ( v49 != v48 );
                  v89 = v49;
                }
LABEL_96:
                while ( v89 != v48 )
                {
                  if ( *v48 < 0xFFFFFFFFFFFFFFFELL )
                    goto LABEL_46;
                  ++v48;
                }
                if ( v49 != v48 )
                  goto LABEL_47;
              }
              else
              {
                v96 = v35;
                v98 = v36;
                v104 = v38;
                v100 = &v47[*(unsigned int *)(v46 + 24)];
                v110 = v3[3];
                v48 = (_QWORD *)sub_16CC9F0(v110, v38);
                v49 = v100;
                v36 = v98;
                v35 = v96;
                if ( v104 == *v48 )
                {
                  v82 = *(_QWORD *)(v110 + 16);
                  if ( v82 == *(_QWORD *)(v110 + 8) )
                    v89 = (_QWORD *)(v82 + 8LL * *(unsigned int *)(v110 + 28));
                  else
                    v89 = (_QWORD *)(v82 + 8LL * *(unsigned int *)(v110 + 24));
                  goto LABEL_96;
                }
                v50 = *(_QWORD *)(v110 + 16);
                if ( v50 == *(_QWORD *)(v110 + 8) )
                {
                  v48 = (_QWORD *)(v50 + 8LL * *(unsigned int *)(v110 + 28));
                  v89 = v48;
                  goto LABEL_96;
                }
                v48 = (_QWORD *)(v50 + 8LL * *(unsigned int *)(v110 + 24));
LABEL_46:
                if ( v49 != v48 )
                  goto LABEL_47;
              }
              v78 = ((unsigned __int64)*((unsigned int *)v35 + 12) << 32) | v36;
              v79 = (unsigned int)v134;
              if ( (unsigned int)v134 >= HIDWORD(v134) )
              {
                v106 = v78;
                v112 = v35;
                sub_16CD150(&v133, v135, 0, 16);
                v79 = (unsigned int)v134;
                v78 = v106;
                v35 = v112;
              }
              v80 = (__int64 **)&v133[16 * v79];
              *v80 = v35;
              v81 = (__int64)v133;
              v80[1] = (__int64 *)v78;
              LODWORD(v134) = v134 + 1;
              sub_14DEB00(
                v81,
                ((16LL * (unsigned int)v134) >> 4) - 1,
                0,
                *(_QWORD *)(v81 + 16LL * (unsigned int)v134 - 16),
                *(_QWORD *)(v81 + 16LL * (unsigned int)v134 - 8));
LABEL_47:
              if ( ++v34 == v116 )
                goto LABEL_74;
            }
            v39 = v3[2];
            v40 = *(_QWORD **)(v39 + 16);
            v41 = *(_QWORD **)(v39 + 8);
            if ( v40 == v41 )
            {
              v85 = &v41[*(unsigned int *)(v39 + 28)];
              if ( v41 == v85 )
              {
                v42 = v41;
              }
              else
              {
                do
                {
                  if ( v38 == *v41 )
                    break;
                  ++v41;
                }
                while ( v85 != v41 );
                v42 = v85;
              }
            }
            else
            {
              v95 = v35;
              v97 = v36;
              v103 = *v35;
              v99 = &v40[*(unsigned int *)(v39 + 24)];
              v109 = v3[2];
              v41 = (_QWORD *)sub_16CC9F0(v109, *v35);
              v38 = v103;
              v42 = v99;
              v36 = v97;
              v35 = v95;
              if ( v103 == *v41 )
              {
                v86 = *(_QWORD *)(v109 + 16);
                if ( v86 == *(_QWORD *)(v109 + 8) )
                  v85 = (_QWORD *)(v86 + 8LL * *(unsigned int *)(v109 + 28));
                else
                  v85 = (_QWORD *)(v86 + 8LL * *(unsigned int *)(v109 + 24));
              }
              else
              {
                v43 = *(_QWORD *)(v109 + 16);
                if ( v43 != *(_QWORD *)(v109 + 8) )
                {
                  v41 = (_QWORD *)(v43 + 8LL * *(unsigned int *)(v109 + 24));
                  goto LABEL_37;
                }
                v85 = (_QWORD *)(v43 + 8LL * *(unsigned int *)(v109 + 28));
                v41 = v85;
              }
            }
            for ( ; v85 != v41; ++v41 )
            {
              if ( *v41 < 0xFFFFFFFFFFFFFFFELL )
                break;
            }
LABEL_37:
            if ( v42 == v41 )
              goto LABEL_47;
            goto LABEL_38;
          }
        }
LABEL_74:
        v70 = (_QWORD *)v30[4];
        v71 = (_QWORD *)v30[3];
        if ( v70 != v71 )
        {
          while ( 1 )
          {
            v73 = *v71;
            v74 = v128;
            if ( v129 == v128 )
            {
              v75 = &v128[8 * HIDWORD(v130)];
              if ( v128 != (_BYTE *)v75 )
              {
                v76 = 0;
                while ( v73 != *v74 )
                {
                  if ( *v74 == -2 )
                    v76 = v74;
                  if ( v75 == ++v74 )
                  {
                    if ( !v76 )
                      goto LABEL_103;
                    *v76 = v73;
                    --v131;
                    v127 = (_BYTE **)((char *)v127 + 1);
                    goto LABEL_87;
                  }
                }
                goto LABEL_77;
              }
LABEL_103:
              if ( HIDWORD(v130) < (unsigned int)v130 )
                break;
            }
            sub_16CCBA0(&v127, *v71);
            if ( v72 )
            {
LABEL_87:
              v77 = (unsigned int)v119;
              if ( (unsigned int)v119 < HIDWORD(v119) )
              {
LABEL_88:
                *(_QWORD *)&v118[8 * v77] = v73;
                LODWORD(v119) = v119 + 1;
                goto LABEL_77;
              }
LABEL_105:
              sub_16CD150(&v118, v120, 0, 8);
              v77 = (unsigned int)v119;
              goto LABEL_88;
            }
LABEL_77:
            if ( v70 == ++v71 )
              goto LABEL_26;
          }
          ++HIDWORD(v130);
          *v75 = v73;
          v77 = (unsigned int)v119;
          v127 = (_BYTE **)((char *)v127 + 1);
          if ( (unsigned int)v119 < HIDWORD(v119) )
            goto LABEL_88;
          goto LABEL_105;
        }
LABEL_26:
        v29 = v119;
      }
      v20 = (unsigned int)v134;
    }
    while ( (_DWORD)v134 );
    if ( v129 != v128 )
      _libc_free((unsigned __int64)v129);
    if ( v122 != v123 )
      _libc_free((unsigned __int64)v123);
  }
  if ( v118 != v120 )
    _libc_free((unsigned __int64)v118);
  if ( v133 != v135 )
    _libc_free((unsigned __int64)v133);
}
