// Function: sub_14DF920
// Address: 0x14df920
//
void __fastcall sub_14DF920(_QWORD *a1, __int64 a2)
{
  __int64 v2; // r13
  _QWORD *v3; // rbx
  __int64 v4; // r12
  bool v5; // zf
  __int64 v6; // rax
  __int64 v7; // rdx
  unsigned __int64 *v8; // rdi
  int v9; // r15d
  _BYTE ***v10; // r10
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
  _QWORD *v22; // r15
  __int64 v23; // rdx
  _QWORD *v24; // r12
  __int64 v25; // rdi
  __int64 v26; // r12
  __int64 v27; // rax
  _QWORD *v28; // rdx
  int v29; // eax
  _QWORD *v30; // r12
  __int64 v31; // r14
  __int64 v32; // rax
  _QWORD *v33; // r14
  _QWORD *v34; // r15
  char v35; // dl
  __int64 v36; // r12
  _QWORD *v37; // rax
  _QWORD *v38; // rsi
  _QWORD *v39; // rcx
  __int64 v40; // rax
  unsigned int v41; // edi
  __int64 v42; // r8
  unsigned int v43; // esi
  __int64 *v44; // rax
  __int64 v45; // r11
  __int64 v46; // r8
  unsigned __int64 v47; // r14
  _QWORD *v48; // rax
  __int64 v49; // rdi
  _QWORD *v50; // rax
  __int64 *v51; // r8
  char v52; // dl
  __int64 v53; // r10
  __int64 v54; // rdx
  _QWORD *v55; // rcx
  _QWORD *v56; // rax
  _QWORD *v57; // rcx
  __int64 v58; // rax
  unsigned int v59; // eax
  _QWORD *v60; // rdx
  __int64 v61; // rdx
  _QWORD *v62; // rcx
  _QWORD *v63; // rax
  _QWORD *v64; // rcx
  __int64 v65; // rax
  __int64 v66; // rdx
  __int64 v67; // rsi
  __int64 v68; // rcx
  unsigned int v69; // r10d
  __int64 *v70; // r9
  int v71; // eax
  __int64 v72; // rdi
  unsigned int v73; // r15d
  __int64 **v74; // rax
  __int64 **v75; // rsi
  __int64 **v76; // rcx
  unsigned __int64 v77; // r9
  __int64 v78; // rax
  __int64 **v79; // rax
  __int64 v80; // rdi
  __int64 v81; // rsi
  __int64 v82; // rsi
  _QWORD *v83; // rdi
  _QWORD *v84; // rsi
  _QWORD *v85; // rdx
  _BYTE *v86; // rax
  __int64 v87; // rcx
  __int64 v88; // r8
  _QWORD *v89; // rdx
  int v90; // eax
  int v91; // r10d
  __int64 *v92; // [rsp+18h] [rbp-6C8h]
  __int64 *v93; // [rsp+18h] [rbp-6C8h]
  _QWORD *v94; // [rsp+20h] [rbp-6C0h]
  _QWORD *v95; // [rsp+20h] [rbp-6C0h]
  _BYTE ***v96; // [rsp+20h] [rbp-6C0h]
  __int64 v97; // [rsp+28h] [rbp-6B8h]
  __int64 v98; // [rsp+28h] [rbp-6B8h]
  __int64 *v99; // [rsp+28h] [rbp-6B8h]
  unsigned __int64 v100; // [rsp+28h] [rbp-6B8h]
  _QWORD *v101; // [rsp+30h] [rbp-6B0h]
  __int64 *v102; // [rsp+30h] [rbp-6B0h]
  __int64 v103; // [rsp+30h] [rbp-6B0h]
  __int64 v104; // [rsp+30h] [rbp-6B0h]
  __int64 v105; // [rsp+30h] [rbp-6B0h]
  __int64 *v106; // [rsp+30h] [rbp-6B0h]
  __int64 v107; // [rsp+38h] [rbp-6A8h]
  unsigned int v108; // [rsp+38h] [rbp-6A8h]
  __int64 v109; // [rsp+38h] [rbp-6A8h]
  _BYTE *v110; // [rsp+40h] [rbp-6A0h] BYREF
  __int64 v111; // [rsp+48h] [rbp-698h]
  _BYTE v112[256]; // [rsp+50h] [rbp-690h] BYREF
  __int64 v113; // [rsp+150h] [rbp-590h] BYREF
  __int64 **v114; // [rsp+158h] [rbp-588h]
  __int64 **v115; // [rsp+160h] [rbp-580h]
  __int64 v116; // [rsp+168h] [rbp-578h]
  int v117; // [rsp+170h] [rbp-570h]
  _BYTE v118[264]; // [rsp+178h] [rbp-568h] BYREF
  _BYTE **v119; // [rsp+280h] [rbp-460h] BYREF
  _BYTE *v120; // [rsp+288h] [rbp-458h]
  _BYTE *v121; // [rsp+290h] [rbp-450h] BYREF
  __int64 v122; // [rsp+298h] [rbp-448h]
  int v123; // [rsp+2A0h] [rbp-440h]
  _BYTE v124[488]; // [rsp+2A8h] [rbp-438h] BYREF
  _BYTE *v125; // [rsp+490h] [rbp-250h] BYREF
  __int64 v126; // [rsp+498h] [rbp-248h]
  _BYTE v127[576]; // [rsp+4A0h] [rbp-240h] BYREF

  v2 = a2;
  v3 = a1;
  v4 = *a1;
  v125 = v127;
  v5 = *(_BYTE *)(v4 + 96) == 0;
  v126 = 0x2000000000LL;
  if ( v5 )
  {
    HIDWORD(v120) = 32;
    v119 = &v121;
    v6 = *(_QWORD *)(v4 + 80);
    if ( v6 )
    {
      v7 = *(_QWORD *)(v6 + 24);
      v8 = (unsigned __int64 *)&v121;
      v9 = 1;
      v121 = *(_BYTE **)(v4 + 80);
      v107 = v4;
      v10 = &v119;
      v122 = v7;
      v11 = 1;
      LODWORD(v120) = 1;
      v101 = v3;
      *(_DWORD *)(v6 + 48) = 0;
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
          LODWORD(v120) = v11;
          if ( !v11 )
            goto LABEL_9;
        }
        v12 = *v18;
        v17[1] = (unsigned __int64)(v18 + 1);
        v13 = (unsigned int)v120;
        v14 = *(_QWORD *)(v12 + 24);
        if ( (unsigned int)v120 >= HIDWORD(v120) )
        {
          v96 = v10;
          sub_16CD150(v10, &v121, 0, 16);
          v8 = (unsigned __int64 *)v119;
          v13 = (unsigned int)v120;
          v10 = v96;
        }
        v15 = &v8[2 * v13];
        *v15 = v12;
        v15[1] = v14;
        LODWORD(v120) = (_DWORD)v120 + 1;
        v11 = (unsigned int)v120;
        *(_DWORD *)(v12 + 48) = v16;
        v8 = (unsigned __int64 *)v119;
      }
      while ( v11 );
LABEL_9:
      v3 = v101;
      v2 = a2;
      *(_DWORD *)(v107 + 100) = 0;
      *(_BYTE *)(v107 + 96) = 1;
      if ( v8 != (unsigned __int64 *)&v121 )
        _libc_free((unsigned __int64)v8);
      v19 = v101[3];
      v20 = (unsigned int)v126;
      v21 = *(_QWORD **)(v19 + 16);
      if ( v21 == *(_QWORD **)(v19 + 8) )
        goto LABEL_12;
      goto LABEL_15;
    }
  }
  else
  {
    *(_DWORD *)(v4 + 100) = 0;
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
      v41 = *(_DWORD *)(*v3 + 72LL);
      if ( v41 )
      {
        v42 = *(_QWORD *)(*v3 + 56LL);
        v43 = (v41 - 1) & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
        v44 = (__int64 *)(v42 + 16LL * v43);
        v45 = *v44;
        if ( v23 == *v44 )
        {
LABEL_49:
          if ( v44 != (__int64 *)(v42 + 16LL * v41) )
          {
            v46 = v44[1];
            if ( v46 )
            {
              v47 = ((unsigned __int64)*(unsigned int *)(v46 + 48) << 32) | *(unsigned int *)(v46 + 16);
              if ( HIDWORD(v126) <= (unsigned int)v20 )
              {
                v109 = v44[1];
                sub_16CD150(&v125, v127, 0, 16);
                LODWORD(v20) = v126;
                v46 = v109;
              }
              v48 = &v125[16 * (unsigned int)v20];
              *v48 = v46;
              v49 = (__int64)v125;
              v48[1] = v47;
              LODWORD(v126) = v126 + 1;
              sub_14DEB00(
                v49,
                ((16LL * (unsigned int)v126) >> 4) - 1,
                0,
                *(_QWORD *)(v49 + 16LL * (unsigned int)v126 - 16),
                *(_QWORD *)(v49 + 16LL * (unsigned int)v126 - 8));
              v20 = (unsigned int)v126;
            }
          }
        }
        else
        {
          v90 = 1;
          while ( v45 != -8 )
          {
            v91 = v90 + 1;
            v43 = (v41 - 1) & (v90 + v43);
            v44 = (__int64 *)(v42 + 16LL * v43);
            v45 = *v44;
            if ( v23 == *v44 )
              goto LABEL_49;
            v90 = v91;
          }
        }
      }
      v50 = v24 + 1;
      if ( v24 + 1 == v22 )
        break;
      while ( 1 )
      {
        v23 = *v50;
        v24 = v50;
        if ( *v50 < 0xFFFFFFFFFFFFFFFELL )
          break;
        if ( v22 == ++v50 )
          goto LABEL_19;
      }
    }
  }
LABEL_19:
  v113 = 0;
  v110 = v112;
  v111 = 0x2000000000LL;
  v114 = (__int64 **)v118;
  v115 = (__int64 **)v118;
  v116 = 32;
  v117 = 0;
  v119 = 0;
  v120 = v124;
  v121 = v124;
  v122 = 32;
  v123 = 0;
  if ( (_DWORD)v20 )
  {
    while ( 1 )
    {
      v25 = (__int64)v125;
      v26 = *(_QWORD *)v125;
      v108 = *((_DWORD *)v125 + 2);
      if ( v20 != 1 )
      {
        v86 = &v125[16 * v20];
        v87 = *((_QWORD *)v86 - 2);
        v88 = *((_QWORD *)v86 - 1);
        *((_QWORD *)v86 - 2) = v26;
        *((_DWORD *)v86 - 2) = *(_DWORD *)(v25 + 8);
        *((_DWORD *)v86 - 1) = *(_DWORD *)(v25 + 12);
        sub_14DEBB0(v25, 0, (__int64)&v86[-v25 - 16] >> 4, v87, v88);
      }
      LODWORD(v126) = v126 - 1;
      v27 = 0;
      LODWORD(v111) = 0;
      if ( !HIDWORD(v111) )
      {
        sub_16CD150(&v110, v112, 0, 8);
        v27 = 8LL * (unsigned int)v111;
      }
      *(_QWORD *)&v110[v27] = v26;
      v28 = v120;
      v29 = v111 + 1;
      LODWORD(v111) = v111 + 1;
      if ( v121 != v120 )
        goto LABEL_25;
      v83 = &v120[8 * HIDWORD(v122)];
      if ( v120 != (_BYTE *)v83 )
      {
        v84 = 0;
        do
        {
          if ( v26 == *v28 )
            goto LABEL_27;
          if ( *v28 == -2 )
            v84 = v28;
          ++v28;
        }
        while ( v83 != v28 );
        if ( v84 )
        {
          *v84 = v26;
          v29 = v111;
          --v123;
          v119 = (_BYTE **)((char *)v119 + 1);
          goto LABEL_27;
        }
      }
      if ( HIDWORD(v122) >= (unsigned int)v122 )
      {
LABEL_25:
        sub_16CCBA0(&v119, v26);
        goto LABEL_26;
      }
      ++HIDWORD(v122);
      *v83 = v26;
      v29 = v111;
      v119 = (_BYTE **)((char *)v119 + 1);
LABEL_27:
      while ( v29 )
      {
        v30 = *(_QWORD **)&v110[8 * v29 - 8];
        LODWORD(v111) = v29 - 1;
        v31 = *(_QWORD *)(*v30 + 8LL);
        if ( !v31 )
          goto LABEL_31;
        while ( 1 )
        {
          v32 = sub_1648700(v31);
          if ( (unsigned __int8)(*(_BYTE *)(v32 + 16) - 25) <= 9u )
            break;
          v31 = *(_QWORD *)(v31 + 8);
          if ( !v31 )
            goto LABEL_31;
        }
LABEL_76:
        v66 = *(unsigned int *)(*v3 + 72LL);
        if ( !(_DWORD)v66 )
          goto LABEL_151;
        v67 = *(_QWORD *)(v32 + 40);
        v68 = *(_QWORD *)(*v3 + 56LL);
        v69 = (v66 - 1) & (((unsigned int)v67 >> 9) ^ ((unsigned int)v67 >> 4));
        v70 = (__int64 *)(v68 + 16LL * v69);
        v71 = 1;
        v72 = *v70;
        if ( v67 != *v70 )
        {
          while ( v72 != -8 )
          {
            v69 = (v66 - 1) & (v71 + v69);
            v70 = (__int64 *)(v68 + 16LL * v69);
            v72 = *v70;
            if ( v67 == *v70 )
              goto LABEL_78;
            ++v71;
          }
LABEL_151:
          BUG();
        }
LABEL_78:
        if ( v70 == (__int64 *)(v68 + 16 * v66) )
          goto LABEL_151;
        v51 = (__int64 *)v70[1];
        if ( v30 == (_QWORD *)v51[1] )
          goto LABEL_74;
        v73 = *((_DWORD *)v51 + 4);
        if ( v108 < v73 )
          goto LABEL_74;
        v74 = v114;
        if ( v115 != v114 )
          goto LABEL_58;
        v75 = &v114[HIDWORD(v116)];
        if ( v114 != v75 )
        {
          v76 = 0;
          do
          {
            if ( v51 == *v74 )
              goto LABEL_74;
            if ( *v74 == (__int64 *)-2LL )
              v76 = v74;
            ++v74;
          }
          while ( v75 != v74 );
          if ( v76 )
          {
            *v76 = v51;
            --v117;
            ++v113;
            goto LABEL_59;
          }
        }
        if ( HIDWORD(v116) < (unsigned int)v116 )
        {
          ++HIDWORD(v116);
          *v75 = v51;
          ++v113;
        }
        else
        {
LABEL_58:
          v102 = (__int64 *)v70[1];
          sub_16CCBA0(&v113, v102);
          v51 = v102;
          if ( !v52 )
            goto LABEL_74;
        }
LABEL_59:
        v53 = *v51;
        if ( *((_BYTE *)v3 + 8) )
        {
          v54 = v3[2];
          v55 = *(_QWORD **)(v54 + 16);
          v56 = *(_QWORD **)(v54 + 8);
          if ( v55 == v56 )
          {
            v57 = &v56[*(unsigned int *)(v54 + 28)];
            if ( v56 == v57 )
            {
              v89 = *(_QWORD **)(v54 + 8);
            }
            else
            {
              do
              {
                if ( v53 == *v56 )
                  break;
                ++v56;
              }
              while ( v57 != v56 );
              v89 = v57;
            }
          }
          else
          {
            v92 = v51;
            v97 = *v51;
            v103 = v3[2];
            v94 = &v55[*(unsigned int *)(v54 + 24)];
            v56 = (_QWORD *)sub_16CC9F0(v103, *v51);
            v53 = v97;
            v57 = v94;
            v51 = v92;
            if ( v97 == *v56 )
            {
              v82 = *(_QWORD *)(v103 + 16);
              if ( v82 == *(_QWORD *)(v103 + 8) )
                v89 = (_QWORD *)(v82 + 8LL * *(unsigned int *)(v103 + 28));
              else
                v89 = (_QWORD *)(v82 + 8LL * *(unsigned int *)(v103 + 24));
            }
            else
            {
              v58 = *(_QWORD *)(v103 + 16);
              if ( v58 != *(_QWORD *)(v103 + 8) )
              {
                v56 = (_QWORD *)(v58 + 8LL * *(unsigned int *)(v103 + 24));
                goto LABEL_64;
              }
              v89 = (_QWORD *)(v58 + 8LL * *(unsigned int *)(v103 + 28));
              v56 = v89;
            }
          }
          while ( v89 != v56 && *v56 >= 0xFFFFFFFFFFFFFFFELL )
            ++v56;
LABEL_64:
          if ( v57 == v56 )
            goto LABEL_74;
        }
        v59 = *(_DWORD *)(v2 + 8);
        if ( v59 >= *(_DWORD *)(v2 + 12) )
        {
          v99 = v51;
          v105 = v53;
          sub_16CD150(v2, v2 + 16, 0, 8);
          v59 = *(_DWORD *)(v2 + 8);
          v51 = v99;
          v53 = v105;
        }
        v60 = (_QWORD *)(*(_QWORD *)v2 + 8LL * v59);
        if ( v60 )
        {
          *v60 = v53;
          v59 = *(_DWORD *)(v2 + 8);
        }
        *(_DWORD *)(v2 + 8) = v59 + 1;
        v61 = v3[3];
        v62 = *(_QWORD **)(v61 + 16);
        v63 = *(_QWORD **)(v61 + 8);
        if ( v62 == v63 )
        {
          v64 = &v63[*(unsigned int *)(v61 + 28)];
          if ( v63 == v64 )
          {
            v85 = *(_QWORD **)(v61 + 8);
          }
          else
          {
            do
            {
              if ( v53 == *v63 )
                break;
              ++v63;
            }
            while ( v64 != v63 );
            v85 = v64;
          }
LABEL_100:
          while ( v85 != v63 )
          {
            if ( *v63 < 0xFFFFFFFFFFFFFFFELL )
              goto LABEL_73;
            ++v63;
          }
          if ( v64 != v63 )
            goto LABEL_74;
        }
        else
        {
          v93 = v51;
          v98 = v53;
          v104 = v3[3];
          v95 = &v62[*(unsigned int *)(v61 + 24)];
          v63 = (_QWORD *)sub_16CC9F0(v104, v53);
          v64 = v95;
          v51 = v93;
          if ( v98 == *v63 )
          {
            v81 = *(_QWORD *)(v104 + 16);
            if ( v81 == *(_QWORD *)(v104 + 8) )
              v85 = (_QWORD *)(v81 + 8LL * *(unsigned int *)(v104 + 28));
            else
              v85 = (_QWORD *)(v81 + 8LL * *(unsigned int *)(v104 + 24));
            goto LABEL_100;
          }
          v65 = *(_QWORD *)(v104 + 16);
          if ( v65 == *(_QWORD *)(v104 + 8) )
          {
            v63 = (_QWORD *)(v65 + 8LL * *(unsigned int *)(v104 + 28));
            v85 = v63;
            goto LABEL_100;
          }
          v63 = (_QWORD *)(v65 + 8LL * *(unsigned int *)(v104 + 24));
LABEL_73:
          if ( v64 != v63 )
            goto LABEL_74;
        }
        v77 = ((unsigned __int64)*((unsigned int *)v51 + 12) << 32) | v73;
        v78 = (unsigned int)v126;
        if ( (unsigned int)v126 >= HIDWORD(v126) )
        {
          v100 = ((unsigned __int64)*((unsigned int *)v51 + 12) << 32) | v73;
          v106 = v51;
          sub_16CD150(&v125, v127, 0, 16);
          v78 = (unsigned int)v126;
          v77 = v100;
          v51 = v106;
        }
        v79 = (__int64 **)&v125[16 * v78];
        *v79 = v51;
        v80 = (__int64)v125;
        v79[1] = (__int64 *)v77;
        LODWORD(v126) = v126 + 1;
        sub_14DEB00(
          v80,
          ((16LL * (unsigned int)v126) >> 4) - 1,
          0,
          *(_QWORD *)(v80 + 16LL * (unsigned int)v126 - 16),
          *(_QWORD *)(v80 + 16LL * (unsigned int)v126 - 8));
LABEL_74:
        while ( 1 )
        {
          v31 = *(_QWORD *)(v31 + 8);
          if ( !v31 )
            break;
          v32 = sub_1648700(v31);
          if ( (unsigned __int8)(*(_BYTE *)(v32 + 16) - 25) <= 9u )
            goto LABEL_76;
        }
LABEL_31:
        v33 = (_QWORD *)v30[4];
        v34 = (_QWORD *)v30[3];
        if ( v33 != v34 )
        {
          while ( 1 )
          {
            v36 = *v34;
            v37 = v120;
            if ( v121 == v120 )
            {
              v38 = &v120[8 * HIDWORD(v122)];
              if ( v120 != (_BYTE *)v38 )
              {
                v39 = 0;
                while ( v36 != *v37 )
                {
                  if ( *v37 == -2 )
                    v39 = v37;
                  if ( v38 == ++v37 )
                  {
                    if ( !v39 )
                      goto LABEL_107;
                    *v39 = v36;
                    --v123;
                    v119 = (_BYTE **)((char *)v119 + 1);
                    goto LABEL_44;
                  }
                }
                goto LABEL_34;
              }
LABEL_107:
              if ( HIDWORD(v122) < (unsigned int)v122 )
                break;
            }
            sub_16CCBA0(&v119, *v34);
            if ( v35 )
            {
LABEL_44:
              v40 = (unsigned int)v111;
              if ( (unsigned int)v111 < HIDWORD(v111) )
              {
LABEL_45:
                *(_QWORD *)&v110[8 * v40] = v36;
                LODWORD(v111) = v111 + 1;
                goto LABEL_34;
              }
LABEL_109:
              sub_16CD150(&v110, v112, 0, 8);
              v40 = (unsigned int)v111;
              goto LABEL_45;
            }
LABEL_34:
            if ( v33 == ++v34 )
              goto LABEL_26;
          }
          ++HIDWORD(v122);
          *v38 = v36;
          v40 = (unsigned int)v111;
          v119 = (_BYTE **)((char *)v119 + 1);
          if ( (unsigned int)v111 < HIDWORD(v111) )
            goto LABEL_45;
          goto LABEL_109;
        }
LABEL_26:
        v29 = v111;
      }
      v20 = (unsigned int)v126;
      if ( !(_DWORD)v126 )
      {
        if ( v121 != v120 )
          _libc_free((unsigned __int64)v121);
        if ( v115 != v114 )
          _libc_free((unsigned __int64)v115);
        break;
      }
    }
  }
  if ( v110 != v112 )
    _libc_free((unsigned __int64)v110);
  if ( v125 != v127 )
    _libc_free((unsigned __int64)v125);
}
