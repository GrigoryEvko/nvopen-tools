// Function: sub_30A4D70
// Address: 0x30a4d70
//
__int64 __fastcall sub_30A4D70(__int64 a1, _QWORD *a2)
{
  __int64 **v2; // rax
  __int64 **v3; // rdx
  __int64 *v4; // r14
  __int64 v5; // r15
  __int64 v6; // rbx
  __int64 v7; // r15
  __int64 v8; // r12
  __int64 v9; // r13
  __int64 v10; // rax
  __int64 v11; // rax
  _QWORD *v12; // rdi
  __int64 v13; // r13
  __int64 v14; // rbx
  __int64 v15; // rax
  __int64 *v16; // rbx
  __int64 *v17; // r12
  char *v18; // rdi
  unsigned __int8 *v19; // r8
  unsigned __int8 *v20; // rsi
  __int64 v21; // rax
  __int64 v22; // r8
  __int64 v23; // rsi
  unsigned int v24; // ecx
  unsigned __int8 **v25; // rax
  unsigned __int8 *v26; // r8
  unsigned __int8 *v27; // rbx
  unsigned __int64 v28; // rsi
  __int64 v29; // rax
  __int64 v30; // rcx
  bool v31; // bl
  __int64 v32; // rax
  __int64 v33; // rax
  bool v34; // zf
  int v35; // eax
  unsigned __int64 v36; // rsi
  __int64 v37; // rax
  __int64 v38; // r12
  __int64 v39; // rax
  char v40; // bl
  int v41; // r9d
  __int64 v42; // rax
  unsigned __int8 *v43; // r12
  int v44; // eax
  unsigned __int64 v45; // rax
  __int64 v46; // rax
  __int64 v47; // r13
  __int64 v48; // rax
  __int64 v49; // rax
  __int64 v50; // rax
  __int64 v51; // rax
  __int64 v52; // rdi
  __int64 v53; // rsi
  __int64 v55; // rax
  __int64 v56; // rax
  __int64 v57; // rcx
  unsigned int v58; // edi
  int v59; // r8d
  unsigned int v60; // edx
  __int64 v61; // r9
  unsigned __int8 **v62; // rax
  unsigned __int8 *v63; // rcx
  unsigned __int8 *v64; // r11
  __int64 v65; // r8
  unsigned __int8 *v66; // r13
  __int64 v67; // rdx
  unsigned int v68; // ecx
  _QWORD *v69; // rdx
  _QWORD *v70; // rax
  __int64 v71; // rdx
  __int64 v72; // rax
  int v73; // edx
  unsigned int v74; // ecx
  unsigned __int8 *v75; // r8
  int v76; // edi
  unsigned __int8 **v77; // rsi
  int v78; // r9d
  unsigned __int8 **v79; // r8
  int v80; // edi
  unsigned int v81; // ecx
  unsigned __int8 *v82; // r8
  _QWORD *v83; // rdi
  __int64 v84; // rsi
  unsigned int v85; // edx
  int v86; // edx
  unsigned __int64 v87; // rax
  unsigned __int64 v88; // rax
  unsigned int v89; // ebx
  __int64 v90; // r12
  _QWORD *v91; // rax
  _QWORD *i; // rdx
  _QWORD *v93; // rsi
  unsigned int v94; // [rsp+8h] [rbp-148h]
  unsigned int v95; // [rsp+Ch] [rbp-144h]
  __int64 **v96; // [rsp+10h] [rbp-140h]
  unsigned int v97; // [rsp+18h] [rbp-138h]
  char v98; // [rsp+1Ch] [rbp-134h]
  unsigned int v99; // [rsp+28h] [rbp-128h]
  unsigned __int8 v100; // [rsp+2Fh] [rbp-121h]
  __int64 **v101; // [rsp+30h] [rbp-120h]
  __int64 v102; // [rsp+38h] [rbp-118h]
  __int64 v103; // [rsp+40h] [rbp-110h]
  __int64 v104; // [rsp+50h] [rbp-100h]
  __int64 v105; // [rsp+70h] [rbp-E0h]
  unsigned __int8 *v106; // [rsp+70h] [rbp-E0h]
  __int64 v108; // [rsp+88h] [rbp-C8h] BYREF
  __int64 v109; // [rsp+90h] [rbp-C0h] BYREF
  char *v110; // [rsp+98h] [rbp-B8h]
  int v111; // [rsp+A0h] [rbp-B0h]
  char v112; // [rsp+A8h] [rbp-A8h] BYREF
  __int64 v113; // [rsp+B0h] [rbp-A0h] BYREF
  _QWORD *v114; // [rsp+B8h] [rbp-98h]
  __int64 v115; // [rsp+C0h] [rbp-90h]
  unsigned int v116; // [rsp+C8h] [rbp-88h]
  _QWORD v117[2]; // [rsp+D0h] [rbp-80h] BYREF
  __int64 v118; // [rsp+E0h] [rbp-70h]
  __int64 v119; // [rsp+E8h] [rbp-68h]
  __int64 v120; // [rsp+F0h] [rbp-60h] BYREF
  __int64 v121; // [rsp+F8h] [rbp-58h]
  __int64 v122; // [rsp+100h] [rbp-50h] BYREF
  char v123; // [rsp+108h] [rbp-48h]

  v2 = *(__int64 ***)(a1 + 24);
  v3 = *(__int64 ***)(a1 + 16);
  v113 = 0;
  v114 = 0;
  v115 = 0;
  v116 = 0;
  v96 = v2;
  if ( v3 == v2 )
  {
    v100 = 0;
    v52 = 0;
    v53 = 0;
    goto LABEL_133;
  }
  v101 = v3;
  v100 = 0;
  v98 = 0;
  while ( 2 )
  {
    v4 = *v101;
    v5 = (*v101)[1];
    if ( !v5 || sub_B2FC80((*v101)[1]) )
      goto LABEL_3;
    v6 = v4[2];
    v95 = 0;
    v94 = 0;
    if ( v4[3] != v6 )
    {
      v105 = v5;
      v7 = v4[3];
      while ( 1 )
      {
        if ( !*(_BYTE *)(v6 + 24) )
        {
          v8 = v6 + 40;
          --*(_DWORD *)(*(_QWORD *)(v6 + 32) + 40LL);
          v9 = v4[3];
          if ( *(_BYTE *)(v6 + 24) )
          {
            if ( *(_BYTE *)(v9 - 16) )
            {
              v71 = *(_QWORD *)(v6 + 16);
              v72 = *(_QWORD *)(v9 - 24);
              if ( v71 != v72 )
              {
                if ( v71 != 0 && v71 != -4096 && v71 != -8192 )
                {
                  sub_BD60C0((_QWORD *)v6);
                  v72 = *(_QWORD *)(v9 - 24);
                }
                *(_QWORD *)(v6 + 16) = v72;
                if ( v72 != -4096 && v72 != 0 && v72 != -8192 )
                  sub_BD6050((unsigned __int64 *)v6, *(_QWORD *)(v9 - 40) & 0xFFFFFFFFFFFFFFF8LL);
              }
            }
            else
            {
              v10 = *(_QWORD *)(v6 + 16);
              *(_BYTE *)(v6 + 24) = 0;
              if ( v10 != -4096 && v10 != 0 && v10 != -8192 )
                sub_BD60C0((_QWORD *)v6);
            }
          }
          else if ( *(_BYTE *)(v9 - 16) )
          {
            *(_QWORD *)v6 = 6;
            v51 = *(_QWORD *)(v9 - 24);
            *(_QWORD *)(v6 + 8) = 0;
            *(_QWORD *)(v6 + 16) = v51;
            if ( v51 != -4096 && v51 != 0 && v51 != -8192 )
              sub_BD6050((unsigned __int64 *)v6, *(_QWORD *)(v9 - 40) & 0xFFFFFFFFFFFFFFF8LL);
            *(_BYTE *)(v6 + 24) = 1;
          }
          *(_QWORD *)(v6 + 32) = *(_QWORD *)(v9 - 8);
          v11 = v4[3];
          v12 = (_QWORD *)(v11 - 40);
          v4[3] = v11 - 40;
          if ( *(_BYTE *)(v11 - 16) )
          {
            *(_BYTE *)(v11 - 16) = 0;
            v56 = *(_QWORD *)(v11 - 24);
            if ( v56 != -4096 && v56 != 0 && v56 != -8192 )
              goto LABEL_126;
          }
          goto LABEL_15;
        }
        v43 = *(unsigned __int8 **)(v6 + 16);
        if ( !v43
          || (v44 = *v43, (unsigned __int8)v44 <= 0x1Cu)
          || (v45 = (unsigned int)(v44 - 34), (unsigned __int8)v45 > 0x33u)
          || (v57 = 0x8000000000041LL, !_bittest64(&v57, v45)) )
        {
LABEL_116:
          v46 = *(_QWORD *)(v6 + 32);
          if ( *(_QWORD *)(v46 + 8) )
            ++v95;
          else
            ++v94;
          --*(_DWORD *)(v46 + 40);
          v8 = v6 + 40;
          v47 = v4[3];
          if ( *(_BYTE *)(v6 + 24) )
          {
            v48 = *(_QWORD *)(v6 + 16);
            if ( *(_BYTE *)(v47 - 16) )
            {
              v67 = *(_QWORD *)(v47 - 24);
              if ( v48 != v67 )
              {
                if ( v48 != -4096 && v48 != 0 && v48 != -8192 )
                {
                  sub_BD60C0((_QWORD *)v6);
                  v67 = *(_QWORD *)(v47 - 24);
                }
                *(_QWORD *)(v6 + 16) = v67;
                if ( v67 != 0 && v67 != -4096 && v67 != -8192 )
                  sub_BD6050((unsigned __int64 *)v6, *(_QWORD *)(v47 - 40) & 0xFFFFFFFFFFFFFFF8LL);
              }
            }
            else
            {
              *(_BYTE *)(v6 + 24) = 0;
              if ( v48 != -4096 && v48 != 0 && v48 != -8192 )
                sub_BD60C0((_QWORD *)v6);
            }
          }
          else if ( *(_BYTE *)(v47 - 16) )
          {
            *(_QWORD *)v6 = 6;
            v55 = *(_QWORD *)(v47 - 24);
            *(_QWORD *)(v6 + 8) = 0;
            *(_QWORD *)(v6 + 16) = v55;
            if ( v55 != 0 && v55 != -4096 && v55 != -8192 )
              sub_BD6050((unsigned __int64 *)v6, *(_QWORD *)(v47 - 40) & 0xFFFFFFFFFFFFFFF8LL);
            *(_BYTE *)(v6 + 24) = 1;
          }
          *(_QWORD *)(v6 + 32) = *(_QWORD *)(v47 - 8);
          v49 = v4[3];
          v12 = (_QWORD *)(v49 - 40);
          v4[3] = v49 - 40;
          if ( *(_BYTE *)(v49 - 16) )
          {
            *(_BYTE *)(v49 - 16) = 0;
            v50 = *(_QWORD *)(v49 - 24);
            if ( v50 != 0 && v50 != -4096 && v50 != -8192 )
LABEL_126:
              sub_BD60C0(v12);
          }
LABEL_15:
          if ( v7 == v8 )
            goto LABEL_18;
          v7 = v4[3];
          goto LABEL_17;
        }
        if ( v116 )
        {
          v58 = v116 - 1;
          v59 = 1;
          v60 = (v116 - 1) & (((unsigned int)v43 >> 9) ^ ((unsigned int)v43 >> 4));
          LODWORD(v61) = v60;
          v62 = (unsigned __int8 **)&v114[2 * v60];
          v63 = *v62;
          v64 = *v62;
          if ( *v62 == v43 )
            goto LABEL_116;
          while ( v64 != (unsigned __int8 *)-4096LL )
          {
            v61 = v58 & ((_DWORD)v61 + v59);
            v64 = (unsigned __int8 *)v114[2 * v61];
            if ( v43 == v64 )
              goto LABEL_116;
            ++v59;
          }
          v65 = *((_QWORD *)v43 - 4);
          if ( !v65 )
          {
            v66 = *(unsigned __int8 **)(v6 + 32);
LABEL_192:
            v78 = 1;
            v79 = 0;
            while ( v63 != (unsigned __int8 *)-4096LL )
            {
              if ( v63 == (unsigned __int8 *)-8192LL && !v79 )
                v79 = v62;
              v60 = v58 & (v78 + v60);
              v62 = (unsigned __int8 **)&v114[2 * v60];
              v63 = *v62;
              if ( v43 == *v62 )
                goto LABEL_151;
              ++v78;
            }
            if ( v79 )
              v62 = v79;
            ++v113;
            v73 = v115 + 1;
            if ( 4 * ((int)v115 + 1) < 3 * v116 )
            {
              if ( v116 - HIDWORD(v115) - v73 <= v116 >> 3 )
              {
                sub_30A4B90((__int64)&v113, v116);
                if ( !v116 )
                {
LABEL_235:
                  LODWORD(v115) = v115 + 1;
                  BUG();
                }
                v77 = 0;
                v73 = v115 + 1;
                v80 = 1;
                v81 = (v116 - 1) & (((unsigned int)v43 >> 9) ^ ((unsigned int)v43 >> 4));
                v62 = (unsigned __int8 **)&v114[2 * v81];
                v82 = *v62;
                if ( v43 != *v62 )
                {
                  while ( v82 != (unsigned __int8 *)-4096LL )
                  {
                    if ( !v77 && v82 == (unsigned __int8 *)-8192LL )
                      v77 = v62;
                    v81 = (v116 - 1) & (v80 + v81);
                    v62 = (unsigned __int8 **)&v114[2 * v81];
                    v82 = *v62;
                    if ( v43 == *v62 )
                      goto LABEL_198;
                    ++v80;
                  }
                  goto LABEL_204;
                }
              }
              goto LABEL_198;
            }
LABEL_182:
            sub_30A4B90((__int64)&v113, 2 * v116);
            if ( !v116 )
              goto LABEL_235;
            v73 = v115 + 1;
            v74 = (v116 - 1) & (((unsigned int)v43 >> 9) ^ ((unsigned int)v43 >> 4));
            v62 = (unsigned __int8 **)&v114[2 * v74];
            v75 = *v62;
            if ( v43 != *v62 )
            {
              v76 = 1;
              v77 = 0;
              while ( v75 != (unsigned __int8 *)-4096LL )
              {
                if ( v75 == (unsigned __int8 *)-8192LL && !v77 )
                  v77 = v62;
                v74 = (v116 - 1) & (v76 + v74);
                v62 = (unsigned __int8 **)&v114[2 * v74];
                v75 = *v62;
                if ( v43 == *v62 )
                  goto LABEL_198;
                ++v76;
              }
LABEL_204:
              if ( v77 )
                v62 = v77;
            }
LABEL_198:
            LODWORD(v115) = v73;
            if ( *v62 != (unsigned __int8 *)-4096LL )
              --HIDWORD(v115);
            *v62 = v43;
            v62[1] = v66;
            goto LABEL_151;
          }
          if ( *(_BYTE *)v65 || *((_QWORD *)v43 + 10) != *(_QWORD *)(v65 + 24) )
          {
            v66 = *(unsigned __int8 **)(v6 + 32);
            goto LABEL_150;
          }
        }
        else
        {
          v65 = *((_QWORD *)v43 - 4);
          if ( !v65 || *(_BYTE *)v65 || *(_QWORD *)(v65 + 24) != *((_QWORD *)v43 + 10) )
          {
            v66 = *(unsigned __int8 **)(v6 + 32);
LABEL_181:
            ++v113;
            goto LABEL_182;
          }
        }
        if ( (*(_BYTE *)(v65 + 33) & 0x20) != 0 )
          goto LABEL_151;
        v66 = *(unsigned __int8 **)(v6 + 32);
        if ( !v116 )
          goto LABEL_181;
        v58 = v116 - 1;
LABEL_150:
        v60 = v58 & (((unsigned int)v43 >> 9) ^ ((unsigned int)v43 >> 4));
        v62 = (unsigned __int8 **)&v114[2 * v60];
        v63 = *v62;
        if ( v43 != *v62 )
          goto LABEL_192;
LABEL_151:
        v6 += 40;
LABEL_17:
        if ( v6 == v7 )
        {
LABEL_18:
          v5 = v105;
          break;
        }
      }
    }
    v103 = *(_QWORD *)(v5 + 80);
    if ( v103 == v5 + 72 )
      goto LABEL_65;
    v99 = 0;
    v97 = 0;
    do
    {
      if ( !v103 )
        BUG();
      v13 = *(_QWORD *)(v103 + 32);
      v104 = v103 + 24;
      if ( v13 == v103 + 24 )
        goto LABEL_61;
      do
      {
        while ( 1 )
        {
LABEL_23:
          if ( !v13 )
            BUG();
          v106 = (unsigned __int8 *)(v13 - 24);
          if ( (unsigned __int8)(*(_BYTE *)(v13 - 24) - 34) > 0x33u )
            goto LABEL_60;
          v14 = 0x8000000000041LL;
          if ( !_bittest64(&v14, (unsigned int)*(unsigned __int8 *)(v13 - 24) - 34) )
            goto LABEL_60;
          v15 = *(_QWORD *)(v13 - 56);
          if ( !v15
            || *(_BYTE *)v15
            || *(_QWORD *)(v15 + 24) != *(_QWORD *)(v13 + 56)
            || (*(_BYTE *)(v15 + 33) & 0x20) == 0 )
          {
            break;
          }
          v13 = *(_QWORD *)(v13 + 8);
          if ( v104 == v13 )
            goto LABEL_61;
        }
        v120 = (__int64)&v122;
        v121 = 0x400000000LL;
        sub_E33A00(v106, (__int64)&v120);
        v16 = (__int64 *)v120;
        v17 = (__int64 *)(v120 + 8LL * (unsigned int)v121);
        if ( (__int64 *)v120 == v17 )
          goto LABEL_49;
        do
        {
          v23 = *v16;
          sub_E33C60(&v109, *v16);
          if ( v111 || sub_B491E0(v109) )
          {
            v18 = v110;
            v19 = *(unsigned __int8 **)(v109
                                      + 32
                                      * (*(unsigned int *)v110 - (unsigned __int64)(*(_DWORD *)(v109 + 4) & 0x7FFFFFF)));
            if ( !v19 )
              goto LABEL_41;
LABEL_32:
            v20 = sub_BD3990(v19, v23);
            if ( !*v20 )
            {
              v21 = sub_D110B0(a2, (unsigned __int64)v20);
              v119 = 0;
              v22 = v4[3];
              v108 = v21;
              if ( v22 == v4[4] )
              {
                sub_D10B90(v4 + 2, v22, (__int64)v117, &v108);
              }
              else
              {
                if ( v22 )
                {
                  *(_BYTE *)(v22 + 24) = 0;
                  if ( (_BYTE)v119 )
                  {
                    *(_QWORD *)v22 = 6;
                    *(_QWORD *)(v22 + 8) = 0;
                    v33 = v118;
                    v34 = v118 == -4096;
                    *(_QWORD *)(v22 + 16) = v118;
                    if ( v33 != 0 && !v34 && v33 != -8192 )
                    {
                      v102 = v22;
                      sub_BD6050((unsigned __int64 *)v22, v117[0] & 0xFFFFFFFFFFFFFFF8LL);
                      v22 = v102;
                    }
                    *(_BYTE *)(v22 + 24) = 1;
                  }
                  *(_QWORD *)(v22 + 32) = v108;
                  v22 = v4[3];
                }
                v4[3] = v22 + 40;
              }
              if ( (_BYTE)v119 )
              {
                LOBYTE(v119) = 0;
                if ( v118 != 0 && v118 != -4096 && v118 != -8192 )
                  sub_BD60C0(v117);
              }
              ++*(_DWORD *)(v108 + 40);
            }
            goto LABEL_40;
          }
          v19 = *(unsigned __int8 **)(v109 - 32);
          if ( v19 )
            goto LABEL_32;
LABEL_40:
          v18 = v110;
LABEL_41:
          if ( v18 != &v112 )
            _libc_free((unsigned __int64)v18);
          ++v16;
        }
        while ( v17 != v16 );
        v17 = (__int64 *)v120;
LABEL_49:
        if ( v17 != &v122 )
          _libc_free((unsigned __int64)v17);
        if ( !v116 )
          goto LABEL_80;
        v24 = (v116 - 1) & (((unsigned int)v106 >> 9) ^ ((unsigned int)v106 >> 4));
        v25 = (unsigned __int8 **)&v114[2 * v24];
        v26 = *v25;
        if ( v106 != *v25 )
        {
          v35 = 1;
          while ( v26 != (unsigned __int8 *)-4096LL )
          {
            v41 = v35 + 1;
            v24 = (v116 - 1) & (v35 + v24);
            v25 = (unsigned __int8 **)&v114[2 * v24];
            v26 = *v25;
            if ( v106 == *v25 )
              goto LABEL_53;
            v35 = v41;
          }
LABEL_80:
          v36 = *(_QWORD *)(v13 - 56);
          if ( v36 && !*(_BYTE *)v36 && *(_QWORD *)(v36 + 24) == *(_QWORD *)(v13 + 56) )
          {
            v37 = sub_D110B0(a2, v36);
            ++v97;
          }
          else
          {
            ++v99;
            v37 = a2[8];
          }
          v117[0] = v37;
          v120 = 6;
          v121 = 0;
          v122 = v13 - 24;
          if ( v13 != -8168 && v13 != -4072 )
            sub_BD73F0((__int64)&v120);
          v123 = 1;
          v38 = v4[3];
          if ( v38 == v4[4] )
          {
            sub_D10B90(v4 + 2, v4[3], (__int64)&v120, v117);
          }
          else
          {
            if ( v38 )
            {
              *(_BYTE *)(v38 + 24) = 0;
              if ( v123 )
              {
                *(_QWORD *)v38 = 6;
                *(_QWORD *)(v38 + 8) = 0;
                v42 = v122;
                v34 = v122 == -4096;
                *(_QWORD *)(v38 + 16) = v122;
                if ( v42 != 0 && !v34 && v42 != -8192 )
                  sub_BD6050((unsigned __int64 *)v38, v120 & 0xFFFFFFFFFFFFFFF8LL);
                *(_BYTE *)(v38 + 24) = 1;
              }
              *(_QWORD *)(v38 + 32) = v117[0];
              v38 = v4[3];
            }
            v4[3] = v38 + 40;
          }
          if ( v123 )
          {
            v123 = 0;
            if ( v122 != 0 && v122 != -4096 && v122 != -8192 )
              sub_BD60C0(&v120);
          }
          ++*(_DWORD *)(v117[0] + 40LL);
          v13 = *(_QWORD *)(v13 + 8);
          if ( v104 == v13 )
            break;
          goto LABEL_23;
        }
LABEL_53:
        if ( v25 == &v114[2 * v116] )
          goto LABEL_80;
        *v25 = (unsigned __int8 *)-8192LL;
        v27 = v25[1];
        LODWORD(v115) = v115 - 1;
        ++HIDWORD(v115);
        v28 = *(_QWORD *)(v13 - 56);
        v29 = *((_QWORD *)v27 + 1);
        if ( v28 && !*(_BYTE *)v28 && *(_QWORD *)(v28 + 24) == *(_QWORD *)(v13 + 56) )
        {
          if ( v29 != v28 )
          {
            v39 = sub_D110B0(a2, v28);
            v34 = *((_QWORD *)v27 + 1) == 0;
            v40 = v100;
            v30 = v39;
            if ( v34 )
              v40 = 1;
            v100 = v40;
LABEL_59:
            sub_D11290(v4, (__int64)v106, (__int64)v106, v30);
          }
        }
        else if ( v29 )
        {
          v30 = a2[8];
          goto LABEL_59;
        }
LABEL_60:
        v13 = *(_QWORD *)(v13 + 8);
      }
      while ( v104 != v13 );
LABEL_61:
      v103 = *(_QWORD *)(v103 + 8);
    }
    while ( v5 + 72 != v103 );
    v31 = v100;
    if ( v95 < v97 && v94 > v99 )
      v31 = v95 < v97 && v94 > v99;
    v100 = v31;
LABEL_65:
    if ( (v98 & 0xF) == 0xF )
    {
      ++v113;
      if ( (_DWORD)v115 )
      {
        v68 = 4 * v115;
        v32 = v116;
        if ( (unsigned int)(4 * v115) < 0x40 )
          v68 = 64;
        if ( v68 >= v116 )
          goto LABEL_163;
        v83 = v114;
        v84 = 2LL * v116;
        if ( (_DWORD)v115 == 1 )
        {
          v90 = 2048;
          v89 = 128;
        }
        else
        {
          _BitScanReverse(&v85, v115 - 1);
          v86 = 1 << (33 - (v85 ^ 0x1F));
          if ( v86 < 64 )
            v86 = 64;
          if ( v86 == v116 )
          {
            v115 = 0;
            v93 = &v114[v84];
            do
            {
              if ( v83 )
                *v83 = -4096;
              v83 += 2;
            }
            while ( v93 != v83 );
            goto LABEL_3;
          }
          v87 = (4 * v86 / 3u + 1) | ((unsigned __int64)(4 * v86 / 3u + 1) >> 1);
          v88 = ((v87 | (v87 >> 2)) >> 4) | v87 | (v87 >> 2) | ((((v87 | (v87 >> 2)) >> 4) | v87 | (v87 >> 2)) >> 8);
          v89 = (v88 | (v88 >> 16)) + 1;
          v90 = 16 * ((v88 | (v88 >> 16)) + 1);
        }
        sub_C7D6A0((__int64)v114, v84 * 8, 8);
        v116 = v89;
        v91 = (_QWORD *)sub_C7D670(v90, 8);
        v115 = 0;
        v114 = v91;
        for ( i = &v91[2 * v116]; i != v91; v91 += 2 )
        {
          if ( v91 )
            *v91 = -4096;
        }
        goto LABEL_3;
      }
      if ( !HIDWORD(v115) )
        goto LABEL_3;
      v32 = v116;
      if ( v116 > 0x40 )
      {
        sub_C7D6A0((__int64)v114, 16LL * v116, 8);
        v114 = 0;
        v115 = 0;
        v116 = 0;
        goto LABEL_3;
      }
LABEL_163:
      v69 = v114;
      v70 = &v114[2 * v32];
      if ( v114 != v70 )
      {
        do
        {
          *v69 = -4096;
          v69 += 2;
        }
        while ( v70 != v69 );
      }
      v115 = 0;
    }
LABEL_3:
    ++v101;
    ++v98;
    if ( v96 != v101 )
      continue;
    break;
  }
  v52 = (__int64)v114;
  v53 = 16LL * v116;
LABEL_133:
  sub_C7D6A0(v52, v53, 8);
  return v100;
}
