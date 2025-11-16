// Function: sub_FA4540
// Address: 0xfa4540
//
__int64 __fastcall sub_FA4540(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 *v3; // r15
  __int64 *v4; // r14
  __int64 *v5; // rbx
  char v6; // r13
  unsigned __int8 *v7; // r12
  unsigned __int8 v8; // al
  unsigned int v9; // ecx
  __int64 v10; // rsi
  __int64 *v11; // r13
  __int64 v12; // r14
  char v13; // al
  char v14; // al
  char v16; // r14
  __int64 v17; // rsi
  __int64 v18; // rcx
  __int64 v19; // rdx
  __int64 v20; // r8
  int v21; // r9d
  __int64 *v22; // rdi
  unsigned int v23; // eax
  __int64 *v24; // rdx
  __int64 v25; // r11
  _QWORD *v26; // rdx
  __int64 *v27; // rax
  __int64 *v28; // rdx
  __int64 v29; // rdx
  __int64 v30; // rdx
  __int64 v31; // r12
  __int64 *v32; // rbx
  __int64 *v33; // r13
  __int64 v34; // rax
  char *v35; // r13
  unsigned __int8 *v36; // r15
  __int64 *v37; // r12
  unsigned __int8 *v38; // rdx
  __int64 v39; // r13
  __int64 *v40; // rax
  __int64 v41; // rdx
  __int64 v42; // rdx
  __int64 v43; // rdx
  __int64 v44; // rdx
  __int64 v45; // rdx
  __int64 v46; // rdx
  __int64 v47; // rdx
  __int64 v48; // rdx
  int v49; // edx
  unsigned __int8 *v50; // rdx
  unsigned __int8 *v51; // rdx
  int v52; // esi
  __int64 v53; // r11
  int v54; // r8d
  unsigned __int8 **v55; // r10
  __int64 v56; // rcx
  unsigned __int8 **v57; // rax
  unsigned __int8 *v58; // rdi
  __int64 v59; // r13
  __int64 v60; // rdx
  __int64 *v61; // rbx
  __int64 v62; // rcx
  __int64 v63; // r8
  __int64 v64; // rcx
  int v65; // edx
  int v66; // ebx
  __int64 v67; // rdx
  __int64 v68; // rdx
  __int64 v69; // rdx
  __int64 v70; // rdx
  unsigned __int8 *v71; // rdx
  __int64 v72; // rdx
  __int64 v73; // rdx
  __int64 v74; // rdx
  __int64 v75; // rdx
  signed __int64 v76; // rax
  __int64 v77; // rdx
  __int64 *v78; // rax
  __int64 *v79; // rdx
  __int64 *v80; // r14
  __int64 v81; // rax
  __int64 v82; // rax
  __int64 v83; // rax
  __int64 v84; // rax
  __int64 v85; // rax
  __int64 v86; // rax
  __int64 v87; // rax
  __int64 v88; // rax
  __int64 *v89; // rdx
  __int64 *v90; // rdx
  __int64 *v91; // rdx
  __int64 *v92; // rdx
  signed __int64 v93; // rax
  signed __int64 v94; // rax
  __int64 v95; // rax
  __int64 v96; // rax
  unsigned __int8 *v97; // rax
  __int64 *v98; // r13
  __int64 v99; // rax
  __int64 v100; // rax
  unsigned __int8 *v101; // rax
  __int64 v102; // rax
  __int64 v103; // rax
  unsigned __int8 *v104; // rax
  __int64 v105; // [rsp+0h] [rbp-C0h]
  __int64 v106; // [rsp+8h] [rbp-B8h]
  __int64 v107; // [rsp+10h] [rbp-B0h]
  int v108; // [rsp+1Ch] [rbp-A4h]
  __int64 v109; // [rsp+20h] [rbp-A0h]
  unsigned __int8 *v110; // [rsp+28h] [rbp-98h]
  __int64 *v111; // [rsp+28h] [rbp-98h]
  __int64 *v112; // [rsp+28h] [rbp-98h]
  __int64 *v113; // [rsp+28h] [rbp-98h]
  __int64 *v114; // [rsp+28h] [rbp-98h]
  __int64 v115; // [rsp+30h] [rbp-90h]
  __int64 *v116; // [rsp+30h] [rbp-90h]
  int v118; // [rsp+48h] [rbp-78h]
  __int64 *v119; // [rsp+48h] [rbp-78h]
  unsigned __int8 v120; // [rsp+48h] [rbp-78h]
  __int64 v121; // [rsp+48h] [rbp-78h]
  char v122[8]; // [rsp+50h] [rbp-70h] BYREF
  __int64 v123; // [rsp+58h] [rbp-68h]
  int v124; // [rsp+60h] [rbp-60h]
  unsigned int v125; // [rsp+68h] [rbp-58h]
  unsigned __int8 *v126; // [rsp+70h] [rbp-50h] BYREF
  __int64 v127; // [rsp+78h] [rbp-48h]
  int v128; // [rsp+80h] [rbp-40h]
  unsigned int v129; // [rsp+88h] [rbp-38h]

  v3 = a1;
  v4 = &a1[a2];
  v115 = 8 * a2;
  if ( a1 == v4 )
  {
    v17 = *a1;
    v110 = (unsigned __int8 *)*a1;
    sub_B8E070((__int64)v122, *a1);
LABEL_31:
    v18 = *((_QWORD *)v110 + 2);
    if ( v18 )
    {
      v19 = *(unsigned int *)(a3 + 24);
      v20 = *(_QWORD *)(a3 + 8);
      v17 = v19;
      v21 = v19 - 1;
      v22 = (__int64 *)(v20 + 56 * v19);
      while ( (_DWORD)v17 )
      {
        v23 = v21 & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
        v24 = (__int64 *)(v20 + 56LL * v23);
        v25 = *v24;
        if ( *v24 != v18 )
        {
          v65 = 1;
          while ( v25 != -4096 )
          {
            v66 = v65 + 1;
            v23 = v21 & (v65 + v23);
            v24 = (__int64 *)(v20 + 56LL * v23);
            v25 = *v24;
            if ( *v24 == v18 )
              goto LABEL_35;
            v65 = v66;
          }
          goto LABEL_41;
        }
LABEL_35:
        if ( v22 == v24 || v115 != 8LL * *((unsigned int *)v24 + 4) )
          goto LABEL_41;
        if ( v3 != v4 )
        {
          v26 = (_QWORD *)v24[1];
          v27 = v3;
          while ( *v27 == *v26 )
          {
            ++v27;
            ++v26;
            if ( v4 == v27 )
              goto LABEL_50;
          }
          goto LABEL_41;
        }
LABEL_50:
        v18 = *(_QWORD *)(v18 + 8);
        if ( !v18 )
          goto LABEL_51;
      }
      goto LABEL_41;
    }
LABEL_51:
    if ( (unsigned __int8)(*v110 - 34) > 0x33u )
      goto LABEL_68;
    v30 = 0x8000000000041LL;
    if ( !_bittest64(&v30, (unsigned int)*v110 - 34) )
      goto LABEL_68;
    v31 = v115 >> 5;
    if ( v115 >> 5 > 0 )
    {
      v32 = v3;
      do
      {
        if ( sub_B491E0(*v32) )
          goto LABEL_60;
        if ( sub_B491E0(v32[1]) )
        {
          ++v32;
          goto LABEL_60;
        }
        if ( sub_B491E0(v32[2]) )
        {
          v32 += 2;
          goto LABEL_60;
        }
        if ( sub_B491E0(v32[3]) )
        {
          v32 += 3;
          goto LABEL_60;
        }
        v32 += 4;
      }
      while ( &v3[4 * v31] != v32 );
      v93 = (char *)v4 - (char *)v32;
      if ( (char *)v4 - (char *)v32 != 16 )
      {
        if ( v93 != 24 )
        {
          if ( v93 == 8 )
          {
            if ( !sub_B491E0(*v32) )
              v32 = v4;
          }
          else
          {
            v32 = v4;
          }
          goto LABEL_60;
        }
        if ( !sub_B491E0(*v32) )
        {
          v98 = v32 + 1;
          if ( !sub_B491E0(v32[1]) )
            goto LABEL_236;
LABEL_242:
          v32 = v98;
        }
LABEL_60:
        v33 = v3;
        while ( sub_B491E0(*v33) )
        {
          if ( !sub_B491E0(v33[1]) )
          {
            ++v33;
            break;
          }
          if ( !sub_B491E0(v33[2]) )
          {
            v33 += 2;
            break;
          }
          if ( !sub_B491E0(v33[3]) )
          {
            v33 += 3;
            break;
          }
          v33 += 4;
          if ( !--v31 )
            goto LABEL_168;
        }
LABEL_66:
        if ( v4 != v32 )
        {
          if ( v4 != v33 )
          {
LABEL_41:
            v14 = 0;
            goto LABEL_25;
          }
          goto LABEL_68;
        }
        goto LABEL_172;
      }
      if ( sub_B491E0(*v32) )
        goto LABEL_60;
      v98 = v32 + 1;
      if ( sub_B491E0(v32[1]) )
        goto LABEL_242;
LABEL_237:
      v32 = v4;
LABEL_238:
      if ( v31 > 0 )
        goto LABEL_60;
      goto LABEL_239;
    }
    if ( v115 == 16 )
    {
      if ( !sub_B491E0(*v3) )
      {
        v32 = v3 + 1;
        if ( sub_B491E0(v3[1]) )
        {
LABEL_239:
          v33 = v3;
          goto LABEL_168;
        }
        goto LABEL_237;
      }
    }
    else if ( v115 == 24 )
    {
      if ( !sub_B491E0(*v3) )
      {
        v98 = v3 + 1;
        if ( !sub_B491E0(v3[1]) )
        {
LABEL_236:
          v32 = v98 + 1;
          if ( sub_B491E0(v98[1]) )
            goto LABEL_238;
          goto LABEL_237;
        }
        v32 = v3 + 1;
        v33 = v3;
LABEL_168:
        v76 = (char *)v4 - (char *)v33;
        if ( (char *)v4 - (char *)v33 != 16 )
        {
          if ( v76 != 24 )
          {
            if ( v76 != 8 )
              goto LABEL_171;
            goto LABEL_232;
          }
          if ( !sub_B491E0(*v33) )
            goto LABEL_66;
          ++v33;
        }
        if ( !sub_B491E0(*v33) )
          goto LABEL_66;
        ++v33;
LABEL_232:
        if ( !sub_B491E0(*v33) )
          goto LABEL_66;
LABEL_171:
        if ( v4 != v32 )
        {
LABEL_68:
          v108 = *((_DWORD *)v110 + 1) & 0x7FFFFFF;
          if ( !v108 )
          {
LABEL_115:
            v14 = 1;
            goto LABEL_25;
          }
          v34 = v115;
          v121 = 0;
          v116 = v3;
          v105 = v34 >> 3;
          v35 = (char *)v3 + (v34 & 0xFFFFFFFFFFFFFFE0LL);
          v109 = v34 >> 5;
          v36 = v110;
          v37 = (__int64 *)v35;
          v106 = ((char *)v4 - v35) >> 3;
          while ( 1 )
          {
            if ( (v36[7] & 0x40) != 0 )
              v38 = (unsigned __int8 *)*((_QWORD *)v36 - 1);
            else
              v38 = &v36[-32 * (*((_DWORD *)v36 + 1) & 0x7FFFFFF)];
            v39 = *(_QWORD *)&v38[32 * v121];
            if ( *(_BYTE *)(*(_QWORD *)(v39 + 8) + 8LL) == 11 )
              goto LABEL_41;
            if ( v109 > 0 )
            {
              v40 = v116;
              while ( 1 )
              {
                v48 = *v40;
                v41 = (*(_BYTE *)(*v40 + 7) & 0x40) != 0
                    ? *(_QWORD *)(v48 - 8)
                    : v48 - 32LL * (*(_DWORD *)(v48 + 4) & 0x7FFFFFF);
                if ( v39 != *(_QWORD *)(v41 + 32 * v121) )
                  goto LABEL_95;
                v42 = v40[1];
                if ( (*(_BYTE *)(v42 + 7) & 0x40) != 0 )
                  v43 = *(_QWORD *)(v42 - 8);
                else
                  v43 = v42 - 32LL * (*(_DWORD *)(v42 + 4) & 0x7FFFFFF);
                if ( v39 != *(_QWORD *)(v43 + 32 * v121) )
                {
                  ++v40;
                  goto LABEL_95;
                }
                v44 = v40[2];
                if ( (*(_BYTE *)(v44 + 7) & 0x40) != 0 )
                  v45 = *(_QWORD *)(v44 - 8);
                else
                  v45 = v44 - 32LL * (*(_DWORD *)(v44 + 4) & 0x7FFFFFF);
                if ( v39 != *(_QWORD *)(v45 + 32 * v121) )
                {
                  v40 += 2;
                  goto LABEL_95;
                }
                v46 = v40[3];
                if ( (*(_BYTE *)(v46 + 7) & 0x40) != 0 )
                  v47 = *(_QWORD *)(v46 - 8);
                else
                  v47 = v46 - 32LL * (*(_DWORD *)(v46 + 4) & 0x7FFFFFF);
                if ( v39 != *(_QWORD *)(v47 + 32 * v121) )
                {
                  v40 += 3;
                  goto LABEL_95;
                }
                v40 += 4;
                if ( v37 == v40 )
                {
                  v67 = v106;
                  goto LABEL_125;
                }
              }
            }
            v67 = v105;
            v40 = v116;
LABEL_125:
            if ( v67 == 2 )
              goto LABEL_149;
            if ( v67 != 3 )
            {
              if ( v67 != 1 )
                goto LABEL_114;
              goto LABEL_128;
            }
            v72 = *v40;
            if ( (*(_BYTE *)(*v40 + 7) & 0x40) != 0 )
            {
              v73 = *(_QWORD *)(v72 - 8);
            }
            else
            {
              v17 = 32LL * (*(_DWORD *)(v72 + 4) & 0x7FFFFFF);
              v73 = v72 - v17;
            }
            if ( v39 == *(_QWORD *)(v73 + 32 * v121) )
              break;
LABEL_95:
            if ( v4 != v40 )
            {
              if ( *v36 != 85 )
                goto LABEL_97;
              v70 = *((_QWORD *)v36 - 4);
              if ( v70 )
              {
                if ( *(_BYTE *)v70
                  || *(_QWORD *)(v70 + 24) != *((_QWORD *)v36 + 10)
                  || (*(_BYTE *)(v70 + 33) & 0x20) == 0
                  || (unsigned int)(*(_DWORD *)(v70 + 36) - 210) > 1
                  || (_DWORD)v121 != 1 )
                {
                  if ( *(_BYTE *)v39 > 0x15u )
                    goto LABEL_101;
                  goto LABEL_134;
                }
                v79 = v116;
                if ( v109 > 0 )
                {
                  v111 = v4;
                  v80 = v116;
                  do
                  {
                    v88 = *v80;
                    if ( (*(_BYTE *)(*v80 + 7) & 0x40) != 0 )
                      v81 = *(_QWORD *)(v88 - 8);
                    else
                      v81 = v88 - 32LL * (*(_DWORD *)(v88 + 4) & 0x7FFFFFF);
                    if ( *sub_BD3990(*(unsigned __int8 **)(v81 + 32), v17) == 60 )
                    {
                      v90 = v80;
                      v4 = v111;
                      goto LABEL_205;
                    }
                    v82 = v80[1];
                    if ( (*(_BYTE *)(v82 + 7) & 0x40) != 0 )
                      v83 = *(_QWORD *)(v82 - 8);
                    else
                      v83 = v82 - 32LL * (*(_DWORD *)(v82 + 4) & 0x7FFFFFF);
                    if ( *sub_BD3990(*(unsigned __int8 **)(v83 + 32), v17) == 60 )
                    {
                      v89 = v80;
                      v4 = v111;
                      v90 = v89 + 1;
                      goto LABEL_205;
                    }
                    v84 = v80[2];
                    if ( (*(_BYTE *)(v84 + 7) & 0x40) != 0 )
                      v85 = *(_QWORD *)(v84 - 8);
                    else
                      v85 = v84 - 32LL * (*(_DWORD *)(v84 + 4) & 0x7FFFFFF);
                    if ( *sub_BD3990(*(unsigned __int8 **)(v85 + 32), v17) == 60 )
                    {
                      v91 = v80;
                      v4 = v111;
                      v90 = v91 + 2;
                      goto LABEL_205;
                    }
                    v86 = v80[3];
                    if ( (*(_BYTE *)(v86 + 7) & 0x40) != 0 )
                      v87 = *(_QWORD *)(v86 - 8);
                    else
                      v87 = v86 - 32LL * (*(_DWORD *)(v86 + 4) & 0x7FFFFFF);
                    if ( *sub_BD3990(*(unsigned __int8 **)(v87 + 32), v17) == 60 )
                    {
                      v92 = v80;
                      v4 = v111;
                      v90 = v92 + 3;
                      goto LABEL_205;
                    }
                    v80 += 4;
                  }
                  while ( v37 != v80 );
                  v79 = v80;
                  v4 = v111;
                }
                v94 = (char *)v4 - (char *)v79;
                if ( (char *)v4 - (char *)v79 == 16 )
                  goto LABEL_250;
                if ( v94 != 24 )
                {
                  if ( v94 == 8 )
                    goto LABEL_224;
                  goto LABEL_97;
                }
                v99 = *v79;
                if ( (*(_BYTE *)(*v79 + 7) & 0x40) != 0 )
                {
                  v100 = *(_QWORD *)(v99 - 8);
                }
                else
                {
                  v17 = 32LL * (*(_DWORD *)(v99 + 4) & 0x7FFFFFF);
                  v100 = v99 - v17;
                }
                v113 = v79;
                v101 = sub_BD3990(*(unsigned __int8 **)(v100 + 32), v17);
                v90 = v113;
                if ( *v101 == 60 )
                  goto LABEL_205;
                v79 = v113 + 1;
LABEL_250:
                v102 = *v79;
                if ( (*(_BYTE *)(*v79 + 7) & 0x40) != 0 )
                {
                  v103 = *(_QWORD *)(v102 - 8);
                }
                else
                {
                  v17 = 32LL * (*(_DWORD *)(v102 + 4) & 0x7FFFFFF);
                  v103 = v102 - v17;
                }
                v114 = v79;
                v104 = sub_BD3990(*(unsigned __int8 **)(v103 + 32), v17);
                v90 = v114;
                if ( *v104 == 60 )
                  goto LABEL_205;
                v79 = v114 + 1;
LABEL_224:
                v95 = *v79;
                if ( (*(_BYTE *)(*v79 + 7) & 0x40) != 0 )
                {
                  v96 = *(_QWORD *)(v95 - 8);
                }
                else
                {
                  v17 = 32LL * (*(_DWORD *)(v95 + 4) & 0x7FFFFFF);
                  v96 = v95 - v17;
                }
                v112 = v79;
                v97 = sub_BD3990(*(unsigned __int8 **)(v96 + 32), v17);
                v90 = v112;
                if ( *v97 == 60 )
                {
LABEL_205:
                  if ( v4 != v90 )
                  {
                    v14 = 0;
                    goto LABEL_25;
                  }
                }
LABEL_97:
                if ( *(_BYTE *)v39 <= 0x15u )
                {
                  v49 = *v36;
                  if ( (unsigned int)(v49 - 48) <= 1 || (unsigned __int8)(v49 - 51) <= 1u )
                  {
                    if ( v121 == 1 )
                      goto LABEL_41;
                    goto LABEL_101;
                  }
                  if ( (_BYTE)v49 != 85 )
                    goto LABEL_101;
                  v70 = *((_QWORD *)v36 - 4);
                  if ( !v70 )
                    goto LABEL_101;
LABEL_134:
                  if ( !*(_BYTE *)v70
                    && *(_QWORD *)(v70 + 24) == *((_QWORD *)v36 + 10)
                    && (*(_BYTE *)(v70 + 33) & 0x20) != 0 )
                  {
                    v14 = 0;
                    goto LABEL_25;
                  }
                }
              }
LABEL_101:
              if ( !sub_F58730(v36, v121) )
                goto LABEL_41;
              if ( (v36[7] & 0x40) != 0 )
                v50 = (unsigned __int8 *)*((_QWORD *)v36 - 1);
              else
                v50 = &v36[-32 * (*((_DWORD *)v36 + 1) & 0x7FFFFFF)];
              v51 = &v50[32 * v121];
              v126 = v51;
              v52 = *(_DWORD *)(a3 + 24);
              if ( v52 )
              {
                v53 = *(_QWORD *)(a3 + 8);
                v17 = (unsigned int)(v52 - 1);
                v54 = 1;
                v55 = 0;
                LODWORD(v56) = v17 & (((unsigned int)v51 >> 9) ^ ((unsigned int)v51 >> 4));
                v57 = (unsigned __int8 **)(v53 + 56LL * (unsigned int)v56);
                v58 = *v57;
                if ( v51 == *v57 )
                  goto LABEL_106;
                while ( v58 != (unsigned __int8 *)-4096LL )
                {
                  if ( !v55 && v58 == (unsigned __int8 *)-8192LL )
                    v55 = v57;
                  v56 = (unsigned int)v17 & ((_DWORD)v56 + v54);
                  v57 = (unsigned __int8 **)(v53 + 56 * v56);
                  v58 = *v57;
                  if ( v51 == *v57 )
                    goto LABEL_106;
                  ++v54;
                }
                if ( !v55 )
                  v55 = v57;
              }
              else
              {
                v55 = 0;
              }
              v17 = (__int64)&v126;
              v57 = (unsigned __int8 **)sub_FA4390(a3, &v126, v55);
              v71 = v126;
              v57[2] = (unsigned __int8 *)0x400000000LL;
              *v57 = v71;
              v57[1] = (unsigned __int8 *)(v57 + 3);
LABEL_106:
              v59 = (__int64)(v57 + 1);
              if ( v116 != v4 )
              {
                v60 = *((unsigned int *)v57 + 4);
                v17 = (__int64)(v57 + 3);
                v61 = v116;
                do
                {
                  v64 = *v61;
                  if ( (*(_BYTE *)(*v61 + 7) & 0x40) != 0 )
                    v62 = *(_QWORD *)(v64 - 8);
                  else
                    v62 = v64 - 32LL * (*(_DWORD *)(v64 + 4) & 0x7FFFFFF);
                  v63 = *(_QWORD *)(v62 + 32 * v121);
                  if ( v60 + 1 > (unsigned __int64)*(unsigned int *)(v59 + 12) )
                  {
                    v107 = *(_QWORD *)(v62 + 32 * v121);
                    sub_C8D5F0(v59, (const void *)v17, v60 + 1, 8u, v63, v60 + 1);
                    v60 = *(unsigned int *)(v59 + 8);
                    v63 = v107;
                  }
                  ++v61;
                  *(_QWORD *)(*(_QWORD *)v59 + 8 * v60) = v63;
                  v60 = (unsigned int)(*(_DWORD *)(v59 + 8) + 1);
                  *(_DWORD *)(v59 + 8) = v60;
                }
                while ( v4 != v61 );
              }
            }
LABEL_114:
            if ( v108 == (_DWORD)++v121 )
              goto LABEL_115;
          }
          ++v40;
LABEL_149:
          v74 = *v40;
          if ( (*(_BYTE *)(*v40 + 7) & 0x40) != 0 )
          {
            v75 = *(_QWORD *)(v74 - 8);
          }
          else
          {
            v17 = 32LL * (*(_DWORD *)(v74 + 4) & 0x7FFFFFF);
            v75 = v74 - v17;
          }
          if ( v39 == *(_QWORD *)(v75 + 32 * v121) )
          {
            ++v40;
LABEL_128:
            v68 = *v40;
            if ( (*(_BYTE *)(*v40 + 7) & 0x40) != 0 )
            {
              v69 = *(_QWORD *)(v68 - 8);
            }
            else
            {
              v17 = 32LL * (*(_DWORD *)(v68 + 4) & 0x7FFFFFF);
              v69 = v68 - v17;
            }
            if ( v39 == *(_QWORD *)(v69 + 32 * v121) )
              goto LABEL_114;
            goto LABEL_95;
          }
          goto LABEL_95;
        }
LABEL_172:
        if ( v3 != v4 )
        {
          v77 = *(_QWORD *)(*v3 - 32);
          v78 = v3;
          do
          {
            while ( 1 )
            {
              if ( ++v78 == v4 )
                goto LABEL_68;
              if ( v77 )
                break;
              v77 = *(_QWORD *)(*v78 - 32);
            }
          }
          while ( v77 == *(_QWORD *)(*v78 - 32) );
          v14 = 0;
          goto LABEL_25;
        }
        goto LABEL_68;
      }
    }
    else if ( v115 != 8 || !sub_B491E0(*v3) )
    {
      v32 = v4;
      v33 = v3;
      goto LABEL_168;
    }
    v32 = v3;
    v33 = v3;
    goto LABEL_168;
  }
  v5 = a1;
  v6 = 0;
  v118 = 0;
  do
  {
    v7 = (unsigned __int8 *)*v5;
    v8 = *(_BYTE *)*v5;
    if ( v8 == 84 )
      return 0;
    v9 = v8 - 39;
    if ( v9 <= 0x38 && ((1LL << v9) & 0x100060000000001LL) != 0 )
      return 0;
    if ( v8 == 60 )
      return 0;
    if ( *(_BYTE *)(*((_QWORD *)v7 + 1) + 8LL) == 11 )
      return 0;
    if ( *((_QWORD *)v7 + 5) == sub_AA56F0(*((_QWORD *)v7 + 5)) )
      return 0;
    if ( (unsigned __int8)(*v7 - 34) <= 0x33u )
    {
      v10 = 0x8000000000041LL;
      if ( _bittest64(&v10, (unsigned int)*v7 - 34) )
      {
        if ( **((_BYTE **)v7 - 4) == 25
          || (unsigned __int8)sub_A73ED0((_QWORD *)v7 + 9, 32)
          || (unsigned __int8)sub_B49560((__int64)v7, 32)
          || (unsigned __int8)sub_A73ED0((_QWORD *)v7 + 9, 6)
          || (unsigned __int8)sub_B49560((__int64)v7, 6) )
        {
          return 0;
        }
      }
    }
    if ( v6 )
    {
      if ( (unsigned int)sub_BD3960((__int64)v7) != v118 )
        return 0;
    }
    else
    {
      v6 = 1;
      v118 = sub_BD3960((__int64)v7);
    }
    ++v5;
  }
  while ( v4 != v5 );
  v110 = (unsigned __int8 *)*a1;
  sub_B8E070((__int64)v122, *a1);
  v119 = v4;
  v11 = a1;
  while ( 1 )
  {
    v12 = *v11;
    if ( !(unsigned __int8)sub_B46250(*v11, (__int64)v110, 4) )
      goto LABEL_41;
    v13 = *(_BYTE *)v12;
    if ( *(_BYTE *)v12 == 62 )
    {
      if ( (*(_BYTE *)(v12 + 7) & 0x40) != 0 )
        v29 = *(_QWORD *)(v12 - 8);
      else
        v29 = v12 - 32LL * (*(_DWORD *)(v12 + 4) & 0x7FFFFFF);
      if ( (unsigned __int8)sub_BD6020(*(_QWORD *)(v29 + 32)) )
        goto LABEL_41;
      v13 = *(_BYTE *)v12;
    }
    if ( v13 == 61 )
    {
      v28 = (*(_BYTE *)(v12 + 7) & 0x40) != 0
          ? *(__int64 **)(v12 - 8)
          : (__int64 *)(v12 - 32LL * (*(_DWORD *)(v12 + 4) & 0x7FFFFFF));
      if ( (unsigned __int8)sub_BD6020(*v28) )
        break;
    }
    sub_B8E070((__int64)&v126, v12);
    if ( v128 != v124 )
    {
      sub_C7D6A0(v127, 32LL * v129, 8);
      v14 = 0;
      goto LABEL_25;
    }
    v16 = sub_F9F240((__int64)&v126, (__int64)v122);
    v17 = 32LL * v129;
    sub_C7D6A0(v127, v17, 8);
    if ( !v16 )
      goto LABEL_41;
    if ( v119 == ++v11 )
    {
      v4 = v119;
      v3 = a1;
      goto LABEL_31;
    }
  }
  v14 = 0;
LABEL_25:
  v120 = v14;
  sub_C7D6A0(v123, 32LL * v125, 8);
  return v120;
}
