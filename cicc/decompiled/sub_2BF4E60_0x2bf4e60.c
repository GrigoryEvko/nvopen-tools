// Function: sub_2BF4E60
// Address: 0x2bf4e60
//
__int64 __fastcall sub_2BF4E60(__int64 a1)
{
  __int64 v1; // r12
  __int64 v2; // rdx
  __int64 v3; // rcx
  __int64 v4; // r8
  __int64 v5; // r9
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // rax
  __int64 v11; // rsi
  __int64 v12; // rbx
  __int64 v13; // rcx
  int v14; // r11d
  __int64 *v15; // rdi
  __int64 v16; // r9
  __int64 v17; // rdx
  __int64 *v18; // rax
  __int64 v19; // r8
  __int64 *v20; // rax
  __int64 v21; // rcx
  __int64 v22; // rsi
  char v23; // al
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // r8
  __int64 v27; // r9
  __int64 v28; // rdx
  __int64 v29; // rcx
  __int64 v30; // r8
  __int64 v31; // r9
  __int64 v32; // rcx
  __int64 v33; // rsi
  __int64 v34; // rax
  __int64 v35; // rsi
  __int64 v36; // r8
  int v37; // r11d
  __int64 v38; // r9
  unsigned __int64 v39; // rcx
  __int64 v40; // rdx
  __int64 v41; // r10
  __int64 v42; // rbx
  __int64 *v43; // r12
  __int64 *v44; // r15
  int v45; // r11d
  unsigned int v46; // edx
  __int64 *v47; // rax
  __int64 v48; // r10
  __int64 v49; // r13
  __int64 v50; // rax
  unsigned __int64 v51; // rdx
  __int64 v52; // rax
  __int64 v53; // r13
  unsigned int v54; // eax
  int v55; // edx
  __int64 v56; // rdi
  char v57; // al
  __int64 v58; // r12
  int v60; // r10d
  unsigned int v61; // r14d
  __int64 v62; // rdi
  __int64 v63; // rcx
  unsigned int *v64; // r12
  unsigned int *v65; // r14
  __int64 v66; // rax
  __int64 v67; // r13
  __int64 *v68; // r12
  __int64 *v69; // r15
  int v70; // r11d
  unsigned int v71; // edx
  __int64 *v72; // rax
  __int64 v73; // r10
  __int64 v74; // r13
  __int64 v75; // rax
  unsigned __int64 v76; // rdx
  __int64 v77; // rax
  __int64 v78; // r13
  unsigned int v79; // eax
  int v80; // edx
  __int64 v81; // rdi
  int v82; // r10d
  unsigned int v83; // r14d
  __int64 v84; // rdi
  __int64 v85; // rcx
  __int64 *v86; // r12
  __int64 *v87; // r14
  __int64 v88; // rax
  __int64 v89; // r13
  int v90; // r10d
  int v91; // r10d
  __int64 v92; // [rsp+48h] [rbp-2D8h] BYREF
  unsigned int *v93; // [rsp+50h] [rbp-2D0h] BYREF
  __int64 v94; // [rsp+58h] [rbp-2C8h] BYREF
  __int64 v95; // [rsp+60h] [rbp-2C0h] BYREF
  __int64 v96; // [rsp+68h] [rbp-2B8h]
  __int64 v97; // [rsp+70h] [rbp-2B0h]
  unsigned int v98; // [rsp+78h] [rbp-2A8h]
  unsigned int *v99; // [rsp+80h] [rbp-2A0h] BYREF
  __int64 v100; // [rsp+88h] [rbp-298h]
  _BYTE v101[48]; // [rsp+90h] [rbp-290h] BYREF
  __int64 *v102; // [rsp+C0h] [rbp-260h] BYREF
  __int64 v103; // [rsp+C8h] [rbp-258h]
  _BYTE v104[48]; // [rsp+D0h] [rbp-250h] BYREF
  _QWORD v105[12]; // [rsp+100h] [rbp-220h] BYREF
  __int64 v106; // [rsp+160h] [rbp-1C0h]
  __int64 v107; // [rsp+168h] [rbp-1B8h]
  _QWORD v108[12]; // [rsp+180h] [rbp-1A0h] BYREF
  __int64 v109; // [rsp+1E0h] [rbp-140h]
  __int64 v110; // [rsp+1E8h] [rbp-138h]
  _QWORD v111[15]; // [rsp+200h] [rbp-120h] BYREF
  _BYTE v112[168]; // [rsp+278h] [rbp-A8h] BYREF

  v92 = a1;
  v1 = *(_QWORD *)(a1 + 48);
  v108[0] = a1;
  v95 = 0;
  v96 = 0;
  v97 = 0;
  v98 = 0;
  v93 = 0;
  sub_2BF3840(v111, v108);
  sub_2ABD910(v105, (__int64)v111, v2, v3, v4, v5);
  sub_2ABD910(v108, (__int64)v112, v6, v7, v8, v9);
  while ( 1 )
  {
    v21 = v106;
    v22 = v109;
    if ( v107 - v106 != v110 - v109 )
      goto LABEL_2;
    if ( v107 == v106 )
      break;
    while ( *(_QWORD *)v21 == *(_QWORD *)v22 )
    {
      v23 = *(_BYTE *)(v21 + 16);
      if ( v23 != *(_BYTE *)(v22 + 16) || v23 && *(_QWORD *)(v21 + 8) != *(_QWORD *)(v22 + 8) )
        break;
      v21 += 24;
      v22 += 24;
      if ( v107 == v21 )
        goto LABEL_16;
    }
LABEL_2:
    v99 = *(unsigned int **)(v107 - 24);
    v10 = (*(__int64 (**)(void))(*(_QWORD *)v99 + 32LL))();
    v11 = v98;
    v12 = v10;
    if ( !v98 )
    {
      ++v95;
      v102 = 0;
LABEL_49:
      LODWORD(v11) = 2 * v98;
LABEL_50:
      sub_2BF4B40((__int64)&v95, v11);
      v11 = (__int64)&v99;
      sub_2BF2320((__int64)&v95, (__int64 *)&v99, &v102);
      v13 = (__int64)v99;
      v15 = v102;
      v17 = (unsigned int)(v97 + 1);
      goto LABEL_45;
    }
    v13 = (__int64)v99;
    v14 = 1;
    v15 = 0;
    v16 = v96;
    v17 = (v98 - 1) & (((unsigned int)v99 >> 9) ^ ((unsigned int)v99 >> 4));
    v18 = (__int64 *)(v96 + 16 * v17);
    v19 = *v18;
    if ( v99 == (unsigned int *)*v18 )
    {
LABEL_4:
      v20 = v18 + 1;
      goto LABEL_5;
    }
    while ( v19 != -4096 )
    {
      if ( v19 == -8192 && !v15 )
        v15 = v18;
      v17 = (v98 - 1) & (v14 + (_DWORD)v17);
      v18 = (__int64 *)(v96 + 16LL * (unsigned int)v17);
      v19 = *v18;
      if ( v99 == (unsigned int *)*v18 )
        goto LABEL_4;
      ++v14;
    }
    if ( !v15 )
      v15 = v18;
    ++v95;
    v17 = (unsigned int)(v97 + 1);
    v102 = v15;
    if ( 4 * (int)v17 >= 3 * v98 )
      goto LABEL_49;
    v19 = v98 >> 3;
    if ( v98 - HIDWORD(v97) - (unsigned int)v17 <= (unsigned int)v19 )
      goto LABEL_50;
LABEL_45:
    LODWORD(v97) = v17;
    if ( *v15 != -4096 )
      --HIDWORD(v97);
    *v15 = v13;
    v20 = v15 + 1;
    v15[1] = 0;
LABEL_5:
    *v20 = v12;
    if ( v1 )
    {
      v17 = v99[22];
      if ( !(_DWORD)v17 )
        v93 = v99;
    }
    sub_2ADA290((__int64)v105, v11, v17, v13, v19, v16);
  }
LABEL_16:
  sub_2AB1B10((__int64)v108);
  sub_2AB1B10((__int64)v105);
  sub_2AB1B10((__int64)v112);
  sub_2AB1B10((__int64)v111);
  v108[0] = v92;
  sub_2BF3840(v111, v108);
  sub_2ABD910(v105, (__int64)v111, v24, v25, v26, v27);
  sub_2ABD910(v108, (__int64)v112, v28, v29, v30, v31);
LABEL_17:
  v32 = v106;
  v33 = v109;
  if ( v107 - v106 != v110 - v109 )
  {
LABEL_18:
    v34 = *(_QWORD *)(v107 - 24);
    v35 = v98;
    v94 = v34;
    if ( v98 )
    {
      v36 = v98 - 1;
      v37 = 1;
      v38 = 0;
      v39 = (unsigned int)v36 & (((unsigned int)v34 >> 9) ^ ((unsigned int)v34 >> 4));
      v40 = v96 + 16 * v39;
      v41 = *(_QWORD *)v40;
      if ( v34 == *(_QWORD *)v40 )
      {
LABEL_20:
        v42 = *(_QWORD *)(v40 + 8);
        goto LABEL_21;
      }
      while ( v41 != -4096 )
      {
        if ( !v38 && v41 == -8192 )
          v38 = v40;
        v39 = (unsigned int)v36 & (v37 + (_DWORD)v39);
        v40 = v96 + 16LL * (unsigned int)v39;
        v41 = *(_QWORD *)v40;
        if ( v34 == *(_QWORD *)v40 )
          goto LABEL_20;
        ++v37;
      }
      if ( !v38 )
        v38 = v40;
      ++v95;
      v39 = (unsigned int)(v97 + 1);
      v102 = (__int64 *)v38;
      if ( 4 * (int)v39 < 3 * v98 )
      {
        v40 = v98 - HIDWORD(v97) - (unsigned int)v39;
        if ( (unsigned int)v40 > v98 >> 3 )
        {
LABEL_86:
          LODWORD(v97) = v39;
          if ( *(_QWORD *)v38 != -4096 )
            --HIDWORD(v97);
          *(_QWORD *)v38 = v34;
          v42 = 0;
          v34 = v94;
          *(_QWORD *)(v38 + 8) = 0;
LABEL_21:
          v99 = (unsigned int *)v101;
          v100 = 0x600000000LL;
          v43 = *(__int64 **)(v34 + 56);
          v44 = &v43[*(unsigned int *)(v34 + 64)];
          if ( v44 == v43 )
          {
LABEL_94:
            v102 = (__int64 *)v104;
            v103 = 0x600000000LL;
            v68 = *(__int64 **)(v94 + 80);
            v69 = &v68[*(unsigned int *)(v94 + 88)];
            if ( v68 == v69 )
              goto LABEL_132;
            while ( 1 )
            {
              v35 = v98;
              v78 = *v68;
              if ( !v98 )
                break;
              v70 = 1;
              v36 = 0;
              v71 = (v98 - 1) & (((unsigned int)v78 >> 9) ^ ((unsigned int)v78 >> 4));
              v72 = (__int64 *)(v96 + 16LL * v71);
              v73 = *v72;
              if ( v78 != *v72 )
              {
                while ( v73 != -4096 )
                {
                  if ( v73 == -8192 && !v36 )
                    v36 = (__int64)v72;
                  v38 = (unsigned int)(v70 + 1);
                  v71 = (v98 - 1) & (v70 + v71);
                  v72 = (__int64 *)(v96 + 16LL * v71);
                  v73 = *v72;
                  if ( v78 == *v72 )
                    goto LABEL_97;
                  ++v70;
                }
                if ( !v36 )
                  v36 = (__int64)v72;
                ++v95;
                v80 = v97 + 1;
                if ( 4 * ((int)v97 + 1) < 3 * v98 )
                {
                  if ( v98 - HIDWORD(v97) - v80 <= v98 >> 3 )
                  {
                    sub_2BF4B40((__int64)&v95, v98);
                    if ( !v98 )
                    {
LABEL_171:
                      LODWORD(v97) = v97 + 1;
                      BUG();
                    }
                    v35 = v96;
                    v82 = 1;
                    v83 = (v98 - 1) & (((unsigned int)v78 >> 9) ^ ((unsigned int)v78 >> 4));
                    v80 = v97 + 1;
                    v84 = 0;
                    v36 = v96 + 16LL * v83;
                    v85 = *(_QWORD *)v36;
                    if ( v78 != *(_QWORD *)v36 )
                    {
                      while ( v85 != -4096 )
                      {
                        if ( v85 != -8192 || v84 )
                          v36 = v84;
                        v83 = (v98 - 1) & (v82 + v83);
                        v38 = v96 + 16LL * v83;
                        v85 = *(_QWORD *)v38;
                        if ( v78 == *(_QWORD *)v38 )
                        {
                          v36 = v96 + 16LL * v83;
                          goto LABEL_105;
                        }
                        ++v82;
                        v84 = v36;
                        v36 = v96 + 16LL * v83;
                      }
                      if ( v84 )
                        v36 = v84;
                    }
                  }
                  goto LABEL_105;
                }
LABEL_103:
                sub_2BF4B40((__int64)&v95, 2 * v98);
                if ( !v98 )
                  goto LABEL_171;
                v35 = v96;
                v79 = (v98 - 1) & (((unsigned int)v78 >> 9) ^ ((unsigned int)v78 >> 4));
                v80 = v97 + 1;
                v36 = v96 + 16LL * v79;
                v81 = *(_QWORD *)v36;
                if ( v78 != *(_QWORD *)v36 )
                {
                  v91 = 1;
                  v38 = 0;
                  while ( v81 != -4096 )
                  {
                    if ( v81 == -8192 && !v38 )
                      v38 = v36;
                    v79 = (v98 - 1) & (v91 + v79);
                    v36 = v96 + 16LL * v79;
                    v81 = *(_QWORD *)v36;
                    if ( v78 == *(_QWORD *)v36 )
                      goto LABEL_105;
                    ++v91;
                  }
                  if ( v38 )
                    v36 = v38;
                }
LABEL_105:
                LODWORD(v97) = v80;
                if ( *(_QWORD *)v36 != -4096 )
                  --HIDWORD(v97);
                *(_QWORD *)v36 = v78;
                v74 = 0;
                *(_QWORD *)(v36 + 8) = 0;
                goto LABEL_98;
              }
LABEL_97:
              v74 = v72[1];
LABEL_98:
              v75 = (unsigned int)v103;
              v39 = HIDWORD(v103);
              v76 = (unsigned int)v103 + 1LL;
              if ( v76 > HIDWORD(v103) )
              {
                v35 = (__int64)v104;
                sub_C8D5F0((__int64)&v102, v104, v76, 8u, v36, v38);
                v75 = (unsigned int)v103;
              }
              v40 = (__int64)v102;
              ++v68;
              v102[v75] = v74;
              v77 = (unsigned int)(v103 + 1);
              LODWORD(v103) = v103 + 1;
              if ( v69 == v68 )
              {
                v86 = v102;
                v87 = &v102[v77];
                if ( v102 != v87 )
                {
                  v88 = *(unsigned int *)(v42 + 88);
                  v35 = v42 + 96;
                  do
                  {
                    v39 = *(unsigned int *)(v42 + 92);
                    v89 = *v86;
                    if ( v88 + 1 > v39 )
                    {
                      sub_C8D5F0(v42 + 80, (const void *)v35, v88 + 1, 8u, v36, v38);
                      v88 = *(unsigned int *)(v42 + 88);
                    }
                    v40 = *(_QWORD *)(v42 + 80);
                    ++v86;
                    *(_QWORD *)(v40 + 8 * v88) = v89;
                    v88 = (unsigned int)(*(_DWORD *)(v42 + 88) + 1);
                    *(_DWORD *)(v42 + 88) = v88;
                  }
                  while ( v87 != v86 );
                  v87 = v102;
                }
                if ( v87 != (__int64 *)v104 )
                  _libc_free((unsigned __int64)v87);
LABEL_132:
                if ( v99 != (unsigned int *)v101 )
                  _libc_free((unsigned __int64)v99);
                sub_2ADA290((__int64)v105, v35, v40, v39, v36, v38);
                goto LABEL_17;
              }
            }
            ++v95;
            goto LABEL_103;
          }
          while ( 1 )
          {
            v35 = v98;
            v53 = *v43;
            if ( !v98 )
              break;
            v45 = 1;
            v36 = 0;
            v46 = (v98 - 1) & (((unsigned int)v53 >> 9) ^ ((unsigned int)v53 >> 4));
            v47 = (__int64 *)(v96 + 16LL * v46);
            v48 = *v47;
            if ( v53 != *v47 )
            {
              while ( v48 != -4096 )
              {
                if ( !v36 && v48 == -8192 )
                  v36 = (__int64)v47;
                v38 = (unsigned int)(v45 + 1);
                v46 = (v98 - 1) & (v45 + v46);
                v47 = (__int64 *)(v96 + 16LL * v46);
                v48 = *v47;
                if ( v53 == *v47 )
                  goto LABEL_24;
                ++v45;
              }
              if ( !v36 )
                v36 = (__int64)v47;
              ++v95;
              v55 = v97 + 1;
              if ( 4 * ((int)v97 + 1) < 3 * v98 )
              {
                if ( v98 - HIDWORD(v97) - v55 <= v98 >> 3 )
                {
                  sub_2BF4B40((__int64)&v95, v98);
                  if ( !v98 )
                  {
LABEL_170:
                    LODWORD(v97) = v97 + 1;
                    BUG();
                  }
                  v35 = v96;
                  v60 = 1;
                  v61 = (v98 - 1) & (((unsigned int)v53 >> 9) ^ ((unsigned int)v53 >> 4));
                  v55 = v97 + 1;
                  v62 = 0;
                  v36 = v96 + 16LL * v61;
                  v63 = *(_QWORD *)v36;
                  if ( v53 != *(_QWORD *)v36 )
                  {
                    while ( v63 != -4096 )
                    {
                      if ( v63 != -8192 || v62 )
                        v36 = v62;
                      v61 = (v98 - 1) & (v60 + v61);
                      v38 = v96 + 16LL * v61;
                      v63 = *(_QWORD *)v38;
                      if ( v53 == *(_QWORD *)v38 )
                      {
                        v36 = v96 + 16LL * v61;
                        goto LABEL_32;
                      }
                      ++v60;
                      v62 = v36;
                      v36 = v96 + 16LL * v61;
                    }
                    if ( v62 )
                      v36 = v62;
                  }
                }
                goto LABEL_32;
              }
LABEL_30:
              sub_2BF4B40((__int64)&v95, 2 * v98);
              if ( !v98 )
                goto LABEL_170;
              v35 = v96;
              v54 = (v98 - 1) & (((unsigned int)v53 >> 9) ^ ((unsigned int)v53 >> 4));
              v55 = v97 + 1;
              v36 = v96 + 16LL * v54;
              v56 = *(_QWORD *)v36;
              if ( v53 != *(_QWORD *)v36 )
              {
                v90 = 1;
                v38 = 0;
                while ( v56 != -4096 )
                {
                  if ( v56 == -8192 && !v38 )
                    v38 = v36;
                  v54 = (v98 - 1) & (v90 + v54);
                  v36 = v96 + 16LL * v54;
                  v56 = *(_QWORD *)v36;
                  if ( v53 == *(_QWORD *)v36 )
                    goto LABEL_32;
                  ++v90;
                }
                if ( v38 )
                  v36 = v38;
              }
LABEL_32:
              LODWORD(v97) = v55;
              if ( *(_QWORD *)v36 != -4096 )
                --HIDWORD(v97);
              *(_QWORD *)v36 = v53;
              v49 = 0;
              *(_QWORD *)(v36 + 8) = 0;
              goto LABEL_25;
            }
LABEL_24:
            v49 = v47[1];
LABEL_25:
            v50 = (unsigned int)v100;
            v39 = HIDWORD(v100);
            v51 = (unsigned int)v100 + 1LL;
            if ( v51 > HIDWORD(v100) )
            {
              v35 = (__int64)v101;
              sub_C8D5F0((__int64)&v99, v101, v51, 8u, v36, v38);
              v50 = (unsigned int)v100;
            }
            v40 = (__int64)v99;
            ++v43;
            *(_QWORD *)&v99[2 * v50] = v49;
            v52 = (unsigned int)(v100 + 1);
            LODWORD(v100) = v100 + 1;
            if ( v44 == v43 )
            {
              v64 = v99;
              v65 = &v99[2 * v52];
              if ( v65 != v99 )
              {
                v66 = *(unsigned int *)(v42 + 64);
                v35 = v42 + 72;
                do
                {
                  v39 = *(unsigned int *)(v42 + 68);
                  v67 = *(_QWORD *)v64;
                  if ( v66 + 1 > v39 )
                  {
                    sub_C8D5F0(v42 + 56, (const void *)v35, v66 + 1, 8u, v36, v38);
                    v66 = *(unsigned int *)(v42 + 64);
                  }
                  v40 = *(_QWORD *)(v42 + 56);
                  v64 += 2;
                  *(_QWORD *)(v40 + 8 * v66) = v67;
                  v66 = (unsigned int)(*(_DWORD *)(v42 + 64) + 1);
                  *(_DWORD *)(v42 + 64) = v66;
                }
                while ( v65 != v64 );
              }
              goto LABEL_94;
            }
          }
          ++v95;
          goto LABEL_30;
        }
LABEL_137:
        sub_2BF4B40((__int64)&v95, v35);
        v35 = (__int64)&v94;
        sub_2BF2320((__int64)&v95, &v94, &v102);
        v34 = v94;
        v38 = (__int64)v102;
        v39 = (unsigned int)(v97 + 1);
        goto LABEL_86;
      }
    }
    else
    {
      ++v95;
      v102 = 0;
    }
    LODWORD(v35) = 2 * v98;
    goto LABEL_137;
  }
  if ( v107 != v106 )
  {
    while ( *(_QWORD *)v32 == *(_QWORD *)v33 )
    {
      v57 = *(_BYTE *)(v32 + 16);
      if ( v57 != *(_BYTE *)(v33 + 16) || v57 && *(_QWORD *)(v32 + 8) != *(_QWORD *)(v33 + 8) )
        break;
      v32 += 24;
      v33 += 24;
      if ( v107 == v32 )
        goto LABEL_57;
    }
    goto LABEL_18;
  }
LABEL_57:
  sub_2AB1B10((__int64)v108);
  sub_2AB1B10((__int64)v105);
  sub_2AB1B10((__int64)v112);
  sub_2AB1B10((__int64)v111);
  if ( v93 )
    sub_2BF4D20((__int64)&v95, (__int64 *)&v93);
  v58 = *sub_2BF4D20((__int64)&v95, &v92);
  sub_C7D6A0(v96, 16LL * v98, 8);
  return v58;
}
