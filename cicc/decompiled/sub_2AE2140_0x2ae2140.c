// Function: sub_2AE2140
// Address: 0x2ae2140
//
__int64 __fastcall sub_2AE2140(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rbx
  __int64 v5; // rcx
  int v6; // eax
  int v7; // esi
  unsigned int v8; // edx
  __int64 *v9; // rax
  __int64 v10; // rdi
  __int64 v11; // r14
  __int64 v12; // rax
  int v13; // r12d
  __int64 *v14; // rbx
  __int64 *v15; // r12
  unsigned int v16; // esi
  __int64 v17; // rcx
  __int64 v18; // r9
  int v19; // r11d
  __int64 *v20; // r10
  unsigned int v21; // edx
  __int64 *v22; // rdi
  __int64 v23; // r8
  __int64 v24; // rax
  int v25; // edx
  __int64 v26; // rdx
  __int64 v27; // r15
  unsigned int v28; // eax
  __int64 *v29; // rcx
  __int64 v30; // rdi
  __int64 v31; // rsi
  _QWORD *v32; // rax
  _QWORD *v33; // rcx
  __int64 i; // r13
  __int64 v35; // r12
  __int64 *v36; // rbx
  __int64 *v37; // rdi
  __int64 *v38; // r15
  __int64 v39; // r8
  int v40; // r10d
  __int64 **v41; // r9
  unsigned int v42; // ecx
  __int64 **v43; // rdx
  __int64 *v44; // rax
  __int64 *v45; // r12
  unsigned int v46; // eax
  __int64 *v47; // rsi
  int v48; // edx
  __int64 v49; // rax
  unsigned __int64 v50; // rdx
  unsigned int v52; // r13d
  __int64 **v53; // rdi
  __int64 *v54; // rcx
  int v55; // ecx
  int v56; // r8d
  __int16 v57; // dx
  __int64 v58; // rsi
  char v59; // al
  char v60; // dl
  __int64 v61; // rdi
  __int64 v62; // rax
  __int64 *v63; // r13
  __int64 *v64; // r12
  __int64 v65; // r9
  __int64 v66; // r8
  int v67; // r11d
  _QWORD *v68; // r10
  unsigned int v69; // edx
  _QWORD *v70; // rdi
  __int64 v71; // rcx
  __int64 v72; // rax
  unsigned int v73; // edx
  __int64 v74; // rsi
  int v75; // ecx
  __int64 v76; // rax
  __int64 v77; // r15
  unsigned __int64 v78; // rdx
  __int64 *v79; // r13
  __int64 v80; // rax
  __int64 *v81; // r12
  __int64 v82; // r9
  __int64 v83; // r8
  int v84; // r11d
  __int64 *v85; // r10
  unsigned int v86; // edx
  __int64 *v87; // rcx
  __int64 v88; // rdi
  __int64 v89; // rax
  int v90; // esi
  int v91; // ecx
  __int64 v92; // rax
  __int64 v93; // r15
  unsigned __int64 v94; // rdx
  __int64 v95; // rax
  __int64 v96; // r13
  unsigned __int64 v97; // rdx
  int v98; // eax
  int v99; // r10d
  __int64 v100; // rdi
  int v101; // r8d
  __int64 v102; // [rsp+10h] [rbp-E0h]
  __int64 v103; // [rsp+20h] [rbp-D0h]
  char v104; // [rsp+2Fh] [rbp-C1h]
  __int64 v105; // [rsp+30h] [rbp-C0h] BYREF
  __int64 *v106; // [rsp+38h] [rbp-B8h] BYREF
  __int64 v107; // [rsp+40h] [rbp-B0h] BYREF
  __int64 v108; // [rsp+48h] [rbp-A8h]
  __int64 v109; // [rsp+50h] [rbp-A0h]
  __int64 v110; // [rsp+58h] [rbp-98h]
  __int64 **v111; // [rsp+60h] [rbp-90h] BYREF
  __int64 v112; // [rsp+68h] [rbp-88h]
  __int64 *v113; // [rsp+70h] [rbp-80h] BYREF
  int v114; // [rsp+78h] [rbp-78h]
  int v115; // [rsp+7Ch] [rbp-74h]
  _BYTE v116[112]; // [rsp+80h] [rbp-70h] BYREF

  v3 = *(_QWORD *)(a1 + 24);
  v4 = *(_QWORD *)(a2 + 40);
  v5 = *(_QWORD *)(v3 + 8);
  v6 = *(_DWORD *)(v3 + 24);
  v102 = v4;
  if ( !v6 )
  {
LABEL_150:
    v11 = 0;
    goto LABEL_4;
  }
  v7 = v6 - 1;
  v8 = (v6 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
  v9 = (__int64 *)(v5 + 16LL * v8);
  v10 = *v9;
  if ( v4 != *v9 )
  {
    v98 = 1;
    while ( v10 != -4096 )
    {
      v101 = v98 + 1;
      v8 = v7 & (v98 + v8);
      v9 = (__int64 *)(v5 + 16LL * v8);
      v10 = *v9;
      if ( v4 == *v9 )
        goto LABEL_3;
      v98 = v101;
    }
    goto LABEL_150;
  }
LABEL_3:
  v11 = v9[1];
LABEL_4:
  v12 = sub_986520(a2);
  v13 = *(_DWORD *)(a2 + 4);
  v107 = 0;
  v108 = 0;
  v14 = (__int64 *)v12;
  v109 = 0;
  v110 = 0;
  v112 = 0;
  v15 = (__int64 *)(v12 + 32LL * (v13 & 0x7FFFFFF));
  v111 = &v113;
  if ( v15 == (__int64 *)v12 )
  {
    v26 = 0;
    goto LABEL_14;
  }
  v16 = 0;
  v17 = 0;
  while ( 1 )
  {
    v24 = *v14;
    v106 = (__int64 *)*v14;
    if ( v16 )
    {
      v18 = v16 - 1;
      v19 = 1;
      v20 = 0;
      v21 = v18 & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
      v22 = (__int64 *)(v17 + 8LL * v21);
      v23 = *v22;
      if ( v24 == *v22 )
        goto LABEL_7;
      while ( v23 != -4096 )
      {
        if ( v23 != -8192 || v20 )
          v22 = v20;
        v21 = v18 & (v19 + v21);
        v23 = *(_QWORD *)(v17 + 8LL * v21);
        if ( v24 == v23 )
          goto LABEL_7;
        ++v19;
        v20 = v22;
        v22 = (__int64 *)(v17 + 8LL * v21);
      }
      if ( !v20 )
        v20 = v22;
      ++v107;
      v25 = v109 + 1;
      v113 = v20;
      if ( 4 * ((int)v109 + 1) < 3 * v16 )
      {
        if ( v16 - (v25 + HIDWORD(v109)) > v16 >> 3 )
          goto LABEL_142;
        goto LABEL_12;
      }
    }
    else
    {
      ++v107;
      v113 = 0;
    }
    v16 *= 2;
LABEL_12:
    sub_CE2A30((__int64)&v107, v16);
    sub_DA5B20((__int64)&v107, (__int64 *)&v106, &v113);
    v24 = (__int64)v106;
    v20 = v113;
    v25 = v109 + 1;
LABEL_142:
    LODWORD(v109) = v25;
    if ( *v20 != -4096 )
      --HIDWORD(v109);
    *v20 = v24;
    v95 = (unsigned int)v112;
    v96 = (__int64)v106;
    v97 = (unsigned int)v112 + 1LL;
    if ( v97 > HIDWORD(v112) )
    {
      sub_C8D5F0((__int64)&v111, &v113, v97, 8u, v23, v18);
      v95 = (unsigned int)v112;
    }
    v111[v95] = (__int64 *)v96;
    LODWORD(v112) = v112 + 1;
LABEL_7:
    v14 += 4;
    if ( v15 == v14 )
      break;
    v17 = v108;
    v16 = v110;
  }
  v26 = (unsigned int)v112;
LABEL_14:
  v115 = 8;
  v113 = (__int64 *)v116;
  v114 = 0;
  if ( !(_DWORD)v26 )
    goto LABEL_53;
  while ( 2 )
  {
    v104 = 0;
    while ( 2 )
    {
      while ( 2 )
      {
        v27 = (__int64)v111[v26 - 1];
        if ( (_DWORD)v110 )
        {
          v28 = (v110 - 1) & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
          v29 = (__int64 *)(v108 + 8LL * v28);
          v30 = *v29;
          if ( v27 == *v29 )
          {
LABEL_18:
            *v29 = -8192;
            LODWORD(v109) = v109 - 1;
            ++HIDWORD(v109);
          }
          else
          {
            v55 = 1;
            while ( v30 != -4096 )
            {
              v56 = v55 + 1;
              v28 = (v110 - 1) & (v55 + v28);
              v29 = (__int64 *)(v108 + 8LL * v28);
              v30 = *v29;
              if ( v27 == *v29 )
                goto LABEL_18;
              v55 = v56;
            }
          }
        }
        v26 = (unsigned int)(v112 - 1);
        LODWORD(v112) = v112 - 1;
        if ( *(_BYTE *)v27 == 84 || *(_BYTE *)v27 <= 0x1Cu )
        {
LABEL_37:
          if ( !(_DWORD)v26 )
            goto LABEL_38;
          continue;
        }
        break;
      }
      v31 = *(_QWORD *)(v27 + 40);
      if ( !*(_BYTE *)(v11 + 84) )
      {
        if ( sub_C8CA60(v11 + 56, v31) )
          goto LABEL_26;
        goto LABEL_36;
      }
      v32 = *(_QWORD **)(v11 + 64);
      v33 = &v32[*(unsigned int *)(v11 + 76)];
      if ( v32 == v33 )
        goto LABEL_37;
      while ( v31 != *v32 )
      {
        if ( v33 == ++v32 )
          goto LABEL_37;
      }
LABEL_26:
      if ( (unsigned __int8)sub_B46970((unsigned __int8 *)v27) || (unsigned __int8)sub_B46420(v27) )
      {
LABEL_36:
        v26 = (unsigned int)v112;
        goto LABEL_37;
      }
      if ( v102 == *(_QWORD *)(v27 + 40) )
      {
        v79 = (__int64 *)sub_986520(v27);
        v80 = 4LL * (*(_DWORD *)(v27 + 4) & 0x7FFFFFF);
        v81 = &v79[v80];
        if ( v79 == &v79[v80] )
          goto LABEL_36;
        while ( 1 )
        {
          v89 = *v79;
          v90 = v110;
          v105 = *v79;
          if ( !(_DWORD)v110 )
            break;
          v82 = (unsigned int)(v110 - 1);
          v83 = v108;
          v84 = 1;
          v85 = 0;
          v86 = v82 & (((unsigned int)v89 >> 9) ^ ((unsigned int)v89 >> 4));
          v87 = (__int64 *)(v108 + 8LL * v86);
          v88 = *v87;
          if ( *v87 != v89 )
          {
            while ( v88 != -4096 )
            {
              if ( v85 || v88 != -8192 )
                v87 = v85;
              v86 = v82 & (v84 + v86);
              v88 = *(_QWORD *)(v108 + 8LL * v86);
              if ( v89 == v88 )
                goto LABEL_113;
              ++v84;
              v85 = v87;
              v87 = (__int64 *)(v108 + 8LL * v86);
            }
            if ( !v85 )
              v85 = v87;
            ++v107;
            v91 = v109 + 1;
            v106 = v85;
            if ( 4 * ((int)v109 + 1) < (unsigned int)(3 * v110) )
            {
              if ( (int)v110 - HIDWORD(v109) - v91 > (unsigned int)v110 >> 3 )
                goto LABEL_128;
              goto LABEL_117;
            }
LABEL_116:
            v90 = 2 * v110;
LABEL_117:
            sub_CE2A30((__int64)&v107, v90);
            sub_DA5B20((__int64)&v107, &v105, &v106);
            v89 = v105;
            v85 = v106;
            v91 = v109 + 1;
LABEL_128:
            LODWORD(v109) = v91;
            if ( *v85 != -4096 )
              --HIDWORD(v109);
            *v85 = v89;
            v92 = (unsigned int)v112;
            v93 = v105;
            v94 = (unsigned int)v112 + 1LL;
            if ( v94 > HIDWORD(v112) )
            {
              sub_C8D5F0((__int64)&v111, &v113, v94, 8u, v83, v82);
              v92 = (unsigned int)v112;
            }
            v111[v92] = (__int64 *)v93;
            LODWORD(v112) = v112 + 1;
          }
LABEL_113:
          v79 += 4;
          if ( v81 == v79 )
            goto LABEL_36;
        }
        ++v107;
        v106 = 0;
        goto LABEL_116;
      }
      for ( i = *(_QWORD *)(v27 + 16); i; i = *(_QWORD *)(i + 8) )
      {
        v35 = *(_QWORD *)(i + 24);
        if ( *(_BYTE *)v35 == 84 )
        {
          if ( v102 != *(_QWORD *)(*(_QWORD *)(v35 - 8)
                                 + 32LL * *(unsigned int *)(v35 + 72)
                                 + 8LL * (unsigned int)sub_BD2910(i)) )
            goto LABEL_35;
        }
        else if ( v102 != *(_QWORD *)(v35 + 40) )
        {
LABEL_35:
          sub_9C95B0((__int64)&v113, v27);
          goto LABEL_36;
        }
      }
      v58 = sub_AA5190(v102);
      if ( v58 )
      {
        v59 = v57;
        v60 = HIBYTE(v57);
      }
      else
      {
        v60 = 0;
        v59 = 0;
      }
      v61 = v103;
      LOBYTE(v61) = v59;
      v62 = v61;
      BYTE1(v62) = v60;
      v103 = v62;
      sub_B444E0((_QWORD *)v27, v58, v62);
      if ( (*(_BYTE *)(v27 + 7) & 0x40) != 0 )
      {
        v63 = *(__int64 **)(v27 - 8);
        v64 = &v63[4 * (*(_DWORD *)(v27 + 4) & 0x7FFFFFF)];
      }
      else
      {
        v64 = (__int64 *)v27;
        v63 = (__int64 *)(v27 - 32LL * (*(_DWORD *)(v27 + 4) & 0x7FFFFFF));
      }
      while ( 2 )
      {
        if ( v64 != v63 )
        {
          v72 = *v63;
          v105 = *v63;
          if ( !(_DWORD)v110 )
          {
            ++v107;
            v106 = 0;
            goto LABEL_89;
          }
          v65 = (unsigned int)(v110 - 1);
          v66 = v108;
          v67 = 1;
          v68 = 0;
          v69 = v65 & (((unsigned int)v72 >> 9) ^ ((unsigned int)v72 >> 4));
          v70 = (_QWORD *)(v108 + 8LL * v69);
          v71 = *v70;
          if ( v72 != *v70 )
          {
            while ( v71 != -4096 )
            {
              if ( v71 != -8192 || v68 )
                v70 = v68;
              v69 = v65 & (v67 + v69);
              v71 = *(_QWORD *)(v108 + 8LL * v69);
              if ( v72 == v71 )
                goto LABEL_85;
              ++v67;
              v68 = v70;
              v70 = (_QWORD *)(v108 + 8LL * v69);
            }
            if ( !v68 )
              v68 = v70;
            ++v107;
            v75 = v109 + 1;
            v106 = v68;
            if ( 4 * ((int)v109 + 1) >= (unsigned int)(3 * v110) )
            {
LABEL_89:
              sub_CE2A30((__int64)&v107, 2 * v110);
              if ( (_DWORD)v110 )
              {
                v72 = v105;
                v66 = v108;
                v73 = (v110 - 1) & (((unsigned int)v105 >> 9) ^ ((unsigned int)v105 >> 4));
                v68 = (_QWORD *)(v108 + 8LL * v73);
                v74 = *v68;
                if ( *v68 == v105 )
                {
LABEL_91:
                  v106 = v68;
                  v75 = v109 + 1;
                }
                else
                {
                  v65 = 1;
                  v100 = 0;
                  while ( v74 != -4096 )
                  {
                    if ( !v100 && v74 == -8192 )
                      v100 = (__int64)v68;
                    v73 = (v110 - 1) & (v65 + v73);
                    v68 = (_QWORD *)(v108 + 8LL * v73);
                    v74 = *v68;
                    if ( v105 == *v68 )
                      goto LABEL_91;
                    v65 = (unsigned int)(v65 + 1);
                  }
                  if ( !v100 )
                    v100 = (__int64)v68;
                  v75 = v109 + 1;
                  v106 = (__int64 *)v100;
                  v68 = (_QWORD *)v100;
                }
              }
              else
              {
                v72 = v105;
                v68 = 0;
                v106 = 0;
                v75 = v109 + 1;
              }
            }
            else if ( (int)v110 - HIDWORD(v109) - v75 <= (unsigned int)v110 >> 3 )
            {
              sub_CE2A30((__int64)&v107, v110);
              sub_DA5B20((__int64)&v107, &v105, &v106);
              v72 = v105;
              v68 = v106;
              v75 = v109 + 1;
            }
            LODWORD(v109) = v75;
            if ( *v68 != -4096 )
              --HIDWORD(v109);
            *v68 = v72;
            v76 = (unsigned int)v112;
            v77 = v105;
            v78 = (unsigned int)v112 + 1LL;
            if ( v78 > HIDWORD(v112) )
            {
              sub_C8D5F0((__int64)&v111, &v113, v78, 8u, v66, v65);
              v76 = (unsigned int)v112;
            }
            v111[v76] = (__int64 *)v77;
            LODWORD(v112) = v112 + 1;
          }
LABEL_85:
          v63 += 4;
          continue;
        }
        break;
      }
      v26 = (unsigned int)v112;
      v104 = 1;
      if ( (_DWORD)v112 )
        continue;
      break;
    }
LABEL_38:
    v36 = v113;
    v37 = v113;
    if ( !v104 )
      goto LABEL_54;
    v38 = &v113[v114];
    if ( v38 != v113 )
    {
      while ( 1 )
      {
        v45 = (__int64 *)*v36;
        if ( !(_DWORD)v110 )
          break;
        v39 = (unsigned int)(v110 - 1);
        v40 = 1;
        v41 = 0;
        v42 = v39 & (((unsigned int)v45 >> 9) ^ ((unsigned int)v45 >> 4));
        v43 = (__int64 **)(v108 + 8LL * v42);
        v44 = *v43;
        if ( v45 == *v43 )
        {
LABEL_42:
          if ( v38 == ++v36 )
            goto LABEL_52;
        }
        else
        {
          while ( v44 != (__int64 *)-4096LL )
          {
            if ( v41 || v44 != (__int64 *)-8192LL )
              v43 = v41;
            v42 = v39 & (v40 + v42);
            v44 = *(__int64 **)(v108 + 8LL * v42);
            if ( v45 == v44 )
              goto LABEL_42;
            ++v40;
            v41 = v43;
            v43 = (__int64 **)(v108 + 8LL * v42);
          }
          if ( !v41 )
            v41 = v43;
          ++v107;
          v48 = v109 + 1;
          if ( 4 * ((int)v109 + 1) < (unsigned int)(3 * v110) )
          {
            if ( (int)v110 - HIDWORD(v109) - v48 <= (unsigned int)v110 >> 3 )
            {
              sub_CE2A30((__int64)&v107, v110);
              if ( !(_DWORD)v110 )
              {
LABEL_183:
                LODWORD(v109) = v109 + 1;
                BUG();
              }
              v39 = 1;
              v52 = (v110 - 1) & (((unsigned int)v45 >> 9) ^ ((unsigned int)v45 >> 4));
              v41 = (__int64 **)(v108 + 8LL * v52);
              v48 = v109 + 1;
              v53 = 0;
              v54 = *v41;
              if ( *v41 != v45 )
              {
                while ( v54 != (__int64 *)-4096LL )
                {
                  if ( !v53 && v54 == (__int64 *)-8192LL )
                    v53 = v41;
                  v52 = (v110 - 1) & (v39 + v52);
                  v41 = (__int64 **)(v108 + 8LL * v52);
                  v54 = *v41;
                  if ( v45 == *v41 )
                    goto LABEL_47;
                  v39 = (unsigned int)(v39 + 1);
                }
                if ( v53 )
                  v41 = v53;
              }
            }
            goto LABEL_47;
          }
LABEL_45:
          sub_CE2A30((__int64)&v107, 2 * v110);
          if ( !(_DWORD)v110 )
            goto LABEL_183;
          v46 = (v110 - 1) & (((unsigned int)v45 >> 9) ^ ((unsigned int)v45 >> 4));
          v41 = (__int64 **)(v108 + 8LL * v46);
          v47 = *v41;
          v48 = v109 + 1;
          if ( v45 != *v41 )
          {
            v99 = 1;
            v39 = 0;
            while ( v47 != (__int64 *)-4096LL )
            {
              if ( !v39 && v47 == (__int64 *)-8192LL )
                v39 = (__int64)v41;
              v46 = (v110 - 1) & (v99 + v46);
              v41 = (__int64 **)(v108 + 8LL * v46);
              v47 = *v41;
              if ( v45 == *v41 )
                goto LABEL_47;
              ++v99;
            }
            if ( v39 )
              v41 = (__int64 **)v39;
          }
LABEL_47:
          LODWORD(v109) = v48;
          if ( *v41 != (__int64 *)-4096LL )
            --HIDWORD(v109);
          *v41 = v45;
          v49 = (unsigned int)v112;
          v50 = (unsigned int)v112 + 1LL;
          if ( v50 > HIDWORD(v112) )
          {
            sub_C8D5F0((__int64)&v111, &v113, v50, 8u, v39, (__int64)v41);
            v49 = (unsigned int)v112;
          }
          ++v36;
          v111[v49] = v45;
          LODWORD(v112) = v112 + 1;
          if ( v38 == v36 )
            goto LABEL_52;
        }
      }
      ++v107;
      goto LABEL_45;
    }
LABEL_52:
    v26 = (unsigned int)v112;
    v114 = 0;
    if ( (_DWORD)v112 )
      continue;
    break;
  }
LABEL_53:
  v37 = v113;
LABEL_54:
  if ( v37 != (__int64 *)v116 )
    _libc_free((unsigned __int64)v37);
  if ( v111 != &v113 )
    _libc_free((unsigned __int64)v111);
  return sub_C7D6A0(v108, 8LL * (unsigned int)v110, 8);
}
