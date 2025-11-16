// Function: sub_2981FA0
// Address: 0x2981fa0
//
__int64 __fastcall sub_2981FA0(__int64 a1)
{
  __int64 v1; // rax
  __int64 *v2; // r14
  __int64 v3; // rax
  unsigned int v4; // ecx
  __int64 v5; // r8
  __int64 v6; // rdi
  __int64 *v7; // rsi
  unsigned int v8; // ecx
  unsigned int v9; // edx
  __int64 *v10; // rax
  __int64 v11; // r9
  __int64 *v12; // r15
  __int64 *v13; // rbx
  int v14; // r10d
  _QWORD *v15; // rdx
  unsigned int v16; // edi
  __int64 *v17; // rax
  __int64 v18; // rcx
  __int64 v19; // r12
  unsigned int v20; // eax
  int v21; // ecx
  __int64 v22; // r8
  _DWORD *v23; // rdx
  _QWORD *v24; // r8
  unsigned int v25; // r13d
  int v26; // r10d
  __int64 v27; // rsi
  int v28; // eax
  __int64 v29; // r13
  __int64 *v30; // rbx
  __int64 *v31; // r13
  int v32; // r14d
  __int64 *v33; // r9
  unsigned int v34; // edx
  __int64 *v35; // rax
  __int64 v36; // r11
  unsigned int v37; // eax
  int v38; // edx
  __int64 v39; // r8
  __int64 v40; // rax
  __int64 *v41; // rax
  __int64 *v42; // rax
  __int64 v43; // rdx
  __int64 v44; // r9
  __int64 v45; // r8
  __int64 *v46; // r10
  int v47; // r11d
  unsigned int v48; // eax
  __int64 *v49; // rdi
  __int64 v50; // rcx
  unsigned int v51; // eax
  __int64 v52; // rsi
  __int64 v53; // rcx
  int v54; // r11d
  __int64 *v55; // r10
  int v56; // r11d
  unsigned int v57; // eax
  __int64 v58; // r8
  __int64 *v59; // r12
  _BYTE *v60; // rsi
  __int64 v61; // rax
  __int64 *v62; // r13
  __int64 *v63; // r12
  int v64; // r11d
  _QWORD *v65; // rcx
  unsigned int v66; // esi
  __int64 *v67; // rax
  __int64 v68; // rdi
  unsigned int v69; // esi
  __int64 v70; // rcx
  int v71; // r11d
  __int64 *v72; // rdi
  unsigned int v73; // edx
  __int64 *v74; // rax
  __int64 v75; // r8
  __int64 v76; // rdx
  unsigned int v77; // esi
  int v78; // eax
  __int64 v79; // r8
  _DWORD *v80; // rcx
  unsigned int v81; // edx
  int v82; // eax
  int v83; // r11d
  __int64 *v84; // r10
  __int64 v85; // rdx
  __int64 *v87; // rax
  _QWORD *v88; // r10
  int v89; // r11d
  unsigned int v90; // esi
  __int64 v91; // r8
  __int64 *v92; // r10
  int v93; // r11d
  unsigned int v94; // edx
  __int64 v95; // r8
  int v96; // r11d
  unsigned int v97; // eax
  int v98; // r10d
  _QWORD *v99; // r9
  int v100; // r10d
  int v101; // r11d
  int v102; // r11d
  int v103; // r8d
  unsigned int v104; // r10d
  _QWORD *v105; // r9
  __int64 v106; // [rsp+0h] [rbp-F0h]
  __int64 *v108; // [rsp+18h] [rbp-D8h]
  __int64 v109; // [rsp+20h] [rbp-D0h] BYREF
  __int64 v110; // [rsp+28h] [rbp-C8h] BYREF
  __int64 v111; // [rsp+30h] [rbp-C0h] BYREF
  __int64 v112; // [rsp+38h] [rbp-B8h]
  __int64 v113; // [rsp+40h] [rbp-B0h]
  unsigned int v114; // [rsp+48h] [rbp-A8h]
  __int64 v115; // [rsp+50h] [rbp-A0h] BYREF
  __int64 v116; // [rsp+58h] [rbp-98h]
  __int64 v117; // [rsp+60h] [rbp-90h]
  __int64 v118; // [rsp+68h] [rbp-88h]
  __int64 v119[2]; // [rsp+70h] [rbp-80h] BYREF
  unsigned __int64 v120; // [rsp+80h] [rbp-70h]
  unsigned __int64 v121; // [rsp+88h] [rbp-68h]
  __int64 v122; // [rsp+90h] [rbp-60h]
  unsigned __int64 *v123; // [rsp+98h] [rbp-58h]
  __int64 *v124; // [rsp+A0h] [rbp-50h]
  __int64 v125; // [rsp+A8h] [rbp-48h]
  __int64 v126; // [rsp+B0h] [rbp-40h]
  __int64 v127; // [rsp+B8h] [rbp-38h]

  v1 = *(_QWORD *)(a1 + 288);
  v106 = a1 + 288;
  if ( *(_QWORD *)(a1 + 296) != v1 )
    *(_QWORD *)(a1 + 296) = v1;
  v111 = 0;
  v112 = 0;
  v2 = *(__int64 **)(a1 + 240);
  v3 = *(unsigned int *)(a1 + 248);
  v113 = 0;
  v114 = 0;
  v108 = &v2[4 * v3];
  if ( v2 != v108 )
  {
    v4 = 0;
    v5 = 0;
    while ( 1 )
    {
      v6 = *v2;
      v7 = (__int64 *)(v5 + 16LL * v4);
      if ( v4 )
      {
        v8 = v4 - 1;
        v9 = v8 & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
        v10 = (__int64 *)(v5 + 16LL * v9);
        v11 = *v10;
        if ( v6 == *v10 )
        {
LABEL_7:
          if ( v7 != v10 )
            goto LABEL_8;
        }
        else
        {
          v28 = 1;
          while ( v11 != -4096 )
          {
            v100 = v28 + 1;
            v9 = v8 & (v28 + v9);
            v10 = (__int64 *)(v5 + 16LL * v9);
            v11 = *v10;
            if ( v6 == *v10 )
              goto LABEL_7;
            v28 = v100;
          }
        }
      }
      *(_DWORD *)sub_297EDC0((__int64)&v111, v2) = 0;
LABEL_8:
      v12 = (__int64 *)v2[2];
      v13 = (__int64 *)v2[1];
      if ( v12 != v13 )
      {
        while ( 1 )
        {
          v19 = *v13;
          if ( !v114 )
            break;
          v14 = 1;
          v15 = 0;
          v16 = (v114 - 1) & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
          v17 = (__int64 *)(v112 + 16LL * v16);
          v18 = *v17;
          if ( v19 == *v17 )
          {
LABEL_11:
            ++v13;
            ++*((_DWORD *)v17 + 2);
            if ( v12 == v13 )
              goto LABEL_19;
          }
          else
          {
            while ( v18 != -4096 )
            {
              if ( v18 == -8192 && !v15 )
                v15 = v17;
              v16 = (v114 - 1) & (v14 + v16);
              v17 = (__int64 *)(v112 + 16LL * v16);
              v18 = *v17;
              if ( v19 == *v17 )
                goto LABEL_11;
              ++v14;
            }
            if ( !v15 )
              v15 = v17;
            ++v111;
            v21 = v113 + 1;
            if ( 4 * ((int)v113 + 1) < 3 * v114 )
            {
              if ( v114 - HIDWORD(v113) - v21 <= v114 >> 3 )
              {
                sub_2809BA0((__int64)&v111, v114);
                if ( !v114 )
                {
LABEL_240:
                  LODWORD(v113) = v113 + 1;
                  BUG();
                }
                v24 = 0;
                v25 = (v114 - 1) & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
                v26 = 1;
                v21 = v113 + 1;
                v15 = (_QWORD *)(v112 + 16LL * v25);
                v27 = *v15;
                if ( v19 != *v15 )
                {
                  while ( v27 != -4096 )
                  {
                    if ( v24 || v27 != -8192 )
                      v15 = v24;
                    v103 = v26 + 1;
                    v104 = (v114 - 1) & (v25 + v26);
                    v25 = v104;
                    v105 = (_QWORD *)(v112 + 16LL * v104);
                    v27 = *v105;
                    if ( v19 == *v105 )
                    {
                      v15 = (_QWORD *)(v112 + 16LL * v104);
                      goto LABEL_16;
                    }
                    v26 = v103;
                    v24 = v15;
                    v15 = v105;
                  }
                  if ( v24 )
                    v15 = v24;
                }
              }
              goto LABEL_16;
            }
LABEL_14:
            sub_2809BA0((__int64)&v111, 2 * v114);
            if ( !v114 )
              goto LABEL_240;
            v20 = (v114 - 1) & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
            v21 = v113 + 1;
            v15 = (_QWORD *)(v112 + 16LL * v20);
            v22 = *v15;
            if ( v19 != *v15 )
            {
              v98 = 1;
              v99 = 0;
              while ( v22 != -4096 )
              {
                if ( !v99 && v22 == -8192 )
                  v99 = v15;
                v20 = (v114 - 1) & (v98 + v20);
                v15 = (_QWORD *)(v112 + 16LL * v20);
                v22 = *v15;
                if ( v19 == *v15 )
                  goto LABEL_16;
                ++v98;
              }
              if ( v99 )
                v15 = v99;
            }
LABEL_16:
            LODWORD(v113) = v21;
            if ( *v15 != -4096 )
              --HIDWORD(v113);
            *v15 = v19;
            v23 = v15 + 1;
            ++v13;
            *v23 = 0;
            *v23 = 1;
            if ( v12 == v13 )
              goto LABEL_19;
          }
        }
        ++v111;
        goto LABEL_14;
      }
LABEL_19:
      v2 += 4;
      if ( v108 == v2 )
        break;
      v5 = v112;
      v4 = v114;
    }
  }
  v119[0] = 0;
  v119[1] = 0;
  v120 = 0;
  v121 = 0;
  v122 = 0;
  v123 = 0;
  v124 = 0;
  v125 = 0;
  v126 = 0;
  v127 = 0;
  sub_2785050(v119, 0);
  v115 = 0;
  v116 = 0;
  v29 = *(unsigned int *)(a1 + 248);
  v30 = *(__int64 **)(a1 + 240);
  v117 = 0;
  v118 = 0;
  v31 = &v30[4 * v29];
  if ( v31 != v30 )
  {
    while ( v114 )
    {
      v32 = 1;
      v33 = 0;
      v34 = (v114 - 1) & (((unsigned int)*v30 >> 9) ^ ((unsigned int)*v30 >> 4));
      v35 = (__int64 *)(v112 + 16LL * v34);
      v36 = *v35;
      if ( *v30 == *v35 )
      {
LABEL_43:
        if ( !*((_DWORD *)v35 + 2) )
          goto LABEL_52;
LABEL_44:
        v30 += 4;
        if ( v31 == v30 )
          goto LABEL_56;
      }
      else
      {
        while ( v36 != -4096 )
        {
          if ( !v33 && v36 == -8192 )
            v33 = v35;
          v34 = (v114 - 1) & (v32 + v34);
          v35 = (__int64 *)(v112 + 16LL * v34);
          v36 = *v35;
          if ( *v30 == *v35 )
            goto LABEL_43;
          ++v32;
        }
        if ( !v33 )
          v33 = v35;
        ++v111;
        v38 = v113 + 1;
        if ( 4 * ((int)v113 + 1) < 3 * v114 )
        {
          if ( v114 - HIDWORD(v113) - v38 > v114 >> 3 )
            goto LABEL_49;
          sub_2809BA0((__int64)&v111, v114);
          if ( !v114 )
          {
LABEL_239:
            LODWORD(v113) = v113 + 1;
            BUG();
          }
          v55 = 0;
          v56 = 1;
          v57 = (v114 - 1) & (((unsigned int)*v30 >> 9) ^ ((unsigned int)*v30 >> 4));
          v38 = v113 + 1;
          v33 = (__int64 *)(v112 + 16LL * v57);
          v58 = *v33;
          if ( *v30 == *v33 )
            goto LABEL_49;
          while ( v58 != -4096 )
          {
            if ( !v55 && v58 == -8192 )
              v55 = v33;
            v57 = (v114 - 1) & (v56 + v57);
            v33 = (__int64 *)(v112 + 16LL * v57);
            v58 = *v33;
            if ( *v30 == *v33 )
              goto LABEL_49;
            ++v56;
          }
          goto LABEL_84;
        }
LABEL_47:
        sub_2809BA0((__int64)&v111, 2 * v114);
        if ( !v114 )
          goto LABEL_239;
        v37 = (v114 - 1) & (((unsigned int)*v30 >> 9) ^ ((unsigned int)*v30 >> 4));
        v38 = v113 + 1;
        v33 = (__int64 *)(v112 + 16LL * v37);
        v39 = *v33;
        if ( *v33 == *v30 )
          goto LABEL_49;
        v102 = 1;
        v55 = 0;
        while ( v39 != -4096 )
        {
          if ( !v55 && v39 == -8192 )
            v55 = v33;
          v37 = (v114 - 1) & (v102 + v37);
          v33 = (__int64 *)(v112 + 16LL * v37);
          v39 = *v33;
          if ( *v30 == *v33 )
            goto LABEL_49;
          ++v102;
        }
LABEL_84:
        if ( v55 )
          v33 = v55;
LABEL_49:
        LODWORD(v113) = v38;
        if ( *v33 != -4096 )
          --HIDWORD(v113);
        v40 = *v30;
        *((_DWORD *)v33 + 2) = 0;
        *v33 = v40;
LABEL_52:
        v41 = v124;
        if ( v124 == (__int64 *)(v126 - 8) )
        {
          sub_2785520((unsigned __int64 *)v119, v30);
          goto LABEL_44;
        }
        if ( v124 )
        {
          *v124 = *v30;
          v41 = v124;
        }
        v30 += 4;
        v124 = v41 + 1;
        if ( v31 == v30 )
          goto LABEL_56;
      }
    }
    ++v111;
    goto LABEL_47;
  }
LABEL_56:
  v42 = (__int64 *)v120;
  if ( v124 != (__int64 *)v120 )
  {
    while ( 1 )
    {
      v109 = *v42;
      if ( v42 == (__int64 *)(v122 - 8) )
      {
        j_j___libc_free_0(v121);
        v85 = *++v123 + 512;
        v121 = *v123;
        v122 = v85;
        v120 = v121;
      }
      else
      {
        v120 = (unsigned __int64)(v42 + 1);
      }
      if ( !(_DWORD)v118 )
      {
        ++v115;
        goto LABEL_64;
      }
      v43 = v109;
      v44 = (unsigned int)(v118 - 1);
      v45 = v116;
      v46 = 0;
      v47 = 1;
      v48 = v44 & (((unsigned int)v109 >> 9) ^ ((unsigned int)v109 >> 4));
      v49 = (__int64 *)(v116 + 8LL * v48);
      v50 = *v49;
      if ( v109 != *v49 )
        break;
LABEL_59:
      v42 = (__int64 *)v120;
      if ( v124 == (__int64 *)v120 )
        goto LABEL_125;
    }
    while ( v50 != -4096 )
    {
      if ( v46 || v50 != -8192 )
        v49 = v46;
      v48 = v44 & (v47 + v48);
      v59 = (__int64 *)(v116 + 8LL * v48);
      v50 = *v59;
      if ( v109 == *v59 )
        goto LABEL_59;
      ++v47;
      v46 = v49;
      v49 = (__int64 *)(v116 + 8LL * v48);
    }
    if ( !v46 )
      v46 = v49;
    ++v115;
    v53 = (unsigned int)(v117 + 1);
    if ( 4 * (int)v53 < (unsigned int)(3 * v118) )
    {
      if ( (int)v118 - HIDWORD(v117) - (int)v53 <= (unsigned int)v118 >> 3 )
      {
        sub_CF4090((__int64)&v115, v118);
        if ( !(_DWORD)v118 )
        {
LABEL_237:
          LODWORD(v117) = v117 + 1;
          BUG();
        }
        v44 = 0;
        v45 = v116;
        v96 = 1;
        v53 = (unsigned int)(v117 + 1);
        v97 = (v118 - 1) & (((unsigned int)v109 >> 9) ^ ((unsigned int)v109 >> 4));
        v46 = (__int64 *)(v116 + 8LL * v97);
        v43 = *v46;
        if ( v109 != *v46 )
        {
          while ( v43 != -4096 )
          {
            if ( v43 == -8192 && !v44 )
              v44 = (__int64)v46;
            v97 = (v118 - 1) & (v96 + v97);
            v46 = (__int64 *)(v116 + 8LL * v97);
            v43 = *v46;
            if ( v109 == *v46 )
              goto LABEL_96;
            ++v96;
          }
          v43 = v109;
          if ( v44 )
            v46 = (__int64 *)v44;
        }
      }
      goto LABEL_96;
    }
LABEL_64:
    sub_CF4090((__int64)&v115, 2 * v118);
    if ( !(_DWORD)v118 )
      goto LABEL_237;
    v43 = v109;
    v45 = v116;
    v51 = (v118 - 1) & (((unsigned int)v109 >> 9) ^ ((unsigned int)v109 >> 4));
    v46 = (__int64 *)(v116 + 8LL * v51);
    v52 = *v46;
    v53 = (unsigned int)(v117 + 1);
    if ( *v46 != v109 )
    {
      v54 = 1;
      v44 = 0;
      while ( v52 != -4096 )
      {
        if ( !v44 && v52 == -8192 )
          v44 = (__int64)v46;
        v51 = (v118 - 1) & (v54 + v51);
        v46 = (__int64 *)(v116 + 8LL * v51);
        v52 = *v46;
        if ( v109 == *v46 )
          goto LABEL_96;
        ++v54;
      }
      if ( v44 )
        v46 = (__int64 *)v44;
    }
LABEL_96:
    LODWORD(v117) = v53;
    if ( *v46 != -4096 )
      --HIDWORD(v117);
    *v46 = v43;
    v60 = *(_BYTE **)(a1 + 296);
    if ( v60 == *(_BYTE **)(a1 + 304) )
    {
      sub_24454E0(v106, v60, &v109);
    }
    else
    {
      if ( v60 )
      {
        *(_QWORD *)v60 = v109;
        v60 = *(_BYTE **)(a1 + 296);
      }
      *(_QWORD *)(a1 + 296) = v60 + 8;
    }
    v61 = sub_29812B0(a1 + 208, &v109, v43, v53, v45, v44);
    v62 = *(__int64 **)(v61 + 8);
    v63 = *(__int64 **)v61;
    if ( v62 == *(__int64 **)v61 )
      goto LABEL_59;
    while ( 1 )
    {
      v76 = *v63;
      v110 = *v63;
      if ( !v114 )
        break;
      v64 = 1;
      v65 = 0;
      v66 = (v114 - 1) & (((unsigned int)v76 >> 9) ^ ((unsigned int)v76 >> 4));
      v67 = (__int64 *)(v112 + 16LL * v66);
      v68 = *v67;
      if ( v76 == *v67 )
      {
LABEL_105:
        --*((_DWORD *)v67 + 2);
        v69 = v114;
        if ( !v114 )
          goto LABEL_116;
        goto LABEL_106;
      }
      while ( v68 != -4096 )
      {
        if ( v68 == -8192 && !v65 )
          v65 = v67;
        v66 = (v114 - 1) & (v64 + v66);
        v67 = (__int64 *)(v112 + 16LL * v66);
        v68 = *v67;
        if ( v76 == *v67 )
          goto LABEL_105;
        ++v64;
      }
      if ( !v65 )
        v65 = v67;
      ++v111;
      v78 = v113 + 1;
      if ( 4 * ((int)v113 + 1) >= 3 * v114 )
        goto LABEL_111;
      if ( v114 - HIDWORD(v113) - v78 <= v114 >> 3 )
      {
        sub_2809BA0((__int64)&v111, v114);
        if ( !v114 )
        {
LABEL_241:
          LODWORD(v113) = v113 + 1;
          BUG();
        }
        v76 = v110;
        v88 = 0;
        v89 = 1;
        v90 = (v114 - 1) & (((unsigned int)v110 >> 9) ^ ((unsigned int)v110 >> 4));
        v78 = v113 + 1;
        v65 = (_QWORD *)(v112 + 16LL * v90);
        v91 = *v65;
        if ( *v65 != v110 )
        {
          while ( v91 != -4096 )
          {
            if ( v91 == -8192 && !v88 )
              v88 = v65;
            v90 = (v114 - 1) & (v89 + v90);
            v65 = (_QWORD *)(v112 + 16LL * v90);
            v91 = *v65;
            if ( v110 == *v65 )
              goto LABEL_113;
            ++v89;
          }
          goto LABEL_156;
        }
      }
LABEL_113:
      LODWORD(v113) = v78;
      if ( *v65 != -4096 )
        --HIDWORD(v113);
      *v65 = v76;
      v80 = v65 + 1;
      *v80 = 0;
      *v80 = -1;
      v69 = v114;
      if ( !v114 )
      {
LABEL_116:
        ++v111;
        goto LABEL_117;
      }
LABEL_106:
      v70 = v110;
      v71 = 1;
      v72 = 0;
      v73 = (v69 - 1) & (((unsigned int)v110 >> 9) ^ ((unsigned int)v110 >> 4));
      v74 = (__int64 *)(v112 + 16LL * v73);
      v75 = *v74;
      if ( v110 != *v74 )
      {
        while ( v75 != -4096 )
        {
          if ( v75 == -8192 && !v72 )
            v72 = v74;
          v73 = (v69 - 1) & (v71 + v73);
          v74 = (__int64 *)(v112 + 16LL * v73);
          v75 = *v74;
          if ( v110 == *v74 )
            goto LABEL_107;
          ++v71;
        }
        if ( !v72 )
          v72 = v74;
        ++v111;
        v82 = v113 + 1;
        if ( 4 * ((int)v113 + 1) >= 3 * v69 )
        {
LABEL_117:
          sub_2809BA0((__int64)&v111, 2 * v69);
          if ( !v114 )
            goto LABEL_238;
          v81 = (v114 - 1) & (((unsigned int)v110 >> 9) ^ ((unsigned int)v110 >> 4));
          v82 = v113 + 1;
          v72 = (__int64 *)(v112 + 16LL * v81);
          v70 = *v72;
          if ( v110 != *v72 )
          {
            v83 = 1;
            v84 = 0;
            while ( v70 != -4096 )
            {
              if ( !v84 && v70 == -8192 )
                v84 = v72;
              v81 = (v114 - 1) & (v83 + v81);
              v72 = (__int64 *)(v112 + 16LL * v81);
              v70 = *v72;
              if ( v110 == *v72 )
                goto LABEL_136;
              ++v83;
            }
            v70 = v110;
            if ( v84 )
              v72 = v84;
          }
        }
        else if ( v69 - HIDWORD(v113) - v82 <= v69 >> 3 )
        {
          sub_2809BA0((__int64)&v111, v69);
          if ( !v114 )
          {
LABEL_238:
            LODWORD(v113) = v113 + 1;
            BUG();
          }
          v70 = v110;
          v92 = 0;
          v93 = 1;
          v94 = (v114 - 1) & (((unsigned int)v110 >> 9) ^ ((unsigned int)v110 >> 4));
          v82 = v113 + 1;
          v72 = (__int64 *)(v112 + 16LL * v94);
          v95 = *v72;
          if ( *v72 != v110 )
          {
            while ( v95 != -4096 )
            {
              if ( v95 == -8192 && !v92 )
                v92 = v72;
              v94 = (v114 - 1) & (v93 + v94);
              v72 = (__int64 *)(v112 + 16LL * v94);
              v95 = *v72;
              if ( v110 == *v72 )
                goto LABEL_136;
              ++v93;
            }
            if ( v92 )
              v72 = v92;
          }
        }
LABEL_136:
        LODWORD(v113) = v82;
        if ( *v72 != -4096 )
          --HIDWORD(v113);
        *v72 = v70;
        *((_DWORD *)v72 + 2) = 0;
LABEL_139:
        v87 = v124;
        if ( v124 == (__int64 *)(v126 - 8) )
        {
          sub_2785520((unsigned __int64 *)v119, &v110);
        }
        else
        {
          if ( v124 )
          {
            *v124 = v110;
            v87 = v124;
          }
          v124 = v87 + 1;
        }
        goto LABEL_108;
      }
LABEL_107:
      if ( !*((_DWORD *)v74 + 2) )
        goto LABEL_139;
LABEL_108:
      if ( v62 == ++v63 )
        goto LABEL_59;
    }
    ++v111;
LABEL_111:
    sub_2809BA0((__int64)&v111, 2 * v114);
    if ( !v114 )
      goto LABEL_241;
    v76 = v110;
    v77 = (v114 - 1) & (((unsigned int)v110 >> 9) ^ ((unsigned int)v110 >> 4));
    v78 = v113 + 1;
    v65 = (_QWORD *)(v112 + 16LL * v77);
    v79 = *v65;
    if ( *v65 != v110 )
    {
      v101 = 1;
      v88 = 0;
      while ( v79 != -4096 )
      {
        if ( !v88 && v79 == -8192 )
          v88 = v65;
        v77 = (v114 - 1) & (v101 + v77);
        v65 = (_QWORD *)(v112 + 16LL * v77);
        v79 = *v65;
        if ( v110 == *v65 )
          goto LABEL_113;
        ++v101;
      }
LABEL_156:
      if ( v88 )
        v65 = v88;
      goto LABEL_113;
    }
    goto LABEL_113;
  }
LABEL_125:
  sub_C7D6A0(v116, 8LL * (unsigned int)v118, 8);
  sub_2784FD0((unsigned __int64 *)v119);
  return sub_C7D6A0(v112, 16LL * v114, 8);
}
