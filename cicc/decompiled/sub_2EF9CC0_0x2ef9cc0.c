// Function: sub_2EF9CC0
// Address: 0x2ef9cc0
//
void __fastcall sub_2EF9CC0(__int64 a1)
{
  __int64 v1; // r14
  __int64 v2; // rdx
  __int64 v3; // rdi
  __int64 *v4; // r15
  __int64 v5; // rdx
  __int64 v6; // r8
  __int64 v7; // r9
  _QWORD *v8; // r14
  __int64 v9; // rcx
  int *v10; // rax
  int *v11; // r12
  int v12; // r13d
  int *v13; // rbx
  char v14; // r11
  int v15; // eax
  unsigned int v16; // eax
  int v17; // esi
  __int64 *v18; // rax
  __int64 v19; // rax
  __int64 v20; // r14
  unsigned int v21; // ebx
  __int64 v22; // r15
  int v23; // r14d
  __int64 v24; // rdx
  __int64 v25; // rax
  int v26; // r13d
  _QWORD *v27; // rax
  _QWORD *v28; // r12
  __int64 v29; // rdi
  int v30; // eax
  int v31; // edx
  unsigned int v32; // eax
  int v33; // esi
  unsigned int v34; // edi
  __int64 v35; // rax
  __int64 v36; // rsi
  __int64 *v37; // rdx
  __int64 *v38; // rax
  __int64 *v39; // rcx
  __int64 *v40; // rdx
  __int64 *v41; // rax
  __int64 v42; // rdi
  _QWORD *v43; // rax
  __int64 v44; // rdx
  __int64 *v45; // rax
  __int64 *v46; // r15
  __int64 v47; // rcx
  __int64 v48; // r8
  __int64 v49; // r9
  _QWORD *v50; // r14
  __int64 v51; // rdx
  int *v52; // rbx
  int *v53; // r12
  char v54; // r11
  int v55; // r13d
  int v56; // eax
  unsigned int v57; // eax
  int v58; // esi
  __int64 *v59; // rax
  int v60; // edi
  unsigned int v61; // esi
  int *v62; // rax
  int v63; // edi
  int *v64; // r10
  int v65; // edi
  int v66; // edi
  int v67; // esi
  int v68; // edi
  _DWORD *v69; // rax
  int v70; // ecx
  int v71; // r9d
  int v72; // esi
  __int64 v73; // r9
  __int64 v74; // r8
  __int64 v75; // r10
  unsigned int v76; // edx
  int *v77; // rax
  int v78; // edi
  int *v79; // rcx
  __int64 v80; // rcx
  __int64 v81; // rdx
  __int64 *v82; // rax
  __int64 *v83; // rax
  int v84; // r8d
  int *v85; // rsi
  __int64 v86; // r10
  int v87; // edi
  int v88; // r8d
  int v89; // r11d
  int v90; // edi
  int v91; // r8d
  int v92; // r11d
  int v93; // edi
  _DWORD *v94; // rsi
  int v95; // r8d
  int v96; // esi
  __int64 v97; // r11
  int v98; // edi
  int v99; // eax
  unsigned int v100; // esi
  int v101; // r10d
  int *v102; // rdi
  int v103; // eax
  int v104; // esi
  int v105; // r10d
  unsigned int v106; // r11d
  unsigned int v107; // r11d
  unsigned int v108; // [rsp+Ch] [rbp-E4h]
  unsigned int v109; // [rsp+Ch] [rbp-E4h]
  __int64 v110; // [rsp+10h] [rbp-E0h]
  int v111; // [rsp+18h] [rbp-D8h]
  unsigned int v112; // [rsp+18h] [rbp-D8h]
  __int64 v113; // [rsp+20h] [rbp-D0h]
  int v114; // [rsp+20h] [rbp-D0h]
  __int64 v115; // [rsp+28h] [rbp-C8h]
  __int64 v116; // [rsp+28h] [rbp-C8h]
  int v117; // [rsp+28h] [rbp-C8h]
  int v118; // [rsp+28h] [rbp-C8h]
  __int64 *v119; // [rsp+30h] [rbp-C0h]
  __int64 v120; // [rsp+30h] [rbp-C0h]
  _QWORD *v121; // [rsp+38h] [rbp-B8h]
  __int64 v122; // [rsp+38h] [rbp-B8h]
  _QWORD *v123; // [rsp+38h] [rbp-B8h]
  __int64 v124; // [rsp+40h] [rbp-B0h]
  __int64 *v125; // [rsp+40h] [rbp-B0h]
  __int64 v126; // [rsp+50h] [rbp-A0h] BYREF
  __int64 v127; // [rsp+58h] [rbp-98h] BYREF
  __int64 v128; // [rsp+60h] [rbp-90h] BYREF
  __int64 *v129; // [rsp+68h] [rbp-88h]
  __int64 v130; // [rsp+70h] [rbp-80h]
  int v131; // [rsp+78h] [rbp-78h]
  char v132; // [rsp+7Ch] [rbp-74h]
  char v133; // [rsp+80h] [rbp-70h] BYREF

  v1 = a1 + 600;
  v129 = (__int64 *)&v133;
  v2 = *(_QWORD *)(a1 + 32);
  v128 = 0;
  v3 = *(_QWORD *)(v2 + 328);
  v132 = 1;
  v130 = 8;
  v131 = 0;
  v113 = v3;
  v110 = v2 + 320;
  if ( v3 == v2 + 320 )
    return;
  v124 = v1;
  do
  {
    v127 = v113;
    v121 = sub_2EEFC50(v124, &v127);
    v4 = *(__int64 **)(v113 + 64);
    v119 = &v4[*(unsigned int *)(v113 + 72)];
    if ( v119 == v4 )
      goto LABEL_26;
    do
    {
      while ( 1 )
      {
        v127 = *v4;
        v8 = sub_2EEFC50(v124, &v127);
        v9 = *((unsigned int *)v121 + 6);
        if ( (_DWORD)v9 )
        {
          v10 = (int *)v121[2];
          v11 = &v10[4 * *((unsigned int *)v121 + 8)];
          if ( v10 != v11 )
          {
            while ( 1 )
            {
              v12 = *v10;
              v13 = v10;
              if ( (unsigned int)*v10 <= 0xFFFFFFFD )
                break;
              v10 += 4;
              if ( v11 == v10 )
                goto LABEL_5;
            }
            if ( v11 != v10 )
              break;
          }
        }
LABEL_5:
        if ( v119 == ++v4 )
          goto LABEL_26;
      }
      v14 = 0;
      v115 = (__int64)(v8 + 17);
      while ( 1 )
      {
        if ( v12 < 0 )
        {
          v15 = *((_DWORD *)v8 + 24);
          v9 = v8[10];
          if ( !v15 )
            goto LABEL_90;
          v5 = (unsigned int)(v15 - 1);
          v16 = v5 & (37 * v12);
          v17 = *(_DWORD *)(v9 + 4LL * v16);
          if ( v12 != v17 )
          {
            v60 = 1;
            while ( v17 != -1 )
            {
              v6 = (unsigned int)(v60 + 1);
              v16 = v5 & (v60 + v16);
              v17 = *(_DWORD *)(v9 + 4LL * v16);
              if ( v17 == v12 )
                goto LABEL_16;
              ++v60;
            }
LABEL_90:
            v61 = *((_DWORD *)v8 + 40);
            if ( v61 )
            {
              v7 = v61 - 1;
              v6 = v8[18];
              v5 = (unsigned int)(37 * v12);
              v9 = (unsigned int)v7 & (37 * v12);
              v62 = (int *)(v6 + 4 * v9);
              v63 = *v62;
              if ( *v62 == v12 )
                goto LABEL_16;
              v111 = 1;
              v64 = 0;
              v108 = 37 * v12;
              while ( v63 != -1 )
              {
                if ( !v64 && v63 == -2 )
                  v64 = v62;
                v9 = (unsigned int)v7 & (v111 + (_DWORD)v9);
                v5 = (unsigned int)(v111 + 1);
                v62 = (int *)(v6 + 4LL * (unsigned int)v9);
                v63 = *v62;
                if ( *v62 == v12 )
                  goto LABEL_16;
                ++v111;
              }
              v65 = *((_DWORD *)v8 + 38);
              v5 = v108;
              if ( v64 )
                v62 = v64;
              ++v8[17];
              v9 = (unsigned int)(v65 + 1);
              if ( 4 * (int)v9 < 3 * v61 )
              {
                v6 = v61 >> 3;
                if ( v61 - *((_DWORD *)v8 + 39) - (unsigned int)v9 <= (unsigned int)v6 )
                {
                  sub_2E29BA0(v115, v61);
                  v84 = *((_DWORD *)v8 + 40);
                  if ( !v84 )
                    goto LABEL_232;
                  v6 = (unsigned int)(v84 - 1);
                  v85 = 0;
                  v86 = v8[18];
                  v87 = 1;
                  v5 = (unsigned int)v6 & v108;
                  v9 = (unsigned int)(*((_DWORD *)v8 + 38) + 1);
                  v62 = (int *)(v86 + 4 * v5);
                  v7 = (unsigned int)*v62;
                  if ( v12 != (_DWORD)v7 )
                  {
                    while ( (_DWORD)v7 != -1 )
                    {
                      if ( !v85 && (_DWORD)v7 == -2 )
                        v85 = v62;
                      v5 = (unsigned int)v6 & ((_DWORD)v5 + v87);
                      v62 = (int *)(v86 + 4 * v5);
                      v7 = (unsigned int)*v62;
                      if ( (_DWORD)v7 == v12 )
                        goto LABEL_98;
                      ++v87;
                    }
                    goto LABEL_153;
                  }
                }
                goto LABEL_98;
              }
            }
            else
            {
              ++v8[17];
            }
            sub_2E29BA0(v115, 2 * v61);
            v88 = *((_DWORD *)v8 + 40);
            if ( !v88 )
            {
LABEL_232:
              ++*((_DWORD *)v8 + 38);
              BUG();
            }
            v6 = (unsigned int)(v88 - 1);
            v7 = v8[18];
            v5 = (unsigned int)v6 & (37 * v12);
            v9 = (unsigned int)(*((_DWORD *)v8 + 38) + 1);
            v62 = (int *)(v7 + 4 * v5);
            v89 = *v62;
            if ( *v62 != v12 )
            {
              v90 = 1;
              v85 = 0;
              while ( v89 != -1 )
              {
                if ( !v85 && v89 == -2 )
                  v85 = v62;
                v5 = (unsigned int)v6 & ((_DWORD)v5 + v90);
                v62 = (int *)(v7 + 4 * v5);
                v89 = *v62;
                if ( *v62 == v12 )
                  goto LABEL_98;
                ++v90;
              }
LABEL_153:
              if ( v85 )
                v62 = v85;
            }
LABEL_98:
            *((_DWORD *)v8 + 38) = v9;
            if ( *v62 != -1 )
              --*((_DWORD *)v8 + 39);
            *v62 = v12;
            v14 = 1;
          }
        }
LABEL_16:
        v13 += 4;
        if ( v13 == v11 )
          break;
        while ( (unsigned int)*v13 > 0xFFFFFFFD )
        {
          v13 += 4;
          if ( v11 == v13 )
            goto LABEL_19;
        }
        if ( v11 == v13 )
          break;
        v12 = *v13;
      }
LABEL_19:
      if ( !v14 )
        goto LABEL_5;
      if ( !v132 )
        goto LABEL_103;
      v18 = v129;
      v9 = HIDWORD(v130);
      v5 = (__int64)&v129[HIDWORD(v130)];
      if ( v129 != (__int64 *)v5 )
      {
        while ( v127 != *v18 )
        {
          if ( (__int64 *)v5 == ++v18 )
            goto LABEL_24;
        }
        goto LABEL_5;
      }
LABEL_24:
      if ( HIDWORD(v130) >= (unsigned int)v130 )
      {
LABEL_103:
        sub_C8CC70((__int64)&v128, v127, v5, v9, v6, v7);
        goto LABEL_5;
      }
      ++v4;
      ++HIDWORD(v130);
      *(_QWORD *)v5 = v127;
      ++v128;
    }
    while ( v119 != v4 );
LABEL_26:
    v19 = sub_2E311E0(v113);
    v20 = *(_QWORD *)(v113 + 56);
    v122 = v19;
    if ( v20 == v19 )
      goto LABEL_41;
    while ( 2 )
    {
      v21 = 1;
      if ( (*(_DWORD *)(v20 + 40) & 0xFFFFFF) == 1 )
        goto LABEL_39;
      v22 = v20;
      v23 = *(_DWORD *)(v20 + 40) & 0xFFFFFF;
      while ( 2 )
      {
        v24 = *(_QWORD *)(v22 + 32);
        v25 = v24 + 40LL * v21;
        if ( !*(_BYTE *)v25
          && (*(_BYTE *)(v25 + 4) & 1) == 0
          && (*(_BYTE *)(v25 + 4) & 2) == 0
          && ((*(_BYTE *)(v25 + 3) & 0x10) == 0 || (*(_DWORD *)v25 & 0xFFF00) != 0) )
        {
          v26 = *(_DWORD *)(v25 + 8);
          v127 = *(_QWORD *)(v24 + 40LL * (v21 + 1) + 24);
          v27 = sub_2EEFC50(v124, &v127);
          v28 = v27;
          if ( v26 < 0 )
          {
            v29 = v27[10];
            v30 = *((_DWORD *)v27 + 24);
            if ( v30 )
            {
              v31 = v30 - 1;
              v32 = (v30 - 1) & (37 * v26);
              v33 = *(_DWORD *)(v29 + 4LL * v32);
              if ( v26 == v33 )
                goto LABEL_37;
              v71 = 1;
              while ( v33 != -1 )
              {
                v32 = v31 & (v71 + v32);
                v33 = *(_DWORD *)(v29 + 4LL * v32);
                if ( v26 == v33 )
                  goto LABEL_37;
                ++v71;
              }
            }
            v72 = *((_DWORD *)v28 + 40);
            v73 = (__int64)(v28 + 17);
            if ( !v72 )
            {
              ++v28[17];
              goto LABEL_181;
            }
            v74 = (unsigned int)(v72 - 1);
            v75 = v28[18];
            v76 = v74 & (37 * v26);
            v77 = (int *)(v75 + 4LL * v76);
            v78 = *v77;
            if ( v26 != *v77 )
            {
              v117 = 1;
              v79 = 0;
              v112 = *((_DWORD *)v28 + 40);
              while ( v78 != -1 )
              {
                if ( !v79 && v78 == -2 )
                  v79 = v77;
                v76 = v74 & (v117 + v76);
                v77 = (int *)(v75 + 4LL * v76);
                v78 = *v77;
                if ( v26 == *v77 )
                  goto LABEL_37;
                ++v117;
              }
              if ( v79 )
                v77 = v79;
              v80 = *((unsigned int *)v28 + 38);
              ++v28[17];
              v81 = (unsigned int)(v80 + 1);
              if ( 4 * (int)v81 < 3 * v112 )
              {
                if ( v112 - *((_DWORD *)v28 + 39) - (unsigned int)v81 <= v112 >> 3 )
                {
                  v118 = 37 * v26;
                  sub_2E29BA0((__int64)(v28 + 17), v112);
                  v103 = *((_DWORD *)v28 + 40);
                  if ( !v103 )
                  {
LABEL_233:
                    ++*((_DWORD *)v28 + 38);
                    BUG();
                  }
                  v74 = (unsigned int)(v103 - 1);
                  v80 = v28[18];
                  v73 = 1;
                  v104 = v74 & v118;
                  v81 = (unsigned int)(*((_DWORD *)v28 + 38) + 1);
                  v102 = 0;
                  v77 = (int *)(v80 + 4LL * ((unsigned int)v74 & v118));
                  v105 = *v77;
                  if ( v26 != *v77 )
                  {
                    while ( v105 != -1 )
                    {
                      if ( v105 == -2 && !v102 )
                        v102 = v77;
                      v106 = v73 + 1;
                      v73 = (unsigned int)(v104 + v73);
                      v104 = v74 & v73;
                      v77 = (int *)(v80 + 4LL * ((unsigned int)v74 & (unsigned int)v73));
                      v105 = *v77;
                      if ( v26 == *v77 )
                        goto LABEL_132;
                      v73 = v106;
                    }
                    goto LABEL_185;
                  }
                }
                goto LABEL_132;
              }
LABEL_181:
              sub_2E29BA0((__int64)(v28 + 17), 2 * v72);
              v99 = *((_DWORD *)v28 + 40);
              if ( !v99 )
                goto LABEL_233;
              v74 = (unsigned int)(v99 - 1);
              v80 = v28[18];
              v100 = v74 & (37 * v26);
              v81 = (unsigned int)(*((_DWORD *)v28 + 38) + 1);
              v77 = (int *)(v80 + 4LL * v100);
              v101 = *v77;
              if ( v26 != *v77 )
              {
                v73 = 1;
                v102 = 0;
                while ( v101 != -1 )
                {
                  if ( !v102 && v101 == -2 )
                    v102 = v77;
                  v107 = v73 + 1;
                  v73 = v100 + (unsigned int)v73;
                  v100 = v74 & v73;
                  v77 = (int *)(v80 + 4LL * ((unsigned int)v74 & (unsigned int)v73));
                  v101 = *v77;
                  if ( v26 == *v77 )
                    goto LABEL_132;
                  v73 = v107;
                }
LABEL_185:
                if ( v102 )
                  v77 = v102;
              }
LABEL_132:
              *((_DWORD *)v28 + 38) = v81;
              if ( *v77 != -1 )
                --*((_DWORD *)v28 + 39);
              *v77 = v26;
              if ( v132 )
              {
                v82 = v129;
                v81 = (__int64)&v129[HIDWORD(v130)];
                if ( v129 != (__int64 *)v81 )
                {
                  while ( v127 != *v82 )
                  {
                    if ( (__int64 *)v81 == ++v82 )
                      goto LABEL_138;
                  }
                  goto LABEL_37;
                }
LABEL_138:
                if ( HIDWORD(v130) < (unsigned int)v130 )
                {
                  ++HIDWORD(v130);
                  *(_QWORD *)v81 = v127;
                  ++v128;
                  goto LABEL_37;
                }
              }
              sub_C8CC70((__int64)&v128, v127, v81, v80, v74, v73);
            }
          }
        }
LABEL_37:
        v21 += 2;
        if ( v21 != v23 )
          continue;
        break;
      }
      v20 = v22;
LABEL_39:
      if ( (*(_BYTE *)v20 & 4) == 0 )
      {
        while ( (*(_BYTE *)(v20 + 44) & 8) != 0 )
          v20 = *(_QWORD *)(v20 + 8);
      }
      v20 = *(_QWORD *)(v20 + 8);
      if ( v122 != v20 )
        continue;
      break;
    }
LABEL_41:
    v113 = *(_QWORD *)(v113 + 8);
  }
  while ( v110 != v113 );
  v34 = HIDWORD(v130);
  if ( HIDWORD(v130) != v131 )
  {
    v120 = v124;
    do
    {
      if ( v132 )
      {
        v35 = v34;
        v36 = *v129;
        v37 = &v129[v34];
        if ( v129 == v37 )
        {
          v126 = *v129;
          goto LABEL_52;
        }
      }
      else
      {
        v36 = *v129;
        v37 = &v129[(unsigned int)v130];
        if ( v129 == v37 )
        {
          v126 = *v129;
LABEL_148:
          v83 = sub_C8CA60((__int64)&v128, v36);
          if ( v83 )
          {
            *v83 = -2;
            ++v131;
            ++v128;
          }
          goto LABEL_57;
        }
      }
      v38 = v129;
      do
      {
        v36 = *v38;
        v39 = v38;
        if ( (unsigned __int64)*v38 < 0xFFFFFFFFFFFFFFFELL )
          goto LABEL_50;
        ++v38;
      }
      while ( v37 != v38 );
      v36 = v39[1];
LABEL_50:
      v126 = v36;
      if ( !v132 )
        goto LABEL_148;
      v35 = v34;
LABEL_52:
      v40 = &v129[v35];
      v41 = v129;
      if ( v129 != v40 )
      {
        while ( *v41 != v36 )
        {
          if ( v40 == ++v41 )
            goto LABEL_57;
        }
        v42 = v34 - 1;
        HIDWORD(v130) = v42;
        *v41 = v129[v42];
        ++v128;
      }
LABEL_57:
      v43 = sub_2EEFC50(v120, &v126);
      v44 = v126;
      v123 = v43;
      v45 = *(__int64 **)(v126 + 64);
      v125 = &v45[*(unsigned int *)(v126 + 72)];
      if ( v45 == v125 )
        goto LABEL_82;
      v46 = *(__int64 **)(v126 + 64);
      while ( 1 )
      {
        v127 = *v46;
        if ( v127 != v44 )
        {
          v50 = sub_2EEFC50(v120, &v127);
          v51 = *((unsigned int *)v123 + 38);
          if ( (_DWORD)v51 )
          {
            v52 = (int *)v123[18];
            v53 = &v52[*((unsigned int *)v123 + 40)];
            if ( v52 != v53 )
            {
              while ( (unsigned int)*v52 > 0xFFFFFFFD )
              {
                if ( v53 == ++v52 )
                  goto LABEL_59;
              }
              if ( v52 != v53 )
                break;
            }
          }
        }
LABEL_59:
        if ( v125 == ++v46 )
          goto LABEL_82;
LABEL_60:
        v44 = v126;
      }
      v54 = 0;
      v116 = (__int64)(v50 + 17);
      do
      {
        v55 = *v52;
        if ( *v52 < 0 )
        {
          v56 = *((_DWORD *)v50 + 24);
          v47 = v50[10];
          if ( v56 )
          {
            v51 = (unsigned int)(v56 - 1);
            v57 = v51 & (37 * v55);
            v58 = *(_DWORD *)(v47 + 4LL * v57);
            if ( v58 == v55 )
              goto LABEL_72;
            v66 = 1;
            while ( v58 != -1 )
            {
              v48 = (unsigned int)(v66 + 1);
              v57 = v51 & (v66 + v57);
              v58 = *(_DWORD *)(v47 + 4LL * v57);
              if ( v55 == v58 )
                goto LABEL_72;
              ++v66;
            }
          }
          v67 = *((_DWORD *)v50 + 40);
          if ( !v67 )
          {
            ++v50[17];
            goto LABEL_167;
          }
          v49 = (unsigned int)(v67 - 1);
          v48 = v50[18];
          v51 = (unsigned int)v49 & (37 * v55);
          v47 = v48 + 4 * v51;
          v68 = *(_DWORD *)v47;
          if ( *(_DWORD *)v47 != v55 )
          {
            v114 = 1;
            v69 = 0;
            v109 = *((_DWORD *)v50 + 40);
            while ( v68 != -1 )
            {
              if ( !v69 && v68 == -2 )
                v69 = (_DWORD *)v47;
              v51 = (unsigned int)v49 & (v114 + (_DWORD)v51);
              v47 = v48 + 4LL * (unsigned int)v51;
              v68 = *(_DWORD *)v47;
              if ( v55 == *(_DWORD *)v47 )
                goto LABEL_72;
              ++v114;
            }
            if ( !v69 )
              v69 = (_DWORD *)v47;
            v70 = *((_DWORD *)v50 + 38);
            ++v50[17];
            v51 = (unsigned int)(v70 + 1);
            if ( 4 * (int)v51 < 3 * v109 )
            {
              v47 = v109 - *((_DWORD *)v50 + 39) - (unsigned int)v51;
              if ( (unsigned int)v47 <= v109 >> 3 )
              {
                sub_2E29BA0(v116, v109);
                v95 = *((_DWORD *)v50 + 40);
                if ( !v95 )
                {
LABEL_234:
                  ++*((_DWORD *)v50 + 38);
                  BUG();
                }
                v48 = (unsigned int)(v95 - 1);
                v49 = v50[18];
                v96 = 1;
                v51 = (unsigned int)(*((_DWORD *)v50 + 38) + 1);
                v47 = 0;
                v97 = (unsigned int)v48 & (37 * v55);
                v69 = (_DWORD *)(v49 + 4 * v97);
                v98 = *v69;
                if ( *v69 != v55 )
                {
                  while ( v98 != -1 )
                  {
                    if ( v98 == -2 && !v47 )
                      v47 = (__int64)v69;
                    v97 = (unsigned int)v48 & ((_DWORD)v97 + v96);
                    v69 = (_DWORD *)(v49 + 4 * v97);
                    v98 = *v69;
                    if ( v55 == *v69 )
                      goto LABEL_119;
                    ++v96;
                  }
                  if ( v47 )
                    v69 = (_DWORD *)v47;
                }
              }
              goto LABEL_119;
            }
LABEL_167:
            sub_2E29BA0(v116, 2 * v67);
            v91 = *((_DWORD *)v50 + 40);
            if ( !v91 )
              goto LABEL_234;
            v48 = (unsigned int)(v91 - 1);
            v49 = v50[18];
            v47 = (unsigned int)v48 & (37 * v55);
            v51 = (unsigned int)(*((_DWORD *)v50 + 38) + 1);
            v69 = (_DWORD *)(v49 + 4 * v47);
            v92 = *v69;
            if ( v55 != *v69 )
            {
              v93 = 1;
              v94 = 0;
              while ( v92 != -1 )
              {
                if ( v92 == -2 && !v94 )
                  v94 = v69;
                v47 = (unsigned int)v48 & ((_DWORD)v47 + v93);
                v69 = (_DWORD *)(v49 + 4 * v47);
                v92 = *v69;
                if ( v55 == *v69 )
                  goto LABEL_119;
                ++v93;
              }
              if ( v94 )
                v69 = v94;
            }
LABEL_119:
            *((_DWORD *)v50 + 38) = v51;
            if ( *v69 != -1 )
              --*((_DWORD *)v50 + 39);
            *v69 = v55;
            v54 = 1;
          }
        }
LABEL_72:
        if ( ++v52 == v53 )
          break;
        while ( (unsigned int)*v52 > 0xFFFFFFFD )
        {
          if ( v53 == ++v52 )
            goto LABEL_75;
        }
      }
      while ( v53 != v52 );
LABEL_75:
      if ( !v54 )
        goto LABEL_59;
      if ( !v132 )
        goto LABEL_145;
      v59 = v129;
      v47 = HIDWORD(v130);
      v51 = (__int64)&v129[HIDWORD(v130)];
      if ( v129 != (__int64 *)v51 )
      {
        while ( v127 != *v59 )
        {
          if ( (__int64 *)v51 == ++v59 )
            goto LABEL_80;
        }
        goto LABEL_59;
      }
LABEL_80:
      if ( HIDWORD(v130) >= (unsigned int)v130 )
      {
LABEL_145:
        sub_C8CC70((__int64)&v128, v127, v51, v47, v48, v49);
        goto LABEL_59;
      }
      ++v46;
      ++HIDWORD(v130);
      *(_QWORD *)v51 = v127;
      ++v128;
      if ( v125 != v46 )
        goto LABEL_60;
LABEL_82:
      v34 = HIDWORD(v130);
    }
    while ( HIDWORD(v130) != v131 );
  }
  if ( !v132 )
    _libc_free((unsigned __int64)v129);
}
