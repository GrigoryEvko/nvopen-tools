// Function: sub_E3C3B0
// Address: 0xe3c3b0
//
__int64 __fastcall sub_E3C3B0(__int64 *a1, __int64 a2)
{
  __int64 v3; // r9
  __int64 v4; // rcx
  __int64 v5; // rax
  __int64 v6; // r8
  __int64 v7; // rcx
  __int64 v8; // r13
  int v9; // eax
  unsigned int v10; // edx
  __int64 v11; // rsi
  unsigned int v12; // edx
  __int64 v13; // rbx
  __int64 v14; // r12
  int v15; // eax
  unsigned int v16; // edi
  __int64 *v17; // rax
  __int64 v18; // rbx
  __int64 v19; // r8
  __int64 v20; // r9
  __int64 v21; // rax
  __int64 v22; // rbx
  __int64 v23; // rax
  __int64 v24; // r12
  unsigned int v25; // esi
  unsigned int v26; // edx
  __int64 *v27; // rax
  __int64 v28; // rcx
  unsigned int v29; // eax
  __int64 v30; // rcx
  __int64 v31; // rsi
  __int64 *v32; // rax
  __int64 *v33; // rbx
  __int64 v34; // rdx
  __int64 v35; // rcx
  __int64 v36; // r8
  __int64 *v37; // r12
  __int64 *v38; // rbx
  __int64 v39; // rsi
  unsigned int v40; // r10d
  unsigned int v41; // esi
  __int64 v42; // rdx
  __int64 v43; // r8
  unsigned int v44; // ecx
  __int64 *v45; // rax
  __int64 v46; // rdi
  __int64 v47; // rdx
  __int64 v48; // rcx
  __int64 v49; // r8
  __int64 v50; // r9
  unsigned int v51; // esi
  __int64 v52; // rdx
  __int64 v53; // r11
  unsigned int v54; // ecx
  __int64 *v55; // rax
  __int64 v56; // rdi
  __int64 *v57; // r8
  int v58; // ecx
  int v59; // ecx
  __int64 result; // rax
  _QWORD **v61; // r12
  _QWORD **j; // rbx
  _QWORD *v63; // rdi
  int v64; // esi
  __int64 v65; // rdi
  unsigned __int64 v66; // rdx
  int v67; // r10d
  __int64 *v68; // r9
  int v69; // ecx
  int v70; // ecx
  int v71; // r10d
  __int64 *v72; // rdi
  int v73; // ecx
  int v74; // edx
  int v75; // r10d
  int v76; // r10d
  int v77; // esi
  __int64 v78; // rbx
  __int64 *v79; // rcx
  __int64 v80; // rdi
  int v81; // r11d
  int v82; // r11d
  __int64 v83; // r10
  __int64 v84; // rcx
  __int64 v85; // r8
  int v86; // edi
  __int64 *v87; // rsi
  int v88; // eax
  __int64 v89; // r11
  unsigned int v90; // esi
  __int64 v91; // r10
  __int64 *v92; // rdi
  int v93; // r9d
  int v94; // eax
  __int64 v95; // r11
  unsigned int v96; // edi
  __int64 *v97; // rsi
  __int64 v98; // r10
  int v99; // r9d
  int v100; // r11d
  int v101; // r11d
  unsigned int v102; // esi
  __int64 v103; // r8
  int v104; // edi
  int v105; // ebx
  int v106; // ebx
  __int64 v107; // r10
  unsigned int v108; // edi
  __int64 *v109; // rsi
  int v110; // r8d
  int v111; // [rsp+4h] [rbp-FCh]
  int v112; // [rsp+4h] [rbp-FCh]
  int v113; // [rsp+4h] [rbp-FCh]
  int v114; // [rsp+4h] [rbp-FCh]
  __int64 v115; // [rsp+28h] [rbp-D8h]
  __int64 v116; // [rsp+28h] [rbp-D8h]
  __int64 v117; // [rsp+28h] [rbp-D8h]
  __int64 *v118; // [rsp+30h] [rbp-D0h]
  __int64 *i; // [rsp+38h] [rbp-C8h]
  __int64 *v120; // [rsp+48h] [rbp-B8h] BYREF
  __int64 v121; // [rsp+50h] [rbp-B0h] BYREF
  unsigned int v122; // [rsp+58h] [rbp-A8h] BYREF
  unsigned int v123; // [rsp+5Ch] [rbp-A4h]
  __int64 v124[4]; // [rsp+60h] [rbp-A0h] BYREF
  _BYTE *v125; // [rsp+80h] [rbp-80h] BYREF
  __int64 v126; // [rsp+88h] [rbp-78h]
  _BYTE v127[112]; // [rsp+90h] [rbp-70h] BYREF

  sub_E3A5B0((__int64)a1, a2);
  v4 = a1[5];
  v125 = v127;
  v126 = 0x800000000LL;
  v118 = (__int64 *)v4;
  v5 = v4 + 8LL * *((unsigned int *)a1 + 12);
  if ( v4 == v5 )
  {
    result = *a1;
    v61 = *(_QWORD ***)(*a1 + 80);
    j = *(_QWORD ***)(*a1 + 72);
    if ( j != v61 )
      goto LABEL_59;
    return result;
  }
  v6 = 0;
  for ( i = (__int64 *)(v5 - 8); ; i = v17 - 1 )
  {
    v7 = a1[2];
    v8 = *i;
    v9 = *((_DWORD *)a1 + 8);
    if ( v9 )
    {
      v10 = (v9 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
      v11 = v7 + 16LL * v10;
      v3 = *(_QWORD *)v11;
      if ( v8 == *(_QWORD *)v11 )
      {
LABEL_5:
        v12 = *(_DWORD *)(v11 + 8);
        a2 = *(unsigned int *)(v11 + 12);
        goto LABEL_6;
      }
      v64 = 1;
      while ( v3 != -4096 )
      {
        v67 = v64 + 1;
        v10 = (v9 - 1) & (v64 + v10);
        v11 = v7 + 16LL * v10;
        v3 = *(_QWORD *)v11;
        if ( v8 == *(_QWORD *)v11 )
          goto LABEL_5;
        v64 = v67;
      }
    }
    a2 = 0;
    v12 = 0;
LABEL_6:
    v13 = *(_QWORD *)(v8 + 16);
    v122 = v12;
    v123 = a2;
    if ( !v13 )
    {
LABEL_16:
      if ( (_DWORD)v6 )
        goto LABEL_21;
      goto LABEL_17;
    }
    do
    {
      a2 = *(_QWORD *)(v13 + 24);
      if ( (unsigned __int8)(*(_BYTE *)a2 - 30) <= 0xAu )
      {
LABEL_11:
        v14 = *(_QWORD *)(a2 + 40);
        if ( v9 )
        {
          v15 = v9 - 1;
          v16 = v15 & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
          a2 = v7 + 16LL * v16;
          v3 = *(_QWORD *)a2;
          if ( v14 == *(_QWORD *)a2 )
          {
LABEL_13:
            if ( v12 > *(_DWORD *)(a2 + 8) || v123 < *(_DWORD *)(a2 + 12) )
              goto LABEL_15;
LABEL_37:
            if ( v6 + 1 > (unsigned __int64)HIDWORD(v126) )
            {
              a2 = (__int64)v127;
              sub_C8D5F0((__int64)&v125, v127, v6 + 1, 8u, v6, v3);
              v6 = (unsigned int)v126;
            }
            *(_QWORD *)&v125[8 * v6] = v14;
            v6 = (unsigned int)(v126 + 1);
            LODWORD(v126) = v126 + 1;
LABEL_15:
            while ( 1 )
            {
              v13 = *(_QWORD *)(v13 + 8);
              if ( !v13 )
                goto LABEL_16;
              a2 = *(_QWORD *)(v13 + 24);
              if ( (unsigned __int8)(*(_BYTE *)a2 - 30) <= 0xAu )
              {
                v12 = v122;
                v7 = a1[2];
                v9 = *((_DWORD *)a1 + 8);
                goto LABEL_11;
              }
            }
          }
          a2 = 1;
          while ( v3 != -4096 )
          {
            v40 = a2 + 1;
            v16 = v15 & (a2 + v16);
            a2 = v7 + 16LL * v16;
            v3 = *(_QWORD *)a2;
            if ( v14 == *(_QWORD *)a2 )
              goto LABEL_13;
            a2 = v40;
          }
        }
        if ( v12 )
          goto LABEL_15;
        goto LABEL_37;
      }
      v13 = *(_QWORD *)(v13 + 8);
    }
    while ( v13 );
    if ( !(_DWORD)v6 )
      goto LABEL_17;
LABEL_21:
    v18 = sub_22077B0(224);
    if ( v18 )
    {
      v120 = (__int64 *)v18;
      memset((void *)v18, 0, 0xE0u);
      *(_DWORD *)(v18 + 188) = 4;
      *(_QWORD *)(v18 + 8) = v18 + 24;
      *(_QWORD *)(v18 + 16) = 0x100000000LL;
      *(_QWORD *)(v18 + 88) = v18 + 104;
      *(_QWORD *)(v18 + 96) = 0x800000000LL;
      *(_QWORD *)(v18 + 176) = v18 + 192;
      v21 = 0;
    }
    else
    {
      v21 = MEMORY[0x10];
      v120 = 0;
      v66 = MEMORY[0x10] + 1LL;
      if ( MEMORY[0x14] < v66 )
      {
        sub_C8D5F0(8, (const void *)0x18, v66, 8u, v19, v20);
        v21 = MEMORY[0x10];
      }
    }
    *(_QWORD *)(*(_QWORD *)(v18 + 8) + 8 * v21) = v8;
    ++*(_DWORD *)(v18 + 16);
    *(_DWORD *)(v18 + 184) = 0;
    v22 = (__int64)v120;
    v124[0] = v8;
    sub_E3B670((__int64)(v120 + 7), v124);
    v23 = (__int64)v120;
    *(_DWORD *)(v22 + 184) = 0;
    v24 = *a1;
    v115 = v23;
    v25 = *(_DWORD *)(*a1 + 32);
    if ( !v25 )
    {
      ++*(_QWORD *)(v24 + 8);
LABEL_101:
      sub_E39660(v24 + 8, 2 * v25);
      v81 = *(_DWORD *)(v24 + 32);
      if ( v81 )
      {
        v82 = v81 - 1;
        v83 = *(_QWORD *)(v24 + 16);
        LODWORD(v84) = v82 & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
        v74 = *(_DWORD *)(v24 + 24) + 1;
        v27 = (__int64 *)(v83 + 16LL * (unsigned int)v84);
        v85 = *v27;
        if ( v8 != *v27 )
        {
          v86 = 1;
          v87 = 0;
          while ( v85 != -4096 )
          {
            if ( v85 == -8192 && !v87 )
              v87 = v27;
            v3 = (unsigned int)(v86 + 1);
            v84 = v82 & (unsigned int)(v84 + v86);
            v27 = (__int64 *)(v83 + 16 * v84);
            v85 = *v27;
            if ( v8 == *v27 )
              goto LABEL_91;
            ++v86;
          }
          if ( v87 )
            v27 = v87;
        }
        goto LABEL_91;
      }
LABEL_186:
      ++*(_DWORD *)(v24 + 24);
      BUG();
    }
    v3 = *(_QWORD *)(v24 + 16);
    v26 = (v25 - 1) & (((unsigned int)v8 >> 4) ^ ((unsigned int)v8 >> 9));
    v27 = (__int64 *)(v3 + 16LL * v26);
    v28 = *v27;
    if ( v8 == *v27 )
      goto LABEL_25;
    v71 = 1;
    v72 = 0;
    while ( v28 != -4096 )
    {
      if ( !v72 && v28 == -8192 )
        v72 = v27;
      v26 = (v25 - 1) & (v71 + v26);
      v27 = (__int64 *)(v3 + 16LL * v26);
      v28 = *v27;
      if ( v8 == *v27 )
        goto LABEL_25;
      ++v71;
    }
    v73 = *(_DWORD *)(v24 + 24);
    if ( v72 )
      v27 = v72;
    ++*(_QWORD *)(v24 + 8);
    v74 = v73 + 1;
    if ( 4 * (v73 + 1) >= 3 * v25 )
      goto LABEL_101;
    if ( v25 - *(_DWORD *)(v24 + 28) - v74 <= v25 >> 3 )
    {
      sub_E39660(v24 + 8, v25);
      v75 = *(_DWORD *)(v24 + 32);
      if ( v75 )
      {
        v76 = v75 - 1;
        v3 = *(_QWORD *)(v24 + 16);
        v77 = 1;
        LODWORD(v78) = v76 & (((unsigned int)v8 >> 4) ^ ((unsigned int)v8 >> 9));
        v74 = *(_DWORD *)(v24 + 24) + 1;
        v79 = 0;
        v27 = (__int64 *)(v3 + 16LL * (unsigned int)v78);
        v80 = *v27;
        if ( v8 != *v27 )
        {
          while ( v80 != -4096 )
          {
            if ( v80 == -8192 && !v79 )
              v79 = v27;
            v78 = v76 & (unsigned int)(v78 + v77);
            v27 = (__int64 *)(v3 + 16 * v78);
            v80 = *v27;
            if ( v8 == *v27 )
              goto LABEL_91;
            ++v77;
          }
          if ( v79 )
            v27 = v79;
        }
        goto LABEL_91;
      }
      goto LABEL_186;
    }
LABEL_91:
    *(_DWORD *)(v24 + 24) = v74;
    if ( *v27 != -4096 )
      --*(_DWORD *)(v24 + 28);
    *v27 = v8;
    v27[1] = v115;
    v24 = *a1;
LABEL_25:
    v124[0] = (__int64)a1;
    v124[1] = (__int64)&v122;
    v124[2] = (__int64)&v125;
    v124[3] = (__int64)&v120;
    v29 = v126;
    while ( 2 )
    {
      v30 = v29--;
      v31 = *(_QWORD *)&v125[8 * v30 - 8];
      LODWORD(v126) = v29;
      v121 = v31;
      if ( v8 != v31 )
      {
        v32 = sub_E39300(v24, v31);
        v33 = v32;
        if ( v32 )
        {
          v24 = *a1;
          if ( v32 != v120 )
          {
            sub_E3C0A0(*a1, (__int64)v120, (__int64)v32);
            v37 = (__int64 *)v33[1];
            v38 = &v37[*((unsigned int *)v33 + 4)];
            while ( v38 != v37 )
            {
              v39 = *v37++;
              sub_E38CF0((__int64)v124, v39, v34, v35, v36, v3);
            }
            v24 = *a1;
          }
          goto LABEL_26;
        }
        v24 = *a1;
        v41 = *(_DWORD *)(*a1 + 32);
        v116 = (__int64)v120;
        if ( v41 )
        {
          v42 = v121;
          v43 = *(_QWORD *)(v24 + 16);
          v44 = (v41 - 1) & (((unsigned int)v121 >> 9) ^ ((unsigned int)v121 >> 4));
          v45 = (__int64 *)(v43 + 16LL * v44);
          v46 = *v45;
          if ( *v45 == v121 )
            goto LABEL_44;
          v112 = 1;
          v68 = 0;
          while ( v46 != -4096 )
          {
            if ( !v68 && v46 == -8192 )
              v68 = v45;
            v44 = (v41 - 1) & (v112 + v44);
            v45 = (__int64 *)(v43 + 16LL * v44);
            v46 = *v45;
            if ( v121 == *v45 )
              goto LABEL_44;
            ++v112;
          }
          v69 = *(_DWORD *)(v24 + 24);
          if ( v68 )
            v45 = v68;
          ++*(_QWORD *)(v24 + 8);
          v70 = v69 + 1;
          if ( 4 * v70 < 3 * v41 )
          {
            if ( v41 - *(_DWORD *)(v24 + 28) - v70 <= v41 >> 3 )
            {
              sub_E39660(v24 + 8, v41);
              v94 = *(_DWORD *)(v24 + 32);
              if ( !v94 )
                goto LABEL_186;
              v42 = v121;
              v95 = *(_QWORD *)(v24 + 16);
              v114 = v94 - 1;
              v96 = (v94 - 1) & (((unsigned int)v121 >> 9) ^ ((unsigned int)v121 >> 4));
              v70 = *(_DWORD *)(v24 + 24) + 1;
              v97 = (__int64 *)(v95 + 16LL * v96);
              v98 = *v97;
              v45 = v97;
              if ( *v97 != v121 )
              {
                v45 = 0;
                v99 = 1;
                while ( v98 != -4096 )
                {
                  if ( !v45 && v98 == -8192 )
                    v45 = v97;
                  v96 = v114 & (v99 + v96);
                  v97 = (__int64 *)(v95 + 16LL * v96);
                  v98 = *v97;
                  if ( v121 == *v97 )
                  {
                    v45 = (__int64 *)(v95 + 16LL * v96);
                    goto LABEL_82;
                  }
                  ++v99;
                }
                if ( !v45 )
                  v45 = v97;
              }
            }
            goto LABEL_82;
          }
        }
        else
        {
          ++*(_QWORD *)(v24 + 8);
        }
        sub_E39660(v24 + 8, 2 * v41);
        v88 = *(_DWORD *)(v24 + 32);
        if ( !v88 )
          goto LABEL_186;
        v42 = v121;
        v89 = *(_QWORD *)(v24 + 16);
        v113 = v88 - 1;
        v90 = (v88 - 1) & (((unsigned int)v121 >> 9) ^ ((unsigned int)v121 >> 4));
        v70 = *(_DWORD *)(v24 + 24) + 1;
        v45 = (__int64 *)(v89 + 16LL * v90);
        v91 = *v45;
        if ( v121 != *v45 )
        {
          v92 = 0;
          v93 = 1;
          while ( v91 != -4096 )
          {
            if ( v91 == -8192 && !v92 )
              v92 = v45;
            v90 = v113 & (v93 + v90);
            v45 = (__int64 *)(v89 + 16LL * v90);
            v91 = *v45;
            if ( v121 == *v45 )
              goto LABEL_82;
            ++v93;
          }
          if ( v92 )
            v45 = v92;
        }
LABEL_82:
        *(_DWORD *)(v24 + 24) = v70;
        if ( *v45 != -4096 )
          --*(_DWORD *)(v24 + 28);
        *v45 = v42;
        v45[1] = v116;
        v116 = (__int64)v120;
LABEL_44:
        sub_E3B670(v116 + 56, &v121);
        sub_E38CF0((__int64)v124, v121, v47, v48, v49, v50);
        v24 = *a1;
        v51 = *(_DWORD *)(*a1 + 64);
        v117 = (__int64)v120;
        v3 = *a1 + 40;
        if ( v51 )
        {
          v52 = v121;
          v53 = *(_QWORD *)(v24 + 48);
          v54 = (v51 - 1) & (((unsigned int)v121 >> 9) ^ ((unsigned int)v121 >> 4));
          v55 = (__int64 *)(v53 + 16LL * v54);
          v56 = *v55;
          if ( *v55 == v121 )
            goto LABEL_26;
          v111 = 1;
          v57 = 0;
          while ( v56 != -4096 )
          {
            if ( !v57 && v56 == -8192 )
              v57 = v55;
            v54 = (v51 - 1) & (v111 + v54);
            v55 = (__int64 *)(v53 + 16LL * v54);
            v56 = *v55;
            if ( v121 == *v55 )
              goto LABEL_26;
            ++v111;
          }
          v58 = *(_DWORD *)(v24 + 56);
          if ( v57 )
            v55 = v57;
          ++*(_QWORD *)(v24 + 40);
          v59 = v58 + 1;
          if ( 4 * v59 < 3 * v51 )
          {
            if ( v51 - *(_DWORD *)(v24 + 60) - v59 <= v51 >> 3 )
            {
              sub_E39120(v24 + 40, v51);
              v105 = *(_DWORD *)(v24 + 64);
              if ( !v105 )
              {
LABEL_185:
                ++*(_DWORD *)(v24 + 56);
                BUG();
              }
              v52 = v121;
              v106 = v105 - 1;
              v107 = *(_QWORD *)(v24 + 48);
              v59 = *(_DWORD *)(v24 + 56) + 1;
              v108 = v106 & (((unsigned int)v121 >> 9) ^ ((unsigned int)v121 >> 4));
              v55 = (__int64 *)(v107 + 16LL * v108);
              v3 = *v55;
              if ( *v55 != v121 )
              {
                v109 = (__int64 *)(v107 + 16LL * (v106 & (((unsigned int)v121 >> 9) ^ ((unsigned int)v121 >> 4))));
                v110 = 1;
                v55 = 0;
                while ( v3 != -4096 )
                {
                  if ( !v55 && v3 == -8192 )
                    v55 = v109;
                  v108 = v106 & (v110 + v108);
                  v109 = (__int64 *)(v107 + 16LL * v108);
                  v3 = *v109;
                  if ( v121 == *v109 )
                  {
                    v55 = (__int64 *)(v107 + 16LL * v108);
                    goto LABEL_52;
                  }
                  ++v110;
                }
                if ( !v55 )
                  v55 = v109;
              }
            }
            goto LABEL_52;
          }
        }
        else
        {
          ++*(_QWORD *)(v24 + 40);
        }
        sub_E39120(v24 + 40, 2 * v51);
        v100 = *(_DWORD *)(v24 + 64);
        if ( !v100 )
          goto LABEL_185;
        v52 = v121;
        v101 = v100 - 1;
        v3 = *(_QWORD *)(v24 + 48);
        v59 = *(_DWORD *)(v24 + 56) + 1;
        v102 = v101 & (((unsigned int)v121 >> 9) ^ ((unsigned int)v121 >> 4));
        v55 = (__int64 *)(v3 + 16LL * v102);
        v103 = *v55;
        if ( *v55 != v121 )
        {
          v104 = 1;
          while ( v103 != -4096 )
          {
            if ( !v33 && v103 == -8192 )
              v33 = v55;
            v102 = v101 & (v104 + v102);
            v55 = (__int64 *)(v3 + 16LL * v102);
            v103 = *v55;
            if ( v121 == *v55 )
              goto LABEL_52;
            ++v104;
          }
          if ( v33 )
            v55 = v33;
        }
LABEL_52:
        *(_DWORD *)(v24 + 56) = v59;
        if ( *v55 != -4096 )
          --*(_DWORD *)(v24 + 60);
        *v55 = v52;
        v55[1] = v117;
        v24 = *a1;
LABEL_26:
        v29 = v126;
      }
      if ( v29 )
        continue;
      break;
    }
    a2 = *(_QWORD *)(v24 + 80);
    if ( a2 != *(_QWORD *)(v24 + 88) )
      break;
    sub_E38EC0((__int64 **)(v24 + 72), a2, (__int64 *)&v120);
    v65 = (__int64)v120;
LABEL_67:
    if ( v65 )
      sub_E38110(v65, a2);
LABEL_17:
    v17 = i;
    if ( v118 == i )
      goto LABEL_58;
LABEL_18:
    v6 = (unsigned int)v126;
  }
  if ( !a2 )
  {
    *(_QWORD *)(v24 + 80) = 8;
    v65 = (__int64)v120;
    goto LABEL_67;
  }
  *(_QWORD *)a2 = v120;
  v17 = i;
  *(_QWORD *)(v24 + 80) += 8LL;
  if ( v118 != i )
    goto LABEL_18;
LABEL_58:
  result = *a1;
  v61 = *(_QWORD ***)(*a1 + 80);
  for ( j = *(_QWORD ***)(*a1 + 72); v61 != j; result = sub_E3ACE0((__int64)v63) )
  {
LABEL_59:
    v63 = *j++;
    *v63 = 0;
  }
  if ( v125 != v127 )
    return _libc_free(v125, a2);
  return result;
}
