// Function: sub_3725D40
// Address: 0x3725d40
//
__int64 __fastcall sub_3725D40(_QWORD *a1, __int64 a2, __int64 a3, __int64 j, __int64 a5)
{
  int v6; // edi
  __int64 v7; // rdx
  __int64 *v8; // r15
  _QWORD *v9; // rbx
  __int64 v10; // r12
  __int64 v11; // r14
  __int64 v12; // rdx
  char v13; // al
  const char *v14; // rdx
  int v15; // ecx
  __int64 v16; // rbx
  __int64 v17; // r13
  unsigned int v18; // ebx
  int v19; // eax
  int v20; // r8d
  unsigned int m; // edx
  __int64 v22; // rax
  unsigned int v23; // edx
  unsigned __int8 v24; // al
  int v25; // edx
  const char *v26; // rcx
  unsigned int v27; // esi
  __int64 v28; // r13
  __int64 v29; // r8
  int v30; // r11d
  __int64 v31; // r9
  unsigned int v32; // ecx
  __int64 *v33; // rdx
  __int64 v34; // rax
  __int64 v35; // rbx
  _QWORD **v36; // rsi
  __int64 v37; // r14
  void (*v38)(); // r13
  const char *v39; // rax
  __int64 v40; // rdx
  unsigned int v41; // eax
  int v42; // eax
  int v43; // r11d
  unsigned int n; // ecx
  __int64 v45; // rdx
  unsigned int v46; // ecx
  int v47; // eax
  __int64 v48; // rdi
  __int64 v49; // r8
  void (*v50)(); // rcx
  __int64 *v51; // rdx
  __int64 v52; // rax
  _QWORD *v53; // rsi
  __int64 v54; // rdi
  __int64 v55; // rsi
  unsigned int v57; // edx
  __int64 v58; // rsi
  int v59; // r11d
  _QWORD *v60; // rdi
  int v61; // r10d
  _QWORD *v62; // rsi
  unsigned int v63; // ebx
  __int64 v64; // rcx
  __int64 v65; // r15
  const char *v66; // rbx
  int v67; // r12d
  char v68; // r13
  __int64 v69; // rdi
  __int64 v70; // rax
  unsigned int v71; // r9d
  int v72; // eax
  __int64 v73; // rsi
  int v74; // edi
  __int64 v75; // rax
  const char *v76; // r9
  int v77; // edx
  int v78; // ecx
  __int64 v79; // rbx
  unsigned int v80; // r12d
  int v81; // eax
  __int64 v82; // rcx
  int v83; // esi
  unsigned int i; // edx
  const char *v85; // rdi
  unsigned int v86; // edx
  int v87; // eax
  __int64 v88; // rsi
  int v89; // r9d
  unsigned int k; // edx
  const char *v91; // rdi
  unsigned int v92; // edx
  __int64 *v93; // [rsp+0h] [rbp-150h]
  __int64 *v94; // [rsp+8h] [rbp-148h]
  __int64 v95; // [rsp+10h] [rbp-140h]
  _QWORD *v96; // [rsp+18h] [rbp-138h]
  __int64 v97; // [rsp+20h] [rbp-130h]
  _QWORD *v98; // [rsp+38h] [rbp-118h]
  __int64 v99; // [rsp+40h] [rbp-110h]
  unsigned int v100; // [rsp+4Ch] [rbp-104h]
  unsigned int v101; // [rsp+4Ch] [rbp-104h]
  unsigned int v102; // [rsp+4Ch] [rbp-104h]
  unsigned int v103; // [rsp+50h] [rbp-100h]
  _QWORD *v104; // [rsp+58h] [rbp-F8h]
  __int64 v105; // [rsp+58h] [rbp-F8h]
  __int64 v106; // [rsp+58h] [rbp-F8h]
  __int64 v107; // [rsp+68h] [rbp-E8h]
  __int64 v108; // [rsp+68h] [rbp-E8h]
  int v109; // [rsp+74h] [rbp-DCh] BYREF
  const char *v110; // [rsp+78h] [rbp-D8h] BYREF
  _QWORD v111[2]; // [rsp+80h] [rbp-D0h] BYREF
  const char *v112; // [rsp+90h] [rbp-C0h] BYREF
  int v113; // [rsp+98h] [rbp-B8h]
  char v114[4]; // [rsp+9Ch] [rbp-B4h] BYREF
  char v115; // [rsp+A0h] [rbp-B0h]
  __int64 v116; // [rsp+B0h] [rbp-A0h] BYREF
  __int64 v117; // [rsp+B8h] [rbp-98h]
  __int64 v118; // [rsp+C0h] [rbp-90h]
  unsigned int v119; // [rsp+C8h] [rbp-88h]
  const char *v120; // [rsp+D0h] [rbp-80h] BYREF
  __int64 v121; // [rsp+D8h] [rbp-78h] BYREF
  __int64 v122; // [rsp+E0h] [rbp-70h]
  __int64 v123; // [rsp+E8h] [rbp-68h]
  const char *v124; // [rsp+F0h] [rbp-60h] BYREF
  _QWORD v125[3]; // [rsp+F8h] [rbp-58h] BYREF
  __int16 v126; // [rsp+110h] [rbp-40h]

  v6 = *((_DWORD *)a1 + 84);
  v116 = 0;
  v117 = 0;
  v118 = 0;
  v119 = 0;
  if ( v6 )
  {
    v65 = a1[41];
    v108 = v65 + 16LL * *((unsigned int *)a1 + 86);
    while ( v108 != v65 )
    {
      if ( *(_QWORD *)v65 == -1 )
      {
        if ( *(_DWORD *)(v65 + 8) != -1 )
          goto LABEL_88;
      }
      else if ( *(_QWORD *)v65 != -2 || *(_DWORD *)(v65 + 8) != -2 )
      {
LABEL_88:
        if ( v108 == v65 )
          break;
        while ( 1 )
        {
          v66 = *(const char **)v65;
          v67 = *(_DWORD *)(v65 + 8);
          v68 = *(_BYTE *)(v65 + 12);
          v69 = *a1;
          v126 = 257;
          v70 = sub_31DCC50(v69, (__int64 *)&v124, a3, j, a5);
          v71 = v119;
          v120 = v66;
          LODWORD(v121) = v67;
          BYTE4(v121) = v68;
          v122 = v70;
          if ( !v119 )
          {
            ++v116;
LABEL_100:
            sub_3725A80((__int64)&v116, 2 * v71);
            v75 = 0;
            if ( !v119 )
              goto LABEL_101;
            v79 = v117;
            v80 = v119 - 1;
            LODWORD(v111[0]) = v121;
            v112 = v120;
            v81 = sub_3723A60((__int64 *)&v112, v111, (_BYTE *)&v121 + 4);
            v82 = 0;
            a5 = BYTE4(v121);
            v83 = 1;
            for ( i = v80 & v81; ; i = v80 & v86 )
            {
              v75 = v79 + 24LL * i;
              v85 = *(const char **)v75;
              if ( v120 == *(const char **)v75
                && (_DWORD)v121 == *(_DWORD *)(v75 + 8)
                && BYTE4(v121) == *(_BYTE *)(v75 + 12) )
              {
                goto LABEL_101;
              }
              if ( v85 == (const char *)-1LL )
              {
                if ( *(_DWORD *)(v75 + 8) == -1 && !*(_BYTE *)(v75 + 12) )
                {
                  if ( v82 )
                    v75 = v82;
LABEL_101:
                  j = (unsigned int)v118;
                  v77 = v118 + 1;
                  goto LABEL_102;
                }
              }
              else if ( v85 == (const char *)-2LL && *(_DWORD *)(v75 + 8) == -2 && *(_BYTE *)(v75 + 12) != 1 && !v82 )
              {
                v82 = v79 + 24LL * i;
              }
              v86 = v83 + i;
              ++v83;
            }
          }
          LODWORD(v111[0]) = v67;
          v112 = v66;
          v101 = v119;
          v105 = v117;
          v72 = sub_3723A60((__int64 *)&v112, v111, (_BYTE *)&v121 + 4);
          a5 = (__int64)v120;
          v73 = 0;
          v74 = 1;
          a3 = v101 - 1;
          for ( j = (unsigned int)a3 & v72; ; j = (unsigned int)a3 & v78 )
          {
            v75 = v105 + 24LL * (unsigned int)j;
            v76 = *(const char **)v75;
            if ( v120 == *(const char **)v75 && (_DWORD)v121 == *(_DWORD *)(v75 + 8) )
              break;
            if ( v76 == (const char *)-1LL )
              goto LABEL_114;
LABEL_93:
            if ( v76 == (const char *)-2LL && *(_DWORD *)(v75 + 8) == -2 && *(_BYTE *)(v75 + 12) != 1 && !v73 )
              v73 = v105 + 24LL * (unsigned int)j;
LABEL_115:
            v78 = v74 + j;
            ++v74;
          }
          if ( BYTE4(v121) == *(_BYTE *)(v75 + 12) )
            goto LABEL_105;
          if ( v76 != (const char *)-1LL )
            goto LABEL_93;
LABEL_114:
          if ( *(_DWORD *)(v75 + 8) != -1 || *(_BYTE *)(v75 + 12) )
            goto LABEL_115;
          v71 = v119;
          if ( v73 )
            v75 = v73;
          ++v116;
          v77 = v118 + 1;
          if ( 4 * ((int)v118 + 1) >= 3 * v119 )
            goto LABEL_100;
          j = v119 >> 3;
          if ( v119 - (v77 + HIDWORD(v118)) <= (unsigned int)j )
          {
            sub_3725A80((__int64)&v116, v119);
            a5 = v119;
            v75 = 0;
            v102 = v119;
            if ( !v119 )
              goto LABEL_101;
            LODWORD(v111[0]) = v121;
            v106 = v117;
            v112 = v120;
            v87 = sub_3723A60((__int64 *)&v112, v111, (_BYTE *)&v121 + 4);
            v88 = 0;
            v89 = 1;
            a5 = v102 - 1;
            for ( k = a5 & v87; ; k = a5 & v92 )
            {
              v75 = v106 + 24LL * k;
              v91 = *(const char **)v75;
              if ( v120 == *(const char **)v75
                && (_DWORD)v121 == *(_DWORD *)(v75 + 8)
                && BYTE4(v121) == *(_BYTE *)(v75 + 12) )
              {
                goto LABEL_101;
              }
              if ( v91 == (const char *)-1LL )
              {
                if ( *(_DWORD *)(v75 + 8) == -1 && !*(_BYTE *)(v75 + 12) )
                {
                  if ( v88 )
                    v75 = v88;
                  goto LABEL_101;
                }
              }
              else if ( v91 == (const char *)-2LL && *(_DWORD *)(v75 + 8) == -2 && *(_BYTE *)(v75 + 12) != 1 && !v88 )
              {
                v88 = v106 + 24LL * k;
              }
              v92 = v89 + k;
              ++v89;
            }
          }
LABEL_102:
          LODWORD(v118) = v77;
          if ( *(_QWORD *)v75 != -1 || *(_DWORD *)(v75 + 8) != -1 || *(_BYTE *)(v75 + 12) )
            --HIDWORD(v118);
          *(_QWORD *)v75 = v120;
          *(_DWORD *)(v75 + 8) = v121;
          *(_BYTE *)(v75 + 12) = BYTE4(v121);
          a3 = v122;
          *(_QWORD *)(v75 + 16) = v122;
LABEL_105:
          v65 += 16;
          if ( v65 == v108 )
            goto LABEL_2;
          while ( 2 )
          {
            if ( *(_QWORD *)v65 == -1 )
            {
              if ( *(_DWORD *)(v65 + 8) != -1 )
                goto LABEL_109;
LABEL_117:
              if ( *(_BYTE *)(v65 + 12) )
                goto LABEL_109;
              v65 += 16;
              if ( v108 == v65 )
                goto LABEL_2;
              continue;
            }
            break;
          }
          if ( *(_QWORD *)v65 == -2 && *(_DWORD *)(v65 + 8) == -2 )
            goto LABEL_117;
LABEL_109:
          if ( v108 == v65 )
            goto LABEL_2;
        }
      }
      if ( *(_BYTE *)(v65 + 12) )
        goto LABEL_88;
      v65 += 16;
    }
  }
LABEL_2:
  (*(void (__fastcall **)(_QWORD, _QWORD, _QWORD))(**(_QWORD **)(*a1 + 224LL) + 208LL))(
    *(_QWORD *)(*a1 + 224LL),
    a1[38],
    0);
  v7 = a1[1];
  v120 = 0;
  v121 = 0;
  v122 = 0;
  v123 = 0;
  v93 = *(__int64 **)(v7 + 192);
  if ( *(__int64 **)(v7 + 184) == v93 )
  {
    v54 = 0;
    v55 = 0;
    goto LABEL_58;
  }
  v94 = *(__int64 **)(v7 + 184);
  v8 = a1;
  do
  {
    v95 = v94[1];
    v97 = *v94;
    if ( v95 != *v94 )
    {
      while ( 1 )
      {
        v9 = *(_QWORD **)v97;
        v96 = *(_QWORD **)v97;
        (*(void (__fastcall **)(_QWORD, _QWORD, _QWORD))(**(_QWORD **)(*v8 + 224) + 208LL))(
          *(_QWORD *)(*v8 + 224),
          *(_QWORD *)(*(_QWORD *)v97 + 40LL),
          0);
        v98 = (_QWORD *)v9[3];
        v104 = (_QWORD *)v9[2];
        if ( v98 != v104 )
          break;
LABEL_51:
        v48 = *v8;
        v49 = *(_QWORD *)(*v8 + 224);
        v50 = *(void (**)())(*(_QWORD *)v49 + 120LL);
        v51 = (__int64 *)(*v96 & 0xFFFFFFFFFFFFFFF8LL);
        if ( (*v96 & 4) != 0 )
        {
          v53 = (_QWORD *)v51[3];
          v52 = v51[4];
        }
        else
        {
          v52 = *v51;
          v53 = v51 + 4;
        }
        v125[1] = v53;
        v126 = 1283;
        v124 = "End of list: ";
        v125[2] = v52;
        if ( v50 != nullsub_98 )
        {
          ((void (__fastcall *)(__int64, const char **, __int64))v50)(v49, &v124, 1);
          v48 = *v8;
        }
        sub_31DC9D0(v48, 0);
        v97 += 8;
        if ( v95 == v97 )
          goto LABEL_56;
      }
      while ( 1 )
      {
        v10 = *v104;
        v11 = *(_QWORD *)(v8[10] + 8LL * ((*(_WORD *)(*v104 + 42LL) & 0x7FFFu) - 1));
        v111[0] = ((__int64 (__fastcall *)(__int64, _QWORD))v8[33])(v8[34], *v104);
        v111[1] = v12;
        v103 = v111[0];
        if ( *(_BYTE *)(v10 + 32) )
        {
          v24 = *(_BYTE *)(v10 + 43);
          v25 = *(_DWORD *)(v10 + 44);
          v26 = *(const char **)(v10 + 24);
          v115 = 1;
          v113 = v25;
          v112 = v26;
          v114[0] = v24 >> 7;
        }
        else
        {
          v115 = 0;
        }
        if ( *(_BYTE *)(v10 + 16) != 1 )
LABEL_183:
          abort();
        v13 = *(_BYTE *)(v10 + 43) >> 7;
        v14 = *(const char **)(v10 + 8);
        v15 = *(_DWORD *)(v10 + 44);
        v16 = v119;
        v124 = v14;
        v17 = v117;
        LODWORD(v125[0]) = v15;
        BYTE4(v125[0]) = v13;
        if ( v119 )
        {
          v110 = v14;
          v18 = v119 - 1;
          v109 = v15;
          v19 = sub_3723A60((__int64 *)&v110, &v109, (_BYTE *)v125 + 4);
          v20 = 1;
          for ( m = v18 & v19; ; m = v18 & v23 )
          {
            v22 = v17 + 24LL * m;
            if ( v124 == *(const char **)v22
              && LODWORD(v125[0]) == *(_DWORD *)(v22 + 8)
              && BYTE4(v125[0]) == *(_BYTE *)(v22 + 12) )
            {
              break;
            }
            if ( *(_QWORD *)v22 == -1 && *(_DWORD *)(v22 + 8) == -1 && !*(_BYTE *)(v22 + 12) )
            {
              v17 = v117;
              v16 = v119;
              goto LABEL_18;
            }
            v23 = v20 + m;
            ++v20;
          }
          v27 = v123;
          v28 = *(_QWORD *)(v22 + 16);
          if ( !(_DWORD)v123 )
          {
LABEL_68:
            ++v120;
            goto LABEL_69;
          }
        }
        else
        {
LABEL_18:
          v27 = v123;
          v28 = *(_QWORD *)(v17 + 24 * v16 + 16);
          if ( !(_DWORD)v123 )
            goto LABEL_68;
        }
        v29 = v27 - 1;
        v30 = 1;
        v31 = 0;
        v32 = v29 & (((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4));
        v33 = (__int64 *)(v121 + 8LL * v32);
        v34 = *v33;
        if ( *v33 != v28 )
        {
          while ( v34 != -4096 )
          {
            if ( v31 || v34 != -8192 )
              v33 = (__int64 *)v31;
            v31 = (unsigned int)(v30 + 1);
            v32 = v29 & (v30 + v32);
            v34 = *(_QWORD *)(v121 + 8LL * v32);
            if ( v28 == v34 )
              goto LABEL_20;
            ++v30;
            v31 = (__int64)v33;
            v33 = (__int64 *)(v121 + 8LL * v32);
          }
          if ( !v31 )
            v31 = (__int64)v33;
          ++v120;
          v47 = v122 + 1;
          if ( 4 * ((int)v122 + 1) >= 3 * v27 )
          {
LABEL_69:
            sub_37258B0((__int64)&v120, 2 * v27);
            if ( !(_DWORD)v123 )
              goto LABEL_184;
            v57 = (v123 - 1) & (((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4));
            v31 = v121 + 8LL * v57;
            v58 = *(_QWORD *)v31;
            v47 = v122 + 1;
            if ( v28 != *(_QWORD *)v31 )
            {
              v59 = 1;
              v60 = 0;
              while ( v58 != -4096 )
              {
                if ( v58 == -8192 && !v60 )
                  v60 = (_QWORD *)v31;
                v57 = (v123 - 1) & (v59 + v57);
                v31 = v121 + 8LL * v57;
                v58 = *(_QWORD *)v31;
                if ( v28 == *(_QWORD *)v31 )
                  goto LABEL_45;
                ++v59;
              }
              if ( v60 )
                v31 = (__int64)v60;
            }
          }
          else if ( v27 - HIDWORD(v122) - v47 <= v27 >> 3 )
          {
            sub_37258B0((__int64)&v120, v27);
            if ( !(_DWORD)v123 )
            {
LABEL_184:
              LODWORD(v122) = v122 + 1;
              BUG();
            }
            v61 = 1;
            v62 = 0;
            v63 = (v123 - 1) & (((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4));
            v31 = v121 + 8LL * v63;
            v64 = *(_QWORD *)v31;
            v47 = v122 + 1;
            if ( v28 != *(_QWORD *)v31 )
            {
              while ( v64 != -4096 )
              {
                if ( v64 == -8192 && !v62 )
                  v62 = (_QWORD *)v31;
                v63 = (v123 - 1) & (v61 + v63);
                v31 = v121 + 8LL * v63;
                v64 = *(_QWORD *)v31;
                if ( v28 == *(_QWORD *)v31 )
                  goto LABEL_45;
                ++v61;
              }
              if ( v62 )
                v31 = (__int64)v62;
            }
          }
LABEL_45:
          LODWORD(v122) = v47;
          if ( *(_QWORD *)v31 != -4096 )
            --HIDWORD(v122);
          *(_QWORD *)v31 = v28;
          (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(*v8 + 224) + 208LL))(
            *(_QWORD *)(*v8 + 224),
            v28,
            0);
        }
LABEL_20:
        (*(void (__fastcall **)(__int64, _QWORD, const char *, _QWORD, __int64, __int64))(*(_QWORD *)*v8 + 424LL))(
          *v8,
          *(_WORD *)(v10 + 42) & 0x7FFF,
          "Abbreviation code",
          0,
          v29,
          v31);
        v35 = *(_QWORD *)(v11 + 16);
        v107 = v35 + 8LL * *(unsigned int *)(v11 + 24);
        while ( v107 != v35 )
        {
          while ( 1 )
          {
            v37 = *(_QWORD *)(*v8 + 224);
            v38 = *(void (**)())(*(_QWORD *)v37 + 120LL);
            v39 = sub_E0CB90(*(_DWORD *)v35);
            v126 = 261;
            v124 = v39;
            v125[0] = v40;
            if ( v38 != nullsub_98 )
              ((void (__fastcall *)(__int64, const char **, __int64))v38)(v37, &v124, 1);
            v41 = *(_DWORD *)v35;
            if ( *(_DWORD *)v35 == 3 )
              break;
            if ( v41 <= 3 )
            {
              if ( v41 - 1 > 1 )
                goto LABEL_182;
              v36 = (_QWORD **)*v8;
              v124 = (const char *)v103;
              sub_32152F0((__int64 *)&v124, v36, *(_WORD *)(v35 + 4));
            }
            else
            {
              if ( v41 != 4 )
LABEL_182:
                BUG();
              if ( *(_WORD *)(v35 + 4) != 25 )
              {
                if ( v119 )
                {
                  v99 = v117;
                  LODWORD(v110) = v113;
                  v100 = v119;
                  v124 = v112;
                  v42 = sub_3723A60((__int64 *)&v124, &v110, v114);
                  v43 = 1;
                  for ( n = (v100 - 1) & v42; ; n = (v100 - 1) & v46 )
                  {
                    v45 = v99 + 24LL * n;
                    if ( v112 == *(const char **)v45 && v113 == *(_DWORD *)(v45 + 8) && v114[0] == *(_BYTE *)(v45 + 12) )
                      break;
                    if ( *(_QWORD *)v45 == -1 && *(_DWORD *)(v45 + 8) == -1 && !*(_BYTE *)(v45 + 12) )
                      break;
                    v46 = v43 + n;
                    ++v43;
                  }
                }
                sub_31DCA50(*v8);
              }
            }
            v35 += 8;
            if ( v107 == v35 )
              goto LABEL_50;
          }
          if ( *(_BYTE *)(v10 + 16) != 1 )
            goto LABEL_183;
          v35 += 8;
          sub_31DCA10(*v8, *(_DWORD *)(v10 + 8));
        }
LABEL_50:
        if ( v98 == ++v104 )
          goto LABEL_51;
      }
    }
LABEL_56:
    v94 += 3;
  }
  while ( v93 != v94 );
  v54 = v121;
  v55 = 8LL * (unsigned int)v123;
LABEL_58:
  sub_C7D6A0(v54, v55, 8);
  return sub_C7D6A0(v117, 24LL * v119, 8);
}
