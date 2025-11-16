// Function: sub_25167D0
// Address: 0x25167d0
//
bool __fastcall sub_25167D0(__int64 a1, __int64 a2)
{
  __int64 *v2; // rax
  __int64 v3; // rdx
  __int64 *v4; // r12
  __int64 v5; // rdi
  __int64 *v6; // rbx
  int v7; // eax
  __int64 v8; // rdx
  char v9; // cl
  __int64 **v10; // rax
  __int64 v11; // rdx
  __int64 *v12; // rdx
  __int64 v13; // rdi
  bool v14; // zf
  __int64 v15; // rcx
  __int64 **v16; // r12
  __int64 *v17; // rdi
  __int64 **v18; // rbx
  bool result; // al
  __int64 *v20; // rax
  unsigned int v21; // esi
  __int64 v22; // r9
  int v23; // r11d
  char *v24; // rcx
  unsigned int v25; // edx
  __int64 **v26; // rax
  __int64 *v27; // r8
  __int64 **v28; // rax
  __int64 **v29; // rax
  int v30; // edx
  __int64 v31; // rax
  __int64 v32; // rbx
  __int64 v33; // rdx
  char v34; // r12
  unsigned int v35; // r15d
  __int64 v36; // rax
  char v37; // r12
  unsigned int v38; // r15d
  __int64 v39; // r13
  _QWORD *v40; // rax
  __int64 v41; // rcx
  __int64 v42; // rdx
  char v43; // cl
  __int64 v44; // r15
  unsigned __int8 *v45; // r12
  __int64 v46; // rbx
  __int64 v47; // rax
  __int64 *v48; // rcx
  int v49; // edi
  __int64 v50; // rdx
  unsigned __int64 *v51; // rdi
  __int64 *v52; // rdx
  _QWORD *v53; // r15
  unsigned __int8 *v54; // rax
  unsigned __int8 *v55; // rax
  __int64 v56; // rdx
  __int64 v57; // rax
  unsigned int v58; // edi
  __int64 *v59; // rdx
  __int64 v60; // r8
  _QWORD *v61; // rsi
  __int64 v62; // rdi
  unsigned int *v63; // rbx
  unsigned int *v64; // r12
  __int64 v65; // rdx
  __int64 v66; // rsi
  __int64 v67; // rbx
  __int64 v68; // rcx
  unsigned int v69; // esi
  int v70; // ebx
  __int64 *v71; // rdi
  __int64 v72; // rcx
  __int64 v73; // r9
  unsigned int v74; // edx
  __int64 *v75; // rax
  __int64 v76; // r8
  __int64 *v77; // rax
  unsigned int v78; // eax
  _QWORD *v79; // rbx
  _QWORD *v80; // r12
  __int64 v81; // rax
  __int64 v82; // rdx
  __int64 v83; // rax
  __int64 *v84; // rax
  int v85; // r15d
  unsigned int v86; // eax
  _QWORD *v87; // rbx
  _QWORD *v88; // r12
  __int64 v89; // rsi
  __int64 v90; // rdx
  __int64 v91; // rcx
  int v92; // eax
  int v93; // edx
  __int64 v94; // rbx
  int v95; // eax
  unsigned int v96; // ecx
  _QWORD *v97; // rax
  _QWORD *i; // rdx
  unsigned int v99; // eax
  int v100; // r12d
  unsigned int v101; // eax
  __int64 *v103; // [rsp+20h] [rbp-170h]
  unsigned __int64 *v104; // [rsp+28h] [rbp-168h]
  __int64 *v105; // [rsp+28h] [rbp-168h]
  __int64 *v106; // [rsp+38h] [rbp-158h]
  __int64 v107; // [rsp+40h] [rbp-150h]
  __int64 **v108; // [rsp+58h] [rbp-138h]
  __int64 v110; // [rsp+70h] [rbp-120h]
  __int64 v111; // [rsp+80h] [rbp-110h] BYREF
  __int64 *v112; // [rsp+88h] [rbp-108h] BYREF
  __int64 *v113; // [rsp+90h] [rbp-100h] BYREF
  __int64 v114; // [rsp+98h] [rbp-F8h] BYREF
  __int64 v115; // [rsp+A0h] [rbp-F0h] BYREF
  __int64 v116; // [rsp+A8h] [rbp-E8h]
  __int64 v117; // [rsp+B0h] [rbp-E0h]
  __int64 v118; // [rsp+C0h] [rbp-D0h] BYREF
  _QWORD *v119; // [rsp+C8h] [rbp-C8h]
  __int64 v120; // [rsp+D0h] [rbp-C0h]
  unsigned int v121; // [rsp+D8h] [rbp-B8h]
  _QWORD *v122; // [rsp+E8h] [rbp-A8h]
  unsigned int v123; // [rsp+F8h] [rbp-98h]
  char v124; // [rsp+100h] [rbp-90h]
  char *v125; // [rsp+110h] [rbp-80h] BYREF
  __int64 v126; // [rsp+118h] [rbp-78h] BYREF
  const char *v127; // [rsp+120h] [rbp-70h] BYREF
  __int64 v128; // [rsp+128h] [rbp-68h]
  __int64 *j; // [rsp+130h] [rbp-60h]

  v2 = *(__int64 **)(a1 + 8);
  if ( *(_BYTE *)(a1 + 28) )
    v3 = *(unsigned int *)(a1 + 20);
  else
    v3 = *(unsigned int *)(a1 + 16);
  v4 = &v2[v3];
  if ( v2 != v4 )
  {
    while ( 1 )
    {
      v5 = *v2;
      v6 = v2;
      if ( (unsigned __int64)*v2 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v4 == ++v2 )
        goto LABEL_6;
    }
    while ( v4 != v6 )
    {
      result = sub_250E810(v5);
      if ( !result )
        return result;
      v20 = v6 + 1;
      if ( v6 + 1 == v4 )
        break;
      while ( 1 )
      {
        v5 = *v20;
        v6 = v20;
        if ( (unsigned __int64)*v20 < 0xFFFFFFFFFFFFFFFELL )
          break;
        if ( v4 == ++v20 )
          goto LABEL_6;
      }
    }
  }
LABEL_6:
  ++*(_QWORD *)a2;
  v7 = *(_DWORD *)(a2 + 16);
  if ( v7 )
  {
    v8 = *(unsigned int *)(a2 + 24);
    v96 = 4 * v7;
    if ( (unsigned int)(4 * v7) < 0x40 )
      v96 = 64;
    if ( v96 < (unsigned int)v8 )
    {
      v99 = v7 - 1;
      if ( v99 )
      {
        _BitScanReverse(&v99, v99);
        v100 = 1 << (33 - (v99 ^ 0x1F));
        if ( v100 < 64 )
          v100 = 64;
        if ( v100 == (_DWORD)v8 )
        {
          sub_2510FE0(a2);
          goto LABEL_11;
        }
      }
      else
      {
        v100 = 64;
      }
      sub_C7D6A0(*(_QWORD *)(a2 + 8), 16LL * *(unsigned int *)(a2 + 24), 8);
      v101 = sub_2507810(v100);
      *(_DWORD *)(a2 + 24) = v101;
      if ( v101 )
      {
        *(_QWORD *)(a2 + 8) = sub_C7D670(16LL * v101, 8);
        sub_2510FE0(a2);
        goto LABEL_11;
      }
      goto LABEL_10;
    }
LABEL_180:
    v97 = *(_QWORD **)(a2 + 8);
    for ( i = &v97[2 * v8]; i != v97; v97 += 2 )
      *v97 = -4096;
    *(_QWORD *)(a2 + 16) = 0;
    goto LABEL_11;
  }
  if ( *(_DWORD *)(a2 + 20) )
  {
    v8 = *(unsigned int *)(a2 + 24);
    if ( (unsigned int)v8 > 0x40 )
    {
      sub_C7D6A0(*(_QWORD *)(a2 + 8), 16LL * *(unsigned int *)(a2 + 24), 8);
      *(_DWORD *)(a2 + 24) = 0;
LABEL_10:
      *(_QWORD *)(a2 + 8) = 0;
      *(_QWORD *)(a2 + 16) = 0;
      goto LABEL_11;
    }
    goto LABEL_180;
  }
LABEL_11:
  v9 = *(_BYTE *)(a1 + 28);
  v10 = *(__int64 ***)(a1 + 8);
  if ( v9 )
  {
    v11 = *(unsigned int *)(a1 + 20);
    v108 = &v10[v11];
    if ( v10 == v108 )
      goto LABEL_18;
  }
  else
  {
    v11 = *(unsigned int *)(a1 + 16);
    v108 = &v10[v11];
    if ( v10 == v108 )
      goto LABEL_18;
  }
  v12 = *(__int64 **)(a1 + 8);
  while ( 1 )
  {
    v13 = *v12;
    if ( (unsigned __int64)*v12 < 0xFFFFFFFFFFFFFFFELL )
      break;
    if ( v108 == (__int64 **)++v12 )
      goto LABEL_16;
  }
  v106 = v12;
  if ( v12 == (__int64 *)v108 )
  {
LABEL_16:
    v14 = v9 == 0;
    v15 = a1;
    if ( !v14 )
      goto LABEL_127;
    goto LABEL_17;
  }
  do
  {
    v31 = *(_QWORD *)(v13 + 40);
    v32 = *(_QWORD *)(v13 + 24);
    v111 = v13;
    v107 = v31;
    v125 = (char *)sub_BD5D20(v13);
    v127 = ".internalized";
    LOWORD(j) = 773;
    v126 = v33;
    v34 = *(_BYTE *)(v111 + 32);
    v35 = *(_DWORD *)(*(_QWORD *)(v111 + 8) + 8LL);
    v36 = sub_BD2DA0(136);
    v37 = v34 & 0xF;
    v38 = v35 >> 8;
    v39 = v36;
    if ( v36 )
      sub_B2C3B0(v36, v32, v37, v38, (__int64)&v125, 0);
    v118 = 0;
    v121 = 128;
    v40 = (_QWORD *)sub_C7D670(0x2000, 8);
    v120 = 0;
    v119 = v40;
    v126 = 2;
    v42 = (__int64)&v40[8 * (unsigned __int64)v121];
    v127 = 0;
    v128 = -4096;
    v125 = (char *)&unk_49DD7B0;
    for ( j = 0; (_QWORD *)v42 != v40; v40 += 8 )
    {
      if ( v40 )
      {
        v43 = v126;
        v40[2] = 0;
        v40[3] = -4096;
        *v40 = &unk_49DD7B0;
        v40[1] = v43 & 6;
        v41 = (__int64)j;
        v40[4] = j;
      }
    }
    v124 = 0;
    if ( (*(_BYTE *)(v39 + 2) & 1) != 0 )
      sub_B2C6D0(v39, (__int64)&unk_49DD7B0, v42, v41);
    v44 = v111;
    v45 = *(unsigned __int8 **)(v39 + 96);
    if ( (*(_BYTE *)(v111 + 2) & 1) != 0 )
    {
      sub_B2C6D0(v111, (__int64)&unk_49DD7B0, v42, v41);
      v46 = *(_QWORD *)(v44 + 96);
      v110 = v46 + 40LL * *(_QWORD *)(v44 + 104);
      if ( (*(_BYTE *)(v44 + 2) & 1) != 0 )
      {
        sub_B2C6D0(v44, (__int64)&unk_49DD7B0, v90, v91);
        v46 = *(_QWORD *)(v44 + 96);
      }
    }
    else
    {
      v46 = *(_QWORD *)(v111 + 96);
      v110 = v46 + 40LL * *(_QWORD *)(v111 + 104);
    }
    if ( v46 != v110 )
    {
      while ( 1 )
      {
        v125 = (char *)sub_BD5D20(v46);
        LOWORD(j) = 261;
        v126 = v56;
        sub_BD6B50(v45, (const char **)&v125);
        v57 = v46;
        v128 = v46;
        BYTE1(v57) = BYTE1(v46) & 0xEF;
        v126 = 2;
        v127 = 0;
        if ( v57 != -8192 && v46 )
          sub_BD73F0((__int64)&v126);
        j = &v118;
        v125 = (char *)&unk_49DD7B0;
        if ( !v121 )
          break;
        v47 = v128;
        v58 = (v121 - 1) & (((unsigned int)v128 >> 9) ^ ((unsigned int)v128 >> 4));
        v59 = &v119[8 * (unsigned __int64)v58];
        v60 = v59[3];
        if ( v128 != v60 )
        {
          v85 = 1;
          v48 = 0;
          while ( v60 != -4096 )
          {
            if ( !v48 && v60 == -8192 )
              v48 = v59;
            v58 = (v121 - 1) & (v85 + v58);
            v59 = &v119[8 * (unsigned __int64)v58];
            v60 = v59[3];
            if ( v128 == v60 )
              goto LABEL_86;
            ++v85;
          }
          if ( !v48 )
            v48 = v59;
          ++v118;
          v49 = v120 + 1;
          v113 = v48;
          if ( 4 * ((int)v120 + 1) < 3 * v121 )
          {
            if ( v121 - HIDWORD(v120) - v49 <= v121 >> 3 )
            {
              sub_CF32C0((__int64)&v118, v121);
              sub_F9E960((__int64)&v118, (__int64)&v125, &v113);
              v47 = v128;
              v49 = v120 + 1;
              v48 = v113;
            }
LABEL_61:
            LODWORD(v120) = v49;
            v50 = v48[3];
            if ( v50 == -4096 )
            {
              v51 = (unsigned __int64 *)(v48 + 1);
              if ( v47 != -4096 )
                goto LABEL_66;
            }
            else
            {
              --HIDWORD(v120);
              if ( v50 != v47 )
              {
                v51 = (unsigned __int64 *)(v48 + 1);
                if ( v50 && v50 != -8192 )
                {
                  v103 = v48;
                  v104 = (unsigned __int64 *)(v48 + 1);
                  sub_BD60C0(v51);
                  v47 = v128;
                  v48 = v103;
                  v51 = v104;
                }
LABEL_66:
                v48[3] = v47;
                if ( v47 == 0 || v47 == -4096 || v47 == -8192 )
                {
                  v47 = v128;
                }
                else
                {
                  v105 = v48;
                  sub_BD6050(v51, v126 & 0xFFFFFFFFFFFFFFF8LL);
                  v47 = v128;
                  v48 = v105;
                }
              }
            }
            v52 = j;
            v48[5] = 6;
            v53 = v48 + 5;
            v48[6] = 0;
            v48[4] = (__int64)v52;
            v48[7] = 0;
            goto LABEL_70;
          }
LABEL_60:
          sub_CF32C0((__int64)&v118, 2 * v121);
          sub_F9E960((__int64)&v118, (__int64)&v125, &v113);
          v47 = v128;
          v48 = v113;
          v49 = v120 + 1;
          goto LABEL_61;
        }
LABEL_86:
        v53 = v59 + 5;
LABEL_70:
        v125 = (char *)&unk_49DB368;
        if ( v47 != 0 && v47 != -4096 && v47 != -8192 )
          sub_BD60C0(&v126);
        v54 = (unsigned __int8 *)v53[2];
        if ( v54 != v45 )
        {
          if ( v54 != 0 && v54 + 4096 != 0 && v54 != (unsigned __int8 *)-8192LL )
            sub_BD60C0(v53);
          v55 = v45;
          v53[2] = v45;
          BYTE1(v55) = BYTE1(v45) & 0xEF;
          if ( v55 != (unsigned __int8 *)-8192LL && v45 )
            sub_BD73F0((__int64)v53);
        }
        v45 += 40;
        v46 += 40;
        if ( v110 == v46 )
          goto LABEL_90;
      }
      ++v118;
      v113 = 0;
      goto LABEL_60;
    }
LABEL_90:
    v61 = (_QWORD *)v111;
    v125 = (char *)&v127;
    v126 = 0x800000000LL;
    *(_BYTE *)(v39 + 128) = *(_BYTE *)(v111 + 128);
    sub_F4BB00(v39, v61, (__int64)&v118, 0, (__int64)&v125, byte_3F871B3, 0, 0, 0);
    v62 = v111;
    *(_WORD *)(v39 + 32) = *(_WORD *)(v39 + 32) & 0xBCC0 | 0x4008;
    v113 = &v115;
    v114 = 0x100000000LL;
    sub_B9A9D0(v62, (__int64)&v113);
    v63 = (unsigned int *)v113;
    v64 = (unsigned int *)&v113[2 * (unsigned int)v114];
    if ( v64 != (unsigned int *)v113 )
    {
      do
      {
        while ( (*(_BYTE *)(v39 + 7) & 0x20) != 0 )
        {
          v63 += 4;
          if ( v64 == v63 )
            goto LABEL_95;
        }
        v65 = *((_QWORD *)v63 + 1);
        v66 = *v63;
        v63 += 4;
        sub_B994D0(v39, v66, v65);
      }
      while ( v64 != v63 );
    }
LABEL_95:
    v67 = v111;
    sub_BA8540(v107 + 24, v39);
    v68 = *(_QWORD *)(v67 + 56);
    *(_QWORD *)(v39 + 64) = v67 + 56;
    v68 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(v39 + 56) = v68 | *(_QWORD *)(v39 + 56) & 7LL;
    *(_QWORD *)(v68 + 8) = v39 + 56;
    *(_QWORD *)(v67 + 56) = *(_QWORD *)(v67 + 56) & 7LL | (v39 + 56);
    *(_BYTE *)(v39 + 33) |= 0x40u;
    v69 = *(_DWORD *)(a2 + 24);
    if ( !v69 )
    {
      v112 = 0;
      ++*(_QWORD *)a2;
      goto LABEL_162;
    }
    v70 = 1;
    v71 = 0;
    v72 = v111;
    v73 = *(_QWORD *)(a2 + 8);
    v74 = (v69 - 1) & (((unsigned int)v111 >> 9) ^ ((unsigned int)v111 >> 4));
    v75 = (__int64 *)(v73 + 16LL * v74);
    v76 = *v75;
    if ( v111 != *v75 )
    {
      while ( v76 != -4096 )
      {
        if ( !v71 && v76 == -8192 )
          v71 = v75;
        v74 = (v69 - 1) & (v70 + v74);
        v75 = (__int64 *)(v73 + 16LL * v74);
        v76 = *v75;
        if ( v111 == *v75 )
          goto LABEL_97;
        ++v70;
      }
      if ( !v71 )
        v71 = v75;
      ++*(_QWORD *)a2;
      v92 = *(_DWORD *)(a2 + 16);
      v112 = v71;
      v93 = v92 + 1;
      if ( 4 * (v92 + 1) < 3 * v69 )
      {
        v94 = a2;
        if ( v69 - *(_DWORD *)(a2 + 20) - v93 > v69 >> 3 )
        {
LABEL_158:
          *(_DWORD *)(a2 + 16) = v93;
          if ( *v71 != -4096 )
            --*(_DWORD *)(a2 + 20);
          *v71 = v72;
          v77 = v71 + 1;
          v71[1] = 0;
          goto LABEL_98;
        }
LABEL_163:
        sub_9E0010(v94, v69);
        sub_25108F0(v94, &v111, &v112);
        v72 = v111;
        v71 = v112;
        v93 = *(_DWORD *)(v94 + 16) + 1;
        goto LABEL_158;
      }
LABEL_162:
      v94 = a2;
      v69 *= 2;
      goto LABEL_163;
    }
LABEL_97:
    v77 = v75 + 1;
LABEL_98:
    *v77 = v39;
    if ( v113 != &v115 )
      _libc_free((unsigned __int64)v113);
    if ( v125 != (char *)&v127 )
      _libc_free((unsigned __int64)v125);
    if ( v124 )
    {
      v86 = v123;
      v124 = 0;
      if ( v123 )
      {
        v87 = v122;
        v88 = &v122[2 * v123];
        do
        {
          if ( *v87 != -4096 && *v87 != -8192 )
          {
            v89 = v87[1];
            if ( v89 )
              sub_B91220((__int64)(v87 + 1), v89);
          }
          v87 += 2;
        }
        while ( v88 != v87 );
        v86 = v123;
      }
      sub_C7D6A0((__int64)v122, 16LL * v86, 8);
    }
    v78 = v121;
    if ( v121 )
    {
      v114 = 2;
      v79 = v119;
      v115 = 0;
      v126 = 2;
      v80 = &v119[8 * (unsigned __int64)v121];
      v125 = (char *)&unk_49DD7B0;
      v81 = -4096;
      v116 = -4096;
      v113 = (__int64 *)&unk_49DD7B0;
      v117 = 0;
      v127 = 0;
      v128 = -8192;
      j = 0;
      while ( 1 )
      {
        v82 = v79[3];
        if ( v81 != v82 )
        {
          v81 = v128;
          if ( v82 != v128 )
          {
            v83 = v79[7];
            if ( v83 != 0 && v83 != -4096 && v83 != -8192 )
            {
              sub_BD60C0(v79 + 5);
              v82 = v79[3];
            }
            v81 = v82;
          }
        }
        *v79 = &unk_49DB368;
        if ( v81 != -4096 && v81 != 0 && v81 != -8192 )
          sub_BD60C0(v79 + 1);
        v79 += 8;
        if ( v80 == v79 )
          break;
        v81 = v116;
      }
      v125 = (char *)&unk_49DB368;
      if ( v128 != 0 && v128 != -4096 && v128 != -8192 )
        sub_BD60C0(&v126);
      v113 = (__int64 *)&unk_49DB368;
      if ( v116 != 0 && v116 != -4096 && v116 != -8192 )
        sub_BD60C0(&v114);
      v78 = v121;
    }
    sub_C7D6A0((__int64)v119, (unsigned __int64)v78 << 6, 8);
    v84 = v106 + 1;
    if ( v106 + 1 == (__int64 *)v108 )
      break;
    while ( 1 )
    {
      v13 = *v84;
      if ( (unsigned __int64)*v84 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v108 == (__int64 **)++v84 )
        goto LABEL_126;
    }
    v106 = v84;
  }
  while ( v84 != (__int64 *)v108 );
LABEL_126:
  v10 = *(__int64 ***)(a1 + 8);
  v15 = a1;
  if ( *(_BYTE *)(a1 + 28) )
  {
LABEL_127:
    v11 = *(unsigned int *)(v15 + 20);
    goto LABEL_18;
  }
LABEL_17:
  v11 = *(unsigned int *)(v15 + 16);
LABEL_18:
  v16 = &v10[v11];
  while ( 1 )
  {
    if ( v16 == v10 )
      return 1;
    v17 = *v10;
    v18 = v10;
    if ( (unsigned __int64)*v10 < 0xFFFFFFFFFFFFFFFELL )
      break;
    ++v10;
  }
  if ( v16 != v10 )
  {
    v113 = *v10;
    v21 = *(_DWORD *)(a2 + 24);
    if ( !v21 )
      goto LABEL_43;
    do
    {
      v22 = *(_QWORD *)(a2 + 8);
      v23 = 1;
      v24 = 0;
      v25 = (v21 - 1) & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
      v26 = (__int64 **)(v22 + 16LL * v25);
      v27 = *v26;
      if ( v17 == *v26 )
        goto LABEL_36;
      while ( v27 != (__int64 *)-4096LL )
      {
        if ( !v24 && v27 == (__int64 *)-8192LL )
          v24 = (char *)v26;
        v25 = (v21 - 1) & (v23 + v25);
        v26 = (__int64 **)(v22 + 16LL * v25);
        v27 = *v26;
        if ( v17 == *v26 )
        {
LABEL_36:
          v28 = v26 + 1;
          goto LABEL_37;
        }
        ++v23;
      }
      if ( !v24 )
        v24 = (char *)v26;
      v95 = *(_DWORD *)(a2 + 16);
      ++*(_QWORD *)a2;
      v30 = v95 + 1;
      v125 = v24;
      if ( 4 * (v95 + 1) >= 3 * v21 )
        goto LABEL_44;
      if ( v21 - *(_DWORD *)(a2 + 20) - v30 <= v21 >> 3 )
      {
LABEL_45:
        sub_9E0010(a2, v21);
        sub_25108F0(a2, (__int64 *)&v113, &v125);
        v17 = v113;
        v24 = v125;
        v30 = *(_DWORD *)(a2 + 16) + 1;
      }
      *(_DWORD *)(a2 + 16) = v30;
      if ( *(_QWORD *)v24 != -4096 )
        --*(_DWORD *)(a2 + 20);
      *(_QWORD *)v24 = v17;
      v28 = (__int64 **)(v24 + 8);
      *((_QWORD *)v24 + 1) = 0;
      v17 = v113;
LABEL_37:
      v118 = a2;
      sub_BD79D0(v17, *v28, (unsigned __int8 (__fastcall *)(__int64, __int64 *))sub_2506E40, (__int64)&v118);
      v29 = v18 + 1;
      if ( v18 + 1 == v16 )
        return 1;
      v17 = *v29;
      for ( ++v18; (unsigned __int64)*v29 >= 0xFFFFFFFFFFFFFFFELL; v18 = v29 )
      {
        if ( v16 == ++v29 )
          return 1;
        v17 = *v29;
      }
      if ( v16 == v18 )
        return 1;
      v21 = *(_DWORD *)(a2 + 24);
      v113 = v17;
    }
    while ( v21 );
LABEL_43:
    ++*(_QWORD *)a2;
    v125 = 0;
LABEL_44:
    v21 *= 2;
    goto LABEL_45;
  }
  return 1;
}
