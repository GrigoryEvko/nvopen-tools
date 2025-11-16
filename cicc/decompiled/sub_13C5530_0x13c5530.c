// Function: sub_13C5530
// Address: 0x13c5530
//
void __fastcall sub_13C5530(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  __int64 v4; // r12
  __int64 v5; // rbx
  __int64 v6; // r14
  _QWORD *v7; // rax
  _QWORD *v8; // rax
  _QWORD *v9; // rax
  __int64 v10; // rsi
  _QWORD *v11; // r15
  __int64 v12; // rax
  __int64 v13; // r15
  __int64 *v14; // rcx
  unsigned int v15; // eax
  unsigned int v16; // eax
  _QWORD *v17; // rax
  __int64 v18; // r14
  _QWORD *v19; // rax
  _QWORD *v20; // r12
  __int64 v21; // rax
  char *v22; // rax
  char *v23; // r14
  __int64 v24; // r12
  char *v25; // rbx
  _QWORD *v26; // rdi
  _QWORD *v27; // rcx
  _QWORD *v28; // rdi
  unsigned int v29; // r8d
  _QWORD *v30; // rcx
  _QWORD *v31; // rsi
  unsigned int v32; // edi
  _QWORD *v33; // rcx
  _QWORD *v34; // rax
  char v35; // dl
  unsigned int v36; // esi
  __int64 v37; // r8
  unsigned int v38; // ecx
  _QWORD *v39; // rdi
  __int64 v40; // rdx
  char *v41; // rax
  char *v42; // rax
  char *v43; // r11
  __int64 v44; // rbx
  char *v45; // r14
  char *v46; // r12
  _QWORD *v47; // rax
  char v48; // dl
  unsigned int v49; // esi
  __int64 v50; // r8
  unsigned int v51; // ecx
  _QWORD *v52; // rax
  __int64 v53; // rdi
  char *v54; // rax
  _QWORD *v55; // rsi
  _QWORD *v56; // rcx
  _QWORD *v57; // rax
  __int64 v58; // rsi
  _QWORD *v59; // r8
  __int64 v60; // rax
  int v61; // r10d
  int v62; // r10d
  __int64 v63; // r8
  unsigned int v64; // eax
  int v65; // edx
  __int64 v66; // rcx
  int v67; // r11d
  _QWORD *v68; // rsi
  int v69; // r11d
  _QWORD *v70; // r10
  int v71; // ecx
  int v72; // r10d
  int v73; // r10d
  __int64 v74; // r8
  int v75; // r11d
  unsigned int v76; // eax
  __int64 v77; // rcx
  _QWORD *v78; // rsi
  _QWORD *v79; // rcx
  _QWORD *v80; // rax
  __int64 v81; // rsi
  _QWORD *v82; // r8
  __int64 v83; // rax
  int v84; // r11d
  int v85; // r11d
  __int64 v86; // r9
  unsigned int v87; // edx
  int v88; // ecx
  __int64 v89; // r8
  int v90; // edi
  _QWORD *v91; // rsi
  int v92; // r11d
  _QWORD *v93; // r10
  int v94; // ecx
  int v95; // r11d
  int v96; // r11d
  __int64 v97; // r9
  int v98; // edi
  unsigned int v99; // edx
  __int64 v100; // r8
  __int64 v101; // [rsp+0h] [rbp-300h]
  __int64 v102; // [rsp+8h] [rbp-2F8h]
  __int64 v103; // [rsp+18h] [rbp-2E8h]
  _QWORD *v104; // [rsp+18h] [rbp-2E8h]
  unsigned int v105; // [rsp+18h] [rbp-2E8h]
  __int64 v106; // [rsp+18h] [rbp-2E8h]
  _QWORD *v107; // [rsp+18h] [rbp-2E8h]
  unsigned int v108; // [rsp+18h] [rbp-2E8h]
  _QWORD *v110; // [rsp+30h] [rbp-2D0h]
  __int64 v111; // [rsp+38h] [rbp-2C8h]
  __int64 v112; // [rsp+38h] [rbp-2C8h]
  __int64 v113; // [rsp+40h] [rbp-2C0h] BYREF
  _BYTE *v114; // [rsp+48h] [rbp-2B8h]
  void *s; // [rsp+50h] [rbp-2B0h]
  _BYTE v116[12]; // [rsp+58h] [rbp-2A8h]
  _BYTE v117[136]; // [rsp+68h] [rbp-298h] BYREF
  __int64 v118; // [rsp+F0h] [rbp-210h] BYREF
  _BYTE *v119; // [rsp+F8h] [rbp-208h]
  void *v120; // [rsp+100h] [rbp-200h]
  _BYTE v121[12]; // [rsp+108h] [rbp-1F8h]
  _BYTE v122[136]; // [rsp+118h] [rbp-1E8h] BYREF
  __int64 v123; // [rsp+1A0h] [rbp-160h] BYREF
  _BYTE *v124; // [rsp+1A8h] [rbp-158h]
  _BYTE *v125; // [rsp+1B0h] [rbp-150h]
  __int64 v126; // [rsp+1B8h] [rbp-148h]
  int v127; // [rsp+1C0h] [rbp-140h]
  _BYTE v128[312]; // [rsp+1C8h] [rbp-138h] BYREF

  v2 = a2;
  v4 = a2 + 24;
  v5 = *(_QWORD *)(a2 + 32);
  v124 = v128;
  v125 = v128;
  v123 = 0;
  v126 = 32;
  v127 = 0;
  if ( v5 == a2 + 24 )
    goto LABEL_16;
  do
  {
    while ( 1 )
    {
      if ( !v5 )
        BUG();
      if ( (*(_BYTE *)(v5 - 24) & 0xFu) - 7 <= 1 )
      {
        v6 = v5 - 56;
        if ( !(unsigned __int8)sub_13C13B0(a1, (_QWORD *)(v5 - 56), 0, 0, 0) )
          break;
      }
      v5 = *(_QWORD *)(v5 + 8);
      if ( v4 == v5 )
        goto LABEL_15;
    }
    v7 = *(_QWORD **)(a1 + 32);
    if ( *(_QWORD **)(a1 + 40) != v7 )
    {
LABEL_8:
      sub_16CCBA0(a1 + 24, v5 - 56);
      goto LABEL_9;
    }
    v28 = &v7[*(unsigned int *)(a1 + 52)];
    v29 = *(_DWORD *)(a1 + 52);
    if ( v7 == v28 )
    {
LABEL_171:
      if ( v29 >= *(_DWORD *)(a1 + 48) )
        goto LABEL_8;
      *(_DWORD *)(a1 + 52) = v29 + 1;
      *v28 = v6;
      ++*(_QWORD *)(a1 + 24);
    }
    else
    {
      v30 = 0;
      while ( v6 != *v7 )
      {
        if ( *v7 == -2 )
          v30 = v7;
        if ( v28 == ++v7 )
        {
          if ( !v30 )
            goto LABEL_171;
          *v30 = v6;
          --*(_DWORD *)(a1 + 56);
          ++*(_QWORD *)(a1 + 24);
          break;
        }
      }
    }
LABEL_9:
    v8 = v124;
    if ( v125 != v124 )
      goto LABEL_10;
    v26 = &v124[8 * HIDWORD(v126)];
    if ( v124 == (_BYTE *)v26 )
    {
LABEL_169:
      if ( HIDWORD(v126) < (unsigned int)v126 )
      {
        ++HIDWORD(v126);
        *v26 = v6;
        ++v123;
        goto LABEL_11;
      }
LABEL_10:
      sub_16CCBA0(&v123, v5 - 56);
      goto LABEL_11;
    }
    v27 = 0;
    while ( v6 != *v8 )
    {
      if ( *v8 == -2 )
        v27 = v8;
      if ( v26 == ++v8 )
      {
        if ( !v27 )
          goto LABEL_169;
        *v27 = v6;
        --v127;
        ++v123;
        break;
      }
    }
LABEL_11:
    v111 = *(_QWORD *)(a1 + 328);
    v9 = (_QWORD *)sub_22077B0(64);
    v10 = v111;
    v9[3] = 2;
    v11 = v9;
    v9[4] = 0;
    v9[5] = v6;
    if ( v5 != 40 && v5 != 48 )
    {
      sub_164C220(v9 + 3);
      v10 = v111;
    }
    v11[6] = a1;
    v11[7] = 0;
    v11[2] = &unk_49EA488;
    sub_2208C80(v11, v10);
    v12 = *(_QWORD *)(a1 + 328);
    ++*(_QWORD *)(a1 + 344);
    *(_QWORD *)(v12 + 56) = v12;
    v5 = *(_QWORD *)(v5 + 8);
  }
  while ( v4 != v5 );
LABEL_15:
  v2 = a2;
LABEL_16:
  v13 = *(_QWORD *)(v2 + 16);
  v113 = 0;
  v114 = v117;
  s = v117;
  v119 = v122;
  v120 = v122;
  *(_QWORD *)v116 = 16;
  *(_DWORD *)&v116[8] = 0;
  v118 = 0;
  *(_QWORD *)v121 = 16;
  *(_DWORD *)&v121[8] = 0;
  v112 = v2 + 8;
  if ( v13 == v2 + 8 )
    goto LABEL_39;
  while ( 2 )
  {
    if ( !v13 )
      BUG();
    if ( (*(_BYTE *)(v13 - 24) & 0xFu) - 7 > 1 )
      goto LABEL_34;
    v14 = &v118;
    v110 = (_QWORD *)(v13 - 56);
    if ( (*(_BYTE *)(v13 + 24) & 1) != 0 )
      v14 = 0;
    if ( !(unsigned __int8)sub_13C13B0(a1, (_QWORD *)(v13 - 56), (__int64)&v113, (__int64)v14, 0) )
    {
      v17 = *(_QWORD **)(a1 + 32);
      if ( *(_QWORD **)(a1 + 40) != v17 )
        goto LABEL_43;
      v31 = &v17[*(unsigned int *)(a1 + 52)];
      v32 = *(_DWORD *)(a1 + 52);
      if ( v17 != v31 )
      {
        v33 = 0;
        while ( v110 != (_QWORD *)*v17 )
        {
          if ( *v17 == -2 )
            v33 = v17;
          if ( v31 == ++v17 )
          {
            if ( !v33 )
              goto LABEL_184;
            *v33 = v110;
            --*(_DWORD *)(a1 + 56);
            ++*(_QWORD *)(a1 + 24);
            goto LABEL_44;
          }
        }
        goto LABEL_44;
      }
LABEL_184:
      if ( v32 < *(_DWORD *)(a1 + 48) )
      {
        *(_DWORD *)(a1 + 52) = v32 + 1;
        *v31 = v110;
        ++*(_QWORD *)(a1 + 24);
      }
      else
      {
LABEL_43:
        sub_16CCBA0(a1 + 24, v110);
      }
LABEL_44:
      v18 = *(_QWORD *)(a1 + 328);
      v19 = (_QWORD *)sub_22077B0(64);
      v19[3] = 2;
      v20 = v19;
      v19[4] = 0;
      v19[5] = v110;
      if ( v13 != 40 && v13 != 48 )
        sub_164C220(v19 + 3);
      v20[6] = a1;
      v20[7] = 0;
      v20[2] = &unk_49EA488;
      sub_2208C80(v20, v18);
      v21 = *(_QWORD *)(a1 + 328);
      ++*(_QWORD *)(a1 + 344);
      *(_QWORD *)(v21 + 56) = v21;
      v22 = (char *)s;
      if ( s == v114 )
        v23 = (char *)s + 8 * *(unsigned int *)&v116[4];
      else
        v23 = (char *)s + 8 * *(unsigned int *)v116;
      if ( s != v23 )
      {
        while ( 1 )
        {
          v24 = *(_QWORD *)v22;
          v25 = v22;
          if ( *(_QWORD *)v22 < 0xFFFFFFFFFFFFFFFELL )
            break;
          v22 += 8;
          if ( v23 == v22 )
            goto LABEL_52;
        }
        if ( v23 != v22 )
        {
          v101 = a1 + 264;
          v34 = v124;
          if ( v125 == v124 )
            goto LABEL_105;
LABEL_81:
          sub_16CCBA0(&v123, v24);
          if ( v35 )
          {
            while ( 1 )
            {
LABEL_113:
              v103 = *(_QWORD *)(a1 + 328);
              v57 = (_QWORD *)sub_22077B0(64);
              v58 = v103;
              v57[5] = v24;
              v59 = v57;
              v57[3] = 2;
              v57[4] = 0;
              if ( v24 != 0 && v24 != -8 && v24 != -16 )
              {
                v104 = v57;
                sub_164C220(v57 + 3);
                v59 = v104;
              }
              v59[6] = a1;
              v59[7] = 0;
              v59[2] = &unk_49EA488;
              sub_2208C80(v59, v58);
              v60 = *(_QWORD *)(a1 + 328);
              ++*(_QWORD *)(a1 + 344);
              *(_QWORD *)(v60 + 56) = v60;
              v36 = *(_DWORD *)(a1 + 288);
              if ( !v36 )
                goto LABEL_117;
LABEL_83:
              v37 = *(_QWORD *)(a1 + 272);
              v38 = (v36 - 1) & (((unsigned int)v24 >> 4) ^ ((unsigned int)v24 >> 9));
              v39 = (_QWORD *)(v37 + 16LL * v38);
              v40 = *v39;
              if ( v24 != *v39 )
              {
                v69 = 1;
                v70 = 0;
                while ( v40 != -8 )
                {
                  if ( v70 || v40 != -16 )
                    v39 = v70;
                  v38 = (v36 - 1) & (v69 + v38);
                  v40 = *(_QWORD *)(v37 + 16LL * v38);
                  if ( v40 == v24 )
                  {
                    v39 = (_QWORD *)(v37 + 16LL * v38);
                    goto LABEL_84;
                  }
                  ++v69;
                  v70 = v39;
                  v39 = (_QWORD *)(v37 + 16LL * v38);
                }
                v71 = *(_DWORD *)(a1 + 280);
                if ( v70 )
                  v39 = v70;
                ++*(_QWORD *)(a1 + 264);
                v65 = v71 + 1;
                if ( 4 * (v71 + 1) >= 3 * v36 )
                  goto LABEL_118;
                if ( v36 - *(_DWORD *)(a1 + 284) - v65 <= v36 >> 3 )
                {
                  v105 = ((unsigned int)v24 >> 4) ^ ((unsigned int)v24 >> 9);
                  sub_13C4B40(v101, v36);
                  v72 = *(_DWORD *)(a1 + 288);
                  if ( !v72 )
                  {
LABEL_219:
                    ++*(_DWORD *)(a1 + 280);
                    BUG();
                  }
                  v73 = v72 - 1;
                  v74 = *(_QWORD *)(a1 + 272);
                  v75 = 1;
                  v76 = v73 & v105;
                  v65 = *(_DWORD *)(a1 + 280) + 1;
                  v68 = 0;
                  v39 = (_QWORD *)(v74 + 16LL * (v73 & v105));
                  v77 = *v39;
                  if ( *v39 != v24 )
                  {
                    while ( v77 != -8 )
                    {
                      if ( !v68 && v77 == -16 )
                        v68 = v39;
                      v76 = v73 & (v75 + v76);
                      v39 = (_QWORD *)(v74 + 16LL * v76);
                      v77 = *v39;
                      if ( *v39 == v24 )
                        goto LABEL_135;
                      ++v75;
                    }
                    goto LABEL_122;
                  }
                }
LABEL_135:
                *(_DWORD *)(a1 + 280) = v65;
                if ( *v39 != -8 )
                  --*(_DWORD *)(a1 + 284);
                *v39 = v24;
                v39[1] = 0;
              }
LABEL_84:
              sub_13C47F0(v39 + 1, (__int64)v110, 5);
              v41 = v25 + 8;
              if ( v25 + 8 == v23 )
                goto LABEL_87;
              while ( 1 )
              {
                v24 = *(_QWORD *)v41;
                v25 = v41;
                if ( *(_QWORD *)v41 < 0xFFFFFFFFFFFFFFFELL )
                  break;
                v41 += 8;
                if ( v23 == v41 )
                  goto LABEL_87;
              }
              if ( v23 == v41 )
              {
LABEL_87:
                if ( (*(_BYTE *)(v13 + 24) & 1) == 0 )
                  goto LABEL_88;
                goto LABEL_53;
              }
              v34 = v124;
              if ( v125 != v124 )
                goto LABEL_81;
LABEL_105:
              v55 = &v34[HIDWORD(v126)];
              if ( v34 != v55 )
                break;
LABEL_127:
              if ( HIDWORD(v126) >= (unsigned int)v126 )
                goto LABEL_81;
              ++HIDWORD(v126);
              *v55 = v24;
              ++v123;
            }
            v56 = 0;
            while ( *v34 != v24 )
            {
              if ( *v34 == -2 )
                v56 = v34;
              if ( v55 == ++v34 )
              {
                if ( !v56 )
                  goto LABEL_127;
                *v56 = v24;
                --v127;
                ++v123;
                goto LABEL_113;
              }
            }
          }
          v36 = *(_DWORD *)(a1 + 288);
          if ( v36 )
            goto LABEL_83;
LABEL_117:
          ++*(_QWORD *)(a1 + 264);
LABEL_118:
          sub_13C4B40(v101, 2 * v36);
          v61 = *(_DWORD *)(a1 + 288);
          if ( !v61 )
            goto LABEL_219;
          v62 = v61 - 1;
          v63 = *(_QWORD *)(a1 + 272);
          v64 = v62 & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
          v65 = *(_DWORD *)(a1 + 280) + 1;
          v39 = (_QWORD *)(v63 + 16LL * v64);
          v66 = *v39;
          if ( *v39 != v24 )
          {
            v67 = 1;
            v68 = 0;
            while ( v66 != -8 )
            {
              if ( !v68 && v66 == -16 )
                v68 = v39;
              v64 = v62 & (v67 + v64);
              v39 = (_QWORD *)(v63 + 16LL * v64);
              v66 = *v39;
              if ( *v39 == v24 )
                goto LABEL_135;
              ++v67;
            }
LABEL_122:
            if ( v68 )
              v39 = v68;
            goto LABEL_135;
          }
          goto LABEL_135;
        }
      }
LABEL_52:
      if ( (*(_BYTE *)(v13 + 24) & 1) != 0 )
        goto LABEL_53;
LABEL_88:
      v42 = (char *)v120;
      if ( v120 == v119 )
        v43 = (char *)v120 + 8 * *(unsigned int *)&v121[4];
      else
        v43 = (char *)v120 + 8 * *(unsigned int *)v121;
      if ( v120 == v43 )
        goto LABEL_53;
      while ( 1 )
      {
        v44 = *(_QWORD *)v42;
        v45 = v42;
        if ( *(_QWORD *)v42 < 0xFFFFFFFFFFFFFFFELL )
          break;
        v42 += 8;
        if ( v43 == v42 )
          goto LABEL_53;
      }
      if ( v42 == v43 )
      {
LABEL_53:
        if ( *(_BYTE *)(*(_QWORD *)(v13 - 32) + 8LL) == 15 )
          sub_13C4F40(a1, v110);
        goto LABEL_22;
      }
      v46 = v43;
      v102 = a1 + 264;
      v47 = v124;
      if ( v125 == v124 )
        goto LABEL_149;
LABEL_96:
      sub_16CCBA0(&v123, v44);
      if ( v48 )
      {
        while ( 1 )
        {
          v106 = *(_QWORD *)(a1 + 328);
          v80 = (_QWORD *)sub_22077B0(64);
          v81 = v106;
          v80[5] = v44;
          v82 = v80;
          v80[3] = 2;
          v80[4] = 0;
          if ( v44 != -8 && v44 != 0 && v44 != -16 )
          {
            v107 = v80;
            sub_164C220(v80 + 3);
            v82 = v107;
          }
          v82[6] = a1;
          v82[7] = 0;
          v82[2] = &unk_49EA488;
          sub_2208C80(v82, v81);
          v83 = *(_QWORD *)(a1 + 328);
          ++*(_QWORD *)(a1 + 344);
          *(_QWORD *)(v83 + 56) = v83;
          v49 = *(_DWORD *)(a1 + 288);
          if ( !v49 )
            break;
LABEL_98:
          v50 = *(_QWORD *)(a1 + 272);
          v51 = (v49 - 1) & (((unsigned int)v44 >> 9) ^ ((unsigned int)v44 >> 4));
          v52 = (_QWORD *)(v50 + 16LL * v51);
          v53 = *v52;
          if ( *v52 != v44 )
          {
            v92 = 1;
            v93 = 0;
            while ( v53 != -8 )
            {
              if ( !v93 && v53 == -16 )
                v93 = v52;
              v51 = (v49 - 1) & (v92 + v51);
              v52 = (_QWORD *)(v50 + 16LL * v51);
              v53 = *v52;
              if ( *v52 == v44 )
                goto LABEL_99;
              ++v92;
            }
            v94 = *(_DWORD *)(a1 + 280);
            if ( v93 )
              v52 = v93;
            ++*(_QWORD *)(a1 + 264);
            v88 = v94 + 1;
            if ( 4 * v88 >= 3 * v49 )
              goto LABEL_162;
            if ( v49 - *(_DWORD *)(a1 + 284) - v88 <= v49 >> 3 )
            {
              v108 = ((unsigned int)v44 >> 9) ^ ((unsigned int)v44 >> 4);
              sub_13C4B40(v102, v49);
              v95 = *(_DWORD *)(a1 + 288);
              if ( !v95 )
              {
LABEL_217:
                ++*(_DWORD *)(a1 + 280);
                BUG();
              }
              v96 = v95 - 1;
              v91 = 0;
              v97 = *(_QWORD *)(a1 + 272);
              v98 = 1;
              v99 = v96 & v108;
              v88 = *(_DWORD *)(a1 + 280) + 1;
              v52 = (_QWORD *)(v97 + 16LL * (v96 & v108));
              v100 = *v52;
              if ( *v52 != v44 )
              {
                while ( v100 != -8 )
                {
                  if ( !v91 && v100 == -16 )
                    v91 = v52;
                  v99 = v96 & (v98 + v99);
                  v52 = (_QWORD *)(v97 + 16LL * v99);
                  v100 = *v52;
                  if ( v44 == *v52 )
                    goto LABEL_181;
                  ++v98;
                }
                goto LABEL_166;
              }
            }
LABEL_181:
            *(_DWORD *)(a1 + 280) = v88;
            if ( *v52 != -8 )
              --*(_DWORD *)(a1 + 284);
            *v52 = v44;
            v52[1] = 0;
          }
LABEL_99:
          sub_13C47F0(v52 + 1, (__int64)v110, 6);
          v54 = v45 + 8;
          if ( v45 + 8 == v46 )
            goto LABEL_53;
          while ( 1 )
          {
            v44 = *(_QWORD *)v54;
            v45 = v54;
            if ( *(_QWORD *)v54 < 0xFFFFFFFFFFFFFFFELL )
              break;
            v54 += 8;
            if ( v46 == v54 )
              goto LABEL_53;
          }
          if ( v46 == v54 )
            goto LABEL_53;
          v47 = v124;
          if ( v125 != v124 )
            goto LABEL_96;
LABEL_149:
          v78 = &v47[HIDWORD(v126)];
          if ( v47 == v78 )
            goto LABEL_173;
          v79 = 0;
          do
          {
            if ( *v47 == v44 )
              goto LABEL_97;
            if ( *v47 == -2 )
              v79 = v47;
            ++v47;
          }
          while ( v78 != v47 );
          if ( !v79 )
          {
LABEL_173:
            if ( HIDWORD(v126) >= (unsigned int)v126 )
              goto LABEL_96;
            ++HIDWORD(v126);
            *v78 = v44;
            ++v123;
          }
          else
          {
            *v79 = v44;
            --v127;
            ++v123;
          }
        }
      }
      else
      {
LABEL_97:
        v49 = *(_DWORD *)(a1 + 288);
        if ( v49 )
          goto LABEL_98;
      }
      ++*(_QWORD *)(a1 + 264);
LABEL_162:
      sub_13C4B40(v102, 2 * v49);
      v84 = *(_DWORD *)(a1 + 288);
      if ( !v84 )
        goto LABEL_217;
      v85 = v84 - 1;
      v86 = *(_QWORD *)(a1 + 272);
      v87 = v85 & (((unsigned int)v44 >> 9) ^ ((unsigned int)v44 >> 4));
      v88 = *(_DWORD *)(a1 + 280) + 1;
      v52 = (_QWORD *)(v86 + 16LL * v87);
      v89 = *v52;
      if ( *v52 != v44 )
      {
        v90 = 1;
        v91 = 0;
        while ( v89 != -8 )
        {
          if ( !v91 && v89 == -16 )
            v91 = v52;
          v87 = v85 & (v90 + v87);
          v52 = (_QWORD *)(v86 + 16LL * v87);
          v89 = *v52;
          if ( *v52 == v44 )
            goto LABEL_181;
          ++v90;
        }
LABEL_166:
        if ( v91 )
          v52 = v91;
        goto LABEL_181;
      }
      goto LABEL_181;
    }
LABEL_22:
    ++v113;
    if ( s == v114 )
      goto LABEL_27;
    v15 = 4 * (*(_DWORD *)&v116[4] - *(_DWORD *)&v116[8]);
    if ( v15 < 0x20 )
      v15 = 32;
    if ( *(_DWORD *)v116 > v15 )
    {
      sub_16CC920(&v113);
    }
    else
    {
      memset(s, -1, 8LL * *(unsigned int *)v116);
LABEL_27:
      *(_QWORD *)&v116[4] = 0;
    }
    ++v118;
    if ( v120 == v119 )
    {
LABEL_33:
      *(_QWORD *)&v121[4] = 0;
    }
    else
    {
      v16 = 4 * (*(_DWORD *)&v121[4] - *(_DWORD *)&v121[8]);
      if ( v16 < 0x20 )
        v16 = 32;
      if ( *(_DWORD *)v121 <= v16 )
      {
        memset(v120, -1, 8LL * *(unsigned int *)v121);
        goto LABEL_33;
      }
      sub_16CC920(&v118);
    }
LABEL_34:
    v13 = *(_QWORD *)(v13 + 8);
    if ( v112 != v13 )
      continue;
    break;
  }
  if ( v120 != v119 )
    _libc_free((unsigned __int64)v120);
  if ( s != v114 )
    _libc_free((unsigned __int64)s);
LABEL_39:
  if ( v125 != v124 )
    _libc_free((unsigned __int64)v125);
}
