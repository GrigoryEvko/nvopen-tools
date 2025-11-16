// Function: sub_1A85C10
// Address: 0x1a85c10
//
__int64 __fastcall sub_1A85C10(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  unsigned int v10; // eax
  unsigned int v11; // r13d
  __int64 *v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rax
  int v17; // eax
  __int64 v18; // rdx
  __int64 v19; // rcx
  int v20; // r8d
  int v21; // r9d
  __int64 v22; // r14
  __int64 v23; // rbx
  __int64 v24; // r13
  char v25; // al
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // rsi
  __int64 *v30; // r12
  __int64 *v31; // r14
  __int64 v32; // rsi
  char *v33; // r12
  char *v34; // r15
  char *v35; // rbx
  unsigned int v36; // esi
  __int64 v37; // rax
  int v38; // r10d
  __int64 *v39; // r9
  unsigned int v40; // edi
  __int64 *v41; // rcx
  __int64 v42; // rdx
  __int64 v43; // r13
  unsigned int v44; // edx
  __int64 v45; // rsi
  int v46; // eax
  __int64 *v47; // rax
  _BYTE *v48; // rsi
  __int64 v49; // r8
  __int64 v50; // rcx
  __int64 *v51; // rax
  __int64 v52; // rdx
  __int64 v53; // rbx
  _BYTE *v54; // rdi
  __int64 v55; // r12
  unsigned int v56; // eax
  __int64 v57; // rsi
  int v58; // eax
  int v59; // eax
  __int64 *v60; // r11
  int v61; // ebx
  __int64 v62; // rsi
  int v63; // edi
  unsigned int v64; // r14d
  __int64 v65; // rax
  int v66; // eax
  __int64 v67; // r13
  unsigned int v68; // r11d
  __int64 *v69; // rax
  __int64 v70; // rsi
  int v71; // eax
  unsigned int v72; // esi
  __int64 *v73; // rax
  __int64 v74; // rbx
  __int64 *v75; // rax
  unsigned int v76; // r11d
  __int64 v77; // r10
  int v78; // r11d
  __int64 **v79; // r10
  _BYTE *v80; // rsi
  double v81; // xmm4_8
  double v82; // xmm5_8
  char *v83; // rbx
  char *v84; // r12
  __int64 v85; // rax
  __int64 v86; // rax
  int i; // eax
  __int64 v88; // rax
  __int64 *v89; // rsi
  __int64 *v90; // r14
  unsigned int v91; // ecx
  __int64 v92; // rax
  int v93; // r15d
  __int64 v94; // rax
  unsigned __int8 v95; // al
  char v96; // al
  char v97; // al
  __int64 v98; // rdi
  int v99; // eax
  __int64 v100; // rax
  unsigned int v101; // ebx
  char *v102; // rdi
  __int64 v103; // rax
  bool v104; // zf
  int v105; // eax
  int v106; // r8d
  __int64 *v107; // rdi
  unsigned int v108; // r14d
  __int64 v109; // rcx
  unsigned int v110; // r14d
  char v111; // al
  int v112; // r10d
  unsigned int v113; // r13d
  __int64 v114; // rsi
  int v115; // eax
  int v116; // eax
  int v117; // r10d
  int v118; // r11d
  __int64 v119; // rsi
  __int64 *v120; // rdi
  unsigned int v121; // r14d
  __int64 v122; // rsi
  int v123; // r11d
  __int64 **v124; // rsi
  int v125; // edi
  int v126; // r10d
  __int64 v127; // rax
  char v128; // dl
  __int64 v129; // rax
  int v130; // r10d
  __int64 *v131; // r8
  int v132; // esi
  __int64 *v133; // r10
  unsigned int v134; // eax
  int v135; // r10d
  int v136; // [rsp+Ch] [rbp-F4h]
  unsigned __int8 v137; // [rsp+Ch] [rbp-F4h]
  __int64 v138; // [rsp+18h] [rbp-E8h]
  char *v140; // [rsp+30h] [rbp-D0h] BYREF
  char *v141; // [rsp+38h] [rbp-C8h]
  char *v142; // [rsp+40h] [rbp-C0h]
  __int64 v143; // [rsp+50h] [rbp-B0h] BYREF
  __int64 v144; // [rsp+58h] [rbp-A8h]
  __int64 v145; // [rsp+60h] [rbp-A0h]
  unsigned int v146; // [rsp+68h] [rbp-98h]
  __int64 *v147; // [rsp+70h] [rbp-90h] BYREF
  __int64 v148; // [rsp+78h] [rbp-88h]
  _QWORD v149[2]; // [rsp+80h] [rbp-80h] BYREF
  __int64 v150; // [rsp+90h] [rbp-70h] BYREF
  __int64 v151; // [rsp+98h] [rbp-68h]
  __int64 v152; // [rsp+A0h] [rbp-60h] BYREF
  __int64 v153; // [rsp+A8h] [rbp-58h]
  _BYTE *v154; // [rsp+B0h] [rbp-50h] BYREF
  _BYTE *v155; // [rsp+B8h] [rbp-48h]
  _BYTE *v156; // [rsp+C0h] [rbp-40h]

  v10 = sub_1636880(a1, a2);
  if ( (_BYTE)v10 )
    return 0;
  v13 = *(__int64 **)(a1 + 8);
  v11 = v10;
  v14 = *v13;
  v15 = v13[1];
  if ( v14 == v15 )
LABEL_387:
    BUG();
  while ( *(_UNKNOWN **)v14 != &unk_4F9D3C0 )
  {
    v14 += 16;
    if ( v15 == v14 )
      goto LABEL_387;
  }
  v16 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v14 + 8) + 104LL))(*(_QWORD *)(v14 + 8), &unk_4F9D3C0);
  v138 = sub_14A4050(v16, a2);
  v17 = sub_14A29A0(v138);
  *(_DWORD *)(a1 + 156) = v17;
  if ( v17 != -1 )
  {
    v22 = a2 + 72;
    v23 = *(_QWORD *)(a2 + 80);
    v143 = 0;
    v144 = 0;
    v145 = 0;
    v147 = 0;
    v148 = 0;
    v149[0] = 0;
    v149[1] = 0;
    if ( a2 + 72 == v23 )
    {
      v24 = 0;
    }
    else
    {
      if ( !v23 )
        BUG();
      while ( 1 )
      {
        v24 = *(_QWORD *)(v23 + 24);
        if ( v24 != v23 + 16 )
          break;
        v23 = *(_QWORD *)(v23 + 8);
        if ( v22 == v23 )
          break;
        if ( !v23 )
          BUG();
      }
    }
    while ( v23 != v22 )
    {
      if ( !v24 )
        BUG();
      v25 = *(_BYTE *)(v24 - 8);
      switch ( v25 )
      {
        case '8':
          if ( *(_BYTE *)(*(_QWORD *)(v24 - 24) + 8LL) != 16 )
            sub_1A82EB0(
              a1,
              *(_QWORD *)(v24 - 24 * ((*(_DWORD *)(v24 - 4) & 0xFFFFFFF) + 1LL)),
              (__int64)&v143,
              (__int64)&v147);
          break;
        case '6':
        case '7':
          goto LABEL_28;
        case ';':
          sub_1A82EB0(a1, *(_QWORD *)(v24 - 72), (__int64)&v143, (__int64)&v147);
          break;
        case ':':
          sub_1A82EB0(a1, *(_QWORD *)(v24 - 96), (__int64)&v143, (__int64)&v147);
          break;
        case 'N':
          v127 = *(_QWORD *)(v24 - 48);
          if ( *(_BYTE *)(v127 + 16) )
            break;
          v128 = *(_BYTE *)(v127 + 33);
          if ( (v128 & 0x20) == 0 )
            break;
          v19 = (unsigned int)(*(_DWORD *)(v127 + 36) - 133);
          if ( (unsigned int)v19 <= 4 && (v20 = 1, v19 = (1LL << (*(_BYTE *)(v127 + 36) + 123)) & 0x15, (_DWORD)v19) )
          {
            sub_1A82EB0(
              a1,
              *(_QWORD *)(v24 - 24LL * (*(_DWORD *)(v24 - 4) & 0xFFFFFFF) - 24),
              (__int64)&v143,
              (__int64)&v147);
            v129 = *(_QWORD *)(v24 - 48);
            if ( *(_BYTE *)(v129 + 16) )
              BUG();
            if ( (*(_DWORD *)(v129 + 36) & 0xFFFFFFFD) == 0x85 )
              sub_1A82EB0(
                a1,
                *(_QWORD *)(v24 - 24 + 24 * (1LL - (*(_DWORD *)(v24 - 4) & 0xFFFFFFF))),
                (__int64)&v143,
                (__int64)&v147);
          }
          else if ( (v128 & 0x20) != 0 )
          {
            v134 = *(_DWORD *)(v127 + 36);
            if ( v134 > 0x19E )
            {
              if ( v134 - 452 > 2 )
                break;
            }
            else if ( v134 <= 0x19C && v134 != 144 )
            {
              break;
            }
            sub_1A82EB0(
              a1,
              *(_QWORD *)(v24 - 24LL * (*(_DWORD *)(v24 - 4) & 0xFFFFFFF) - 24),
              (__int64)&v143,
              (__int64)&v147);
          }
          break;
        case 'K':
          v119 = *(_QWORD *)(v24 - 72);
          if ( *(_BYTE *)(*(_QWORD *)v119 + 8LL) == 15 )
          {
            sub_1A82EB0(a1, v119, (__int64)&v143, (__int64)&v147);
            sub_1A82EB0(a1, *(_QWORD *)(v24 - 48), (__int64)&v143, (__int64)&v147);
          }
          break;
        default:
          if ( v25 != 72 || *(_BYTE *)(*(_QWORD *)(v24 - 24) + 8LL) == 16 )
            break;
LABEL_28:
          sub_1A82EB0(a1, *(_QWORD *)(v24 - 48), (__int64)&v143, (__int64)&v147);
          break;
      }
      v24 = *(_QWORD *)(v24 + 8);
      v18 = 0;
      while ( 1 )
      {
        v26 = v23 - 24;
        if ( !v23 )
          v26 = 0;
        if ( v24 != v26 + 40 )
          break;
        v23 = *(_QWORD *)(v23 + 8);
        if ( v22 == v23 )
          break;
        if ( !v23 )
          BUG();
        v24 = *(_QWORD *)(v23 + 24);
      }
    }
    v140 = 0;
    v27 = v144;
    v141 = 0;
    v142 = 0;
    if ( v144 != v143 )
    {
      while ( 1 )
      {
        v29 = *(_QWORD *)(v27 - 16);
        if ( *(_BYTE *)(v27 - 8) )
          break;
        *(_BYTE *)(v27 - 8) = 1;
        sub_1A826A0(&v150, v29, v18, v19, v20, v21);
        v18 = (unsigned int)v151;
        v30 = (__int64 *)(v150 + 8LL * (unsigned int)v151);
        if ( (__int64 *)v150 != v30 )
        {
          v31 = (__int64 *)v150;
          do
          {
            v32 = *v31++;
            sub_1A82EB0(a1, v32, (__int64)&v143, (__int64)&v147);
          }
          while ( v30 != v31 );
          v30 = (__int64 *)v150;
        }
        if ( v30 == &v152 )
        {
LABEL_35:
          v27 = v144;
          if ( v144 == v143 )
            goto LABEL_43;
        }
        else
        {
          _libc_free((unsigned __int64)v30);
          v27 = v144;
          if ( v144 == v143 )
            goto LABEL_43;
        }
      }
      v28 = *(_QWORD *)v29;
      if ( *(_BYTE *)(*(_QWORD *)v29 + 8LL) == 16 )
        v28 = **(_QWORD **)(v28 + 16);
      if ( *(_DWORD *)(a1 + 156) == *(_DWORD *)(v28 + 8) >> 8 )
      {
        v150 = 6;
        v151 = 0;
        v152 = v29;
        if ( v29 != -8 && v29 != -16 )
          sub_164C220((__int64)&v150);
        v102 = v141;
        if ( v141 == v142 )
        {
          sub_1893930((unsigned __int64 **)&v140, v141, &v150);
        }
        else
        {
          if ( v141 )
          {
            *(_QWORD *)v141 = 6;
            *((_QWORD *)v102 + 1) = 0;
            v103 = v152;
            v104 = v152 == -8;
            *((_QWORD *)v102 + 2) = v152;
            if ( v103 != 0 && !v104 && v103 != -16 )
              sub_1649AC0((unsigned __int64 *)v102, v150 & 0xFFFFFFFFFFFFFFF8LL);
            v102 = v141;
          }
          v141 = v102 + 24;
        }
        LOBYTE(v19) = v152 != 0;
        LOBYTE(v18) = v152 != -8;
        if ( ((unsigned __int8)v18 & (v152 != 0)) != 0 && v152 != -16 )
          sub_1649B30(&v150);
      }
      v144 -= 16;
      goto LABEL_35;
    }
LABEL_43:
    j___libc_free_0(v148);
    if ( v143 )
      j_j___libc_free_0(v143, v145 - v143);
    v33 = v140;
    v34 = v141;
    v143 = 0;
    v144 = 0;
    v145 = 0;
    v146 = 0;
    v150 = 0;
    v151 = 0;
    v152 = 0;
    v153 = 0;
    v154 = 0;
    v155 = 0;
    v156 = 0;
    if ( v140 == v141 )
      goto LABEL_122;
    v35 = v140;
    v36 = 0;
    v37 = 0;
    while ( 1 )
    {
      v43 = *((_QWORD *)v35 + 2);
      if ( !v36 )
        break;
      v38 = 1;
      v39 = 0;
      v40 = (v36 - 1) & (((unsigned int)v43 >> 9) ^ ((unsigned int)v43 >> 4));
      v41 = (__int64 *)(v37 + 8LL * v40);
      v42 = *v41;
      if ( v43 == *v41 )
        goto LABEL_48;
      while ( v42 != -8 )
      {
        if ( v39 || v42 != -16 )
          v41 = v39;
        LODWORD(v39) = v38 + 1;
        v40 = (v36 - 1) & (v38 + v40);
        v42 = *(_QWORD *)(v37 + 8LL * v40);
        if ( v43 == v42 )
          goto LABEL_48;
        ++v38;
        v39 = v41;
        v41 = (__int64 *)(v37 + 8LL * v40);
      }
      if ( !v39 )
        v39 = v41;
      ++v150;
      v46 = v152 + 1;
      if ( 4 * ((int)v152 + 1) >= 3 * v36 )
        goto LABEL_52;
      if ( v36 - (v46 + HIDWORD(v152)) <= v36 >> 3 )
      {
        sub_1353F00((__int64)&v150, v36);
        if ( !(_DWORD)v153 )
        {
LABEL_386:
          LODWORD(v152) = v152 + 1;
          BUG();
        }
        v106 = 1;
        v107 = 0;
        v108 = (v153 - 1) & (((unsigned int)v43 >> 9) ^ ((unsigned int)v43 >> 4));
        v39 = (__int64 *)(v151 + 8LL * v108);
        v109 = *v39;
        v46 = v152 + 1;
        if ( *v39 != v43 )
        {
          while ( v109 != -8 )
          {
            if ( v109 == -16 && !v107 )
              v107 = v39;
            v108 = (v153 - 1) & (v106 + v108);
            v39 = (__int64 *)(v151 + 8LL * v108);
            v109 = *v39;
            if ( v43 == *v39 )
              goto LABEL_54;
            ++v106;
          }
          if ( v107 )
            v39 = v107;
        }
      }
LABEL_54:
      LODWORD(v152) = v46;
      if ( *v39 != -8 )
        --HIDWORD(v152);
      *v39 = v43;
      v47 = (__int64 *)*((_QWORD *)v35 + 2);
      v48 = v155;
      v147 = v47;
      if ( v155 == v156 )
      {
        sub_12879C0((__int64)&v154, v155, &v147);
LABEL_48:
        v35 += 24;
        if ( v34 == v35 )
          goto LABEL_63;
        goto LABEL_49;
      }
      if ( v155 )
      {
        *(_QWORD *)v155 = v47;
        v48 = v155;
      }
      v35 += 24;
      v155 = v48 + 8;
      if ( v34 == v35 )
      {
LABEL_63:
        while ( 1 )
        {
          v53 = *((_QWORD *)v33 + 2);
          if ( !v146 )
            break;
          LODWORD(v49) = v146 - 1;
          v50 = (v146 - 1) & (((unsigned int)v53 >> 9) ^ ((unsigned int)v53 >> 4));
          v51 = (__int64 *)(v144 + 16 * v50);
          v52 = *v51;
          if ( v53 == *v51 )
          {
LABEL_62:
            v33 += 24;
            *((_DWORD *)v51 + 2) = -1;
            if ( v34 == v33 )
              goto LABEL_70;
          }
          else
          {
            v112 = 1;
            v39 = 0;
            while ( v52 != -8 )
            {
              if ( v52 == -16 && !v39 )
                v39 = v51;
              v50 = (unsigned int)v49 & (v112 + (_DWORD)v50);
              v51 = (__int64 *)(v144 + 16LL * (unsigned int)v50);
              v52 = *v51;
              if ( v53 == *v51 )
                goto LABEL_62;
              ++v112;
            }
            if ( v39 )
              v51 = v39;
            ++v143;
            v50 = (unsigned int)(v145 + 1);
            if ( 4 * (int)v50 < 3 * v146 )
            {
              v52 = v146 - HIDWORD(v145) - (unsigned int)v50;
              if ( (unsigned int)v52 <= v146 >> 3 )
              {
                sub_1542080((__int64)&v143, v146);
                if ( !v146 )
                {
LABEL_383:
                  LODWORD(v145) = v145 + 1;
                  BUG();
                }
                v52 = v146 - 1;
                v49 = 0;
                v113 = v52 & (((unsigned int)v53 >> 9) ^ ((unsigned int)v53 >> 4));
                LODWORD(v39) = 1;
                v50 = (unsigned int)(v145 + 1);
                v51 = (__int64 *)(v144 + 16LL * v113);
                v114 = *v51;
                if ( v53 != *v51 )
                {
                  while ( v114 != -8 )
                  {
                    if ( v114 == -16 && !v49 )
                      v49 = (__int64)v51;
                    v135 = (_DWORD)v39 + 1;
                    LODWORD(v39) = v52 & (v113 + (_DWORD)v39);
                    v113 = (unsigned int)v39;
                    v51 = (__int64 *)(v144 + 16LL * (unsigned int)v39);
                    v114 = *v51;
                    if ( v53 == *v51 )
                      goto LABEL_67;
                    LODWORD(v39) = v135;
                  }
                  if ( v49 )
                    v51 = (__int64 *)v49;
                }
              }
              goto LABEL_67;
            }
LABEL_65:
            sub_1542080((__int64)&v143, 2 * v146);
            if ( !v146 )
              goto LABEL_383;
            v50 = (unsigned int)(v145 + 1);
            v52 = (v146 - 1) & (((unsigned int)v53 >> 9) ^ ((unsigned int)v53 >> 4));
            v51 = (__int64 *)(v144 + 16 * v52);
            v49 = *v51;
            if ( v53 != *v51 )
            {
              v126 = 1;
              v39 = 0;
              while ( v49 != -8 )
              {
                if ( !v39 && v49 == -16 )
                  v39 = v51;
                v52 = (v146 - 1) & ((_DWORD)v52 + v126);
                v51 = (__int64 *)(v144 + 16 * v52);
                v49 = *v51;
                if ( v53 == *v51 )
                  goto LABEL_67;
                ++v126;
              }
              if ( v39 )
                v51 = v39;
            }
LABEL_67:
            LODWORD(v145) = v50;
            if ( *v51 != -8 )
              --HIDWORD(v145);
            v33 += 24;
            *((_DWORD *)v51 + 2) = 0;
            *v51 = v53;
            *((_DWORD *)v51 + 2) = -1;
            if ( v34 == v33 )
            {
LABEL_70:
              v54 = v155;
              if ( v155 == v154 )
                goto LABEL_120;
              while ( 1 )
              {
                v55 = *((_QWORD *)v54 - 1);
                if ( (_DWORD)v153 )
                {
                  v52 = (unsigned int)(v153 - 1);
                  v56 = v52 & (((unsigned int)v55 >> 9) ^ ((unsigned int)v55 >> 4));
                  v50 = v151 + 8LL * v56;
                  v57 = *(_QWORD *)v50;
                  if ( v55 == *(_QWORD *)v50 )
                  {
LABEL_73:
                    *(_QWORD *)v50 = -16;
                    LODWORD(v152) = v152 - 1;
                    ++HIDWORD(v152);
                  }
                  else
                  {
                    v50 = 1;
                    while ( v57 != -8 )
                    {
                      LODWORD(v49) = v50 + 1;
                      v56 = v52 & (v50 + v56);
                      v50 = v151 + 8LL * v56;
                      v57 = *(_QWORD *)v50;
                      if ( v55 == *(_QWORD *)v50 )
                        goto LABEL_73;
                      v50 = (unsigned int)v49;
                    }
                  }
                }
                v54 = v155 - 8;
                v155 -= 8;
                v58 = *(unsigned __int8 *)(v55 + 16);
                if ( (unsigned __int8)v58 <= 0x17u )
                  v59 = *(unsigned __int16 *)(v55 + 18);
                else
                  v59 = v58 - 24;
                if ( v59 != 55 )
                {
                  sub_1A826A0(&v147, v55, v52, v50, v49, (int)v39);
                  v60 = &v147[(unsigned int)v148];
                  if ( v147 == v60 )
                  {
                    v61 = -1;
                  }
                  else
                  {
                    v50 = (__int64)v147;
                    v61 = -1;
                    v62 = *v147;
                    LODWORD(v39) = v146;
                    v63 = *(_DWORD *)(a1 + 156);
                    v64 = v146 - 1;
                    if ( !v146 )
                      goto LABEL_136;
LABEL_79:
                    v52 = v64 & (((unsigned int)v62 >> 9) ^ ((unsigned int)v62 >> 4));
                    v65 = v144 + 16 * v52;
                    v49 = *(_QWORD *)v65;
                    if ( v62 != *(_QWORD *)v65 )
                    {
                      for ( i = 1; ; i = v136 )
                      {
                        if ( v49 == -8 )
                          goto LABEL_136;
                        v52 = v64 & (i + (_DWORD)v52);
                        v136 = i + 1;
                        v65 = v144 + 16LL * (unsigned int)v52;
                        v49 = *(_QWORD *)v65;
                        if ( v62 == *(_QWORD *)v65 )
                          break;
                      }
                    }
                    if ( v144 + 16LL * v146 != v65 )
                    {
                      v66 = *(_DWORD *)(v65 + 8);
                      goto LABEL_82;
                    }
                    while ( 1 )
                    {
LABEL_136:
                      v86 = *(_QWORD *)v62;
                      if ( *(_BYTE *)(*(_QWORD *)v62 + 8LL) == 16 )
                        v86 = **(_QWORD **)(v86 + 16);
                      v66 = *(_DWORD *)(v86 + 8) >> 8;
LABEL_82:
                      if ( v63 == v66 || v63 == v61 )
                        break;
                      if ( v61 == -1 )
                      {
                        v61 = v66;
                      }
                      else if ( v61 != v66 && v66 != -1 )
                      {
                        break;
                      }
                      v50 += 8;
                      if ( v60 == (__int64 *)v50 )
                        goto LABEL_88;
                      v62 = *(_QWORD *)v50;
                      if ( v146 )
                        goto LABEL_79;
                    }
                    v61 = *(_DWORD *)(a1 + 156);
                  }
LABEL_88:
                  if ( v147 != v149 )
                    _libc_free((unsigned __int64)v147);
                  v67 = v144;
                  v68 = v146;
                  goto LABEL_91;
                }
                if ( (*(_BYTE *)(v55 + 23) & 0x40) != 0 )
                  v88 = *(_QWORD *)(v55 - 8);
                else
                  v88 = v55 - 24LL * (*(_DWORD *)(v55 + 20) & 0xFFFFFFF);
                v68 = v146;
                v89 = *(__int64 **)(v88 + 24);
                v90 = *(__int64 **)(v88 + 48);
                v67 = v144;
                if ( !v146 )
                  goto LABEL_168;
                v52 = v146 - 1;
                v91 = v52 & (((unsigned int)v89 >> 9) ^ ((unsigned int)v89 >> 4));
                v92 = v144 + 16LL * v91;
                v49 = *(_QWORD *)v92;
                if ( v89 != *(__int64 **)v92 )
                  break;
LABEL_147:
                v50 = v144 + 16LL * v146;
                if ( v50 != v92 )
                {
                  v93 = *(_DWORD *)(v92 + 8);
                  goto LABEL_149;
                }
                v50 = *v89;
                if ( *(_BYTE *)(*v89 + 8) == 16 )
                  goto LABEL_245;
                v93 = *(_DWORD *)(v50 + 8) >> 8;
LABEL_341:
                v50 = v92;
LABEL_149:
                LODWORD(v49) = v52 & (((unsigned int)v90 >> 9) ^ ((unsigned int)v90 >> 4));
                v94 = v144 + 16LL * (unsigned int)v49;
                v39 = *(__int64 **)v94;
                if ( v90 == *(__int64 **)v94 )
                {
LABEL_150:
                  if ( v50 != v94 )
                  {
                    v61 = *(_DWORD *)(v94 + 8);
                    v95 = *((_BYTE *)v90 + 16);
                    if ( *((_BYTE *)v89 + 16) <= 0x10u )
                      goto LABEL_152;
                    goto LABEL_173;
                  }
                }
                else
                {
                  v116 = 1;
                  while ( v39 != (__int64 *)-8LL )
                  {
                    v117 = v116 + 1;
                    LODWORD(v49) = v52 & (v116 + v49);
                    v94 = v144 + 16LL * (unsigned int)v49;
                    v39 = *(__int64 **)v94;
                    if ( v90 == *(__int64 **)v94 )
                      goto LABEL_150;
                    v116 = v117;
                  }
                }
LABEL_170:
                v100 = *v90;
                if ( *(_BYTE *)(*v90 + 8) == 16 )
                  v100 = **(_QWORD **)(v100 + 16);
                v101 = *(_DWORD *)(v100 + 8);
                v95 = *((_BYTE *)v90 + 16);
                v61 = v101 >> 8;
                if ( *((_BYTE *)v89 + 16) <= 0x10u )
                {
LABEL_152:
                  if ( v95 > 0x10u )
                  {
                    LOBYTE(v50) = v61 == -1;
                    if ( v61 == -1 )
                      goto LABEL_119;
                    v111 = sub_1A825E0(a1, v89, v61, v50);
                    v50 = v61 == -1;
                    if ( v111 )
                      goto LABEL_91;
                    goto LABEL_195;
                  }
                  if ( v93 == -1 )
                    goto LABEL_119;
                  LOBYTE(v50) = v61 == -1;
                  if ( v61 == -1 )
                    goto LABEL_119;
                  v96 = sub_1A825E0(a1, v89, v61, v50);
                  v50 = v61 == -1;
                  if ( v96 )
                    goto LABEL_91;
                  goto LABEL_156;
                }
LABEL_173:
                if ( v95 > 0x10u )
                {
                  LOBYTE(v50) = v61 == -1;
                  goto LABEL_195;
                }
                if ( v93 == -1 )
                  goto LABEL_119;
                LOBYTE(v50) = v61 == -1;
LABEL_156:
                v137 = v50;
                v97 = sub_1A825E0(a1, v90, v93, v50);
                v50 = v137;
                if ( v97 )
                  goto LABEL_157;
LABEL_195:
                v105 = *(_DWORD *)(a1 + 156);
                if ( v61 == v105 || v105 == v93 )
                {
                  v61 = *(_DWORD *)(a1 + 156);
                }
                else if ( v93 != -1 )
                {
                  if ( v61 == v93 || (v61 = *(_DWORD *)(a1 + 156), (_BYTE)v50) )
LABEL_157:
                    v61 = v93;
                }
LABEL_91:
                if ( v68 )
                {
                  v52 = v68 - 1;
                  v50 = (unsigned int)v52 & (((unsigned int)v55 >> 9) ^ ((unsigned int)v55 >> 4));
                  v69 = (__int64 *)(v67 + 16 * v50);
                  v70 = *v69;
                  if ( v55 == *v69 )
                  {
LABEL_93:
                    v71 = *((_DWORD *)v69 + 2);
                  }
                  else
                  {
                    v115 = 1;
                    while ( v70 != -8 )
                    {
                      v125 = v115 + 1;
                      v50 = (unsigned int)v52 & (v115 + (_DWORD)v50);
                      v69 = (__int64 *)(v67 + 16LL * (unsigned int)v50);
                      v70 = *v69;
                      if ( v55 == *v69 )
                        goto LABEL_93;
                      v115 = v125;
                    }
                    v71 = 0;
                  }
                  if ( v71 == v61 )
                    goto LABEL_118;
                  v72 = v52 & (((unsigned int)v55 >> 9) ^ ((unsigned int)v55 >> 4));
                  v73 = (__int64 *)(v67 + 16LL * v72);
                  v50 = *v73;
                  if ( v55 == *v73 )
                    goto LABEL_96;
                  LODWORD(v49) = 1;
                  v120 = 0;
                  while ( v50 != -8 )
                  {
                    if ( !v120 && v50 == -16 )
                      v120 = v73;
                    LODWORD(v39) = v49 + 1;
                    v72 = v52 & (v49 + v72);
                    v73 = (__int64 *)(v67 + 16LL * v72);
                    v50 = *v73;
                    if ( v55 == *v73 )
                      goto LABEL_96;
                    LODWORD(v49) = v49 + 1;
                  }
                  if ( v120 )
                    v73 = v120;
                  ++v143;
                  v52 = (unsigned int)(v145 + 1);
                  if ( 4 * (int)v52 >= 3 * v68 )
                  {
LABEL_161:
                    sub_1542080((__int64)&v143, 2 * v68);
                    if ( v146 )
                    {
                      LODWORD(v49) = v146 - 1;
                      LODWORD(v39) = v144;
                      v50 = (v146 - 1) & (((unsigned int)v55 >> 9) ^ ((unsigned int)v55 >> 4));
                      v52 = (unsigned int)(v145 + 1);
                      v73 = (__int64 *)(v144 + 16 * v50);
                      v98 = *v73;
                      if ( v55 != *v73 )
                      {
                        v132 = 1;
                        v133 = 0;
                        while ( v98 != -8 )
                        {
                          if ( !v133 && v98 == -16 )
                            v133 = v73;
                          v50 = (unsigned int)v49 & (v132 + (_DWORD)v50);
                          v73 = (__int64 *)(v144 + 16LL * (unsigned int)v50);
                          v98 = *v73;
                          if ( v55 == *v73 )
                            goto LABEL_163;
                          ++v132;
                        }
                        if ( v133 )
                          v73 = v133;
                      }
                      goto LABEL_163;
                    }
                    goto LABEL_384;
                  }
                  v50 = v68 >> 3;
                  if ( v68 - ((_DWORD)v52 + HIDWORD(v145)) <= (unsigned int)v50 )
                  {
                    sub_1542080((__int64)&v143, v68);
                    if ( v146 )
                    {
                      LODWORD(v49) = v144;
                      v39 = 0;
                      v121 = (v146 - 1) & (((unsigned int)v55 >> 9) ^ ((unsigned int)v55 >> 4));
                      v52 = (unsigned int)(v145 + 1);
                      v50 = 1;
                      v73 = (__int64 *)(v144 + 16LL * v121);
                      v122 = *v73;
                      if ( v55 != *v73 )
                      {
                        while ( v122 != -8 )
                        {
                          if ( !v39 && v122 == -16 )
                            v39 = v73;
                          v121 = (v146 - 1) & (v50 + v121);
                          v73 = (__int64 *)(v144 + 16LL * v121);
                          v122 = *v73;
                          if ( v55 == *v73 )
                            goto LABEL_163;
                          v50 = (unsigned int)(v50 + 1);
                        }
                        if ( v39 )
                          v73 = v39;
                      }
                      goto LABEL_163;
                    }
LABEL_384:
                    LODWORD(v145) = v145 + 1;
                    BUG();
                  }
LABEL_163:
                  LODWORD(v145) = v52;
                  if ( *v73 != -8 )
                    --HIDWORD(v145);
                  *v73 = v55;
                  *((_DWORD *)v73 + 2) = 0;
LABEL_96:
                  *((_DWORD *)v73 + 2) = v61;
                  v74 = *(_QWORD *)(v55 + 8);
                  if ( !v74 )
                    goto LABEL_118;
                  while ( 1 )
                  {
                    v75 = sub_1648700(v74);
                    v147 = v75;
                    if ( (_DWORD)v153 )
                    {
                      LODWORD(v49) = v153 - 1;
                      v52 = ((_DWORD)v153 - 1) & (((unsigned int)v75 >> 9) ^ ((unsigned int)v75 >> 4));
                      v50 = *(_QWORD *)(v151 + 8 * v52);
                      if ( v75 == (__int64 *)v50 )
                        goto LABEL_99;
                      LODWORD(v39) = 1;
                      while ( v50 != -8 )
                      {
                        v52 = (unsigned int)v49 & ((_DWORD)v39 + (_DWORD)v52);
                        v50 = *(_QWORD *)(v151 + 8LL * (unsigned int)v52);
                        if ( v75 == (__int64 *)v50 )
                          goto LABEL_99;
                        LODWORD(v39) = (_DWORD)v39 + 1;
                      }
                    }
                    v52 = v146;
                    if ( !v146 )
                      goto LABEL_99;
                    LODWORD(v49) = v144;
                    LODWORD(v39) = ((unsigned int)v75 >> 9) ^ ((unsigned int)v75 >> 4);
                    v76 = (v146 - 1) & (unsigned int)v39;
                    v50 = v144 + 16LL * v76;
                    v77 = *(_QWORD *)v50;
                    if ( v75 != *(__int64 **)v50 )
                    {
                      v50 = 1;
                      while ( v77 != -8 )
                      {
                        v110 = v50 + 1;
                        v76 = (v146 - 1) & (v50 + v76);
                        v50 = v144 + 16LL * v76;
                        v77 = *(_QWORD *)v50;
                        if ( v75 == *(__int64 **)v50 )
                          goto LABEL_103;
                        v50 = v110;
                      }
LABEL_99:
                      v74 = *(_QWORD *)(v74 + 8);
                      if ( !v74 )
                        goto LABEL_118;
                      continue;
                    }
LABEL_103:
                    v52 = v144 + 16LL * v146;
                    if ( v50 == v52 )
                      goto LABEL_99;
                    v52 = *(unsigned int *)(a1 + 156);
                    if ( *(_DWORD *)(v50 + 8) == (_DWORD)v52 )
                      goto LABEL_99;
                    if ( !(_DWORD)v153 )
                      break;
                    LODWORD(v49) = v153 - 1;
                    v78 = 1;
                    v79 = 0;
                    LODWORD(v39) = (v153 - 1) & (unsigned int)v39;
                    v50 = v151 + 8LL * (unsigned int)v39;
                    v52 = *(_QWORD *)v50;
                    if ( v75 == *(__int64 **)v50 )
                      goto LABEL_99;
                    while ( v52 != -8 )
                    {
                      if ( v79 || v52 != -16 )
                        v50 = (__int64)v79;
                      LODWORD(v39) = v49 & (v78 + (_DWORD)v39);
                      v52 = *(_QWORD *)(v151 + 8LL * (unsigned int)v39);
                      if ( v75 == (__int64 *)v52 )
                        goto LABEL_99;
                      ++v78;
                      v79 = (__int64 **)v50;
                      v50 = v151 + 8LL * (unsigned int)v39;
                    }
                    if ( !v79 )
                      v79 = (__int64 **)v50;
                    ++v150;
                    v50 = (unsigned int)(v152 + 1);
                    if ( 4 * (int)v50 >= (unsigned int)(3 * v153) )
                      goto LABEL_255;
                    v52 = (unsigned int)(v153 - HIDWORD(v152) - v50);
                    if ( (unsigned int)v52 <= (unsigned int)v153 >> 3 )
                    {
                      sub_1353F00((__int64)&v150, v153);
                      if ( !(_DWORD)v153 )
                      {
LABEL_382:
                        LODWORD(v152) = v152 + 1;
                        BUG();
                      }
                      LODWORD(v39) = v151;
                      v123 = 1;
                      LODWORD(v49) = (_DWORD)v147;
                      v50 = (unsigned int)(v152 + 1);
                      v124 = 0;
                      v52 = ((_DWORD)v153 - 1) & (((unsigned int)v147 >> 9) ^ ((unsigned int)v147 >> 4));
                      v79 = (__int64 **)(v151 + 8 * v52);
                      v75 = *v79;
                      if ( v147 != *v79 )
                      {
                        while ( v75 != (__int64 *)-8LL )
                        {
                          if ( !v124 && v75 == (__int64 *)-16LL )
                            v124 = v79;
                          v52 = ((_DWORD)v153 - 1) & (unsigned int)(v123 + v52);
                          v79 = (__int64 **)(v151 + 8LL * (unsigned int)v52);
                          v75 = *v79;
                          if ( v147 == *v79 )
                            goto LABEL_112;
                          ++v123;
                        }
                        v75 = v147;
                        if ( v124 )
                          v79 = v124;
                      }
                    }
LABEL_112:
                    LODWORD(v152) = v50;
                    if ( *v79 != (__int64 *)-8LL )
                      --HIDWORD(v152);
                    *v79 = v75;
                    v80 = v155;
                    if ( v155 == v156 )
                    {
                      sub_1287830((__int64)&v154, v155, &v147);
                      goto LABEL_99;
                    }
                    if ( v155 )
                    {
                      *(_QWORD *)v155 = v147;
                      v80 = v155;
                    }
                    v155 = v80 + 8;
                    v74 = *(_QWORD *)(v74 + 8);
                    if ( !v74 )
                      goto LABEL_118;
                  }
                  ++v150;
LABEL_255:
                  sub_1353F00((__int64)&v150, 2 * v153);
                  if ( !(_DWORD)v153 )
                    goto LABEL_382;
                  LODWORD(v39) = v151;
                  v50 = (unsigned int)(v152 + 1);
                  v52 = ((_DWORD)v153 - 1) & (((unsigned int)v147 >> 9) ^ ((unsigned int)v147 >> 4));
                  v79 = (__int64 **)(v151 + 8 * v52);
                  v75 = *v79;
                  if ( v147 != *v79 )
                  {
                    v118 = 1;
                    v49 = 0;
                    while ( v75 != (__int64 *)-8LL )
                    {
                      if ( v75 == (__int64 *)-16LL && !v49 )
                        v49 = (__int64)v79;
                      v52 = ((_DWORD)v153 - 1) & (unsigned int)(v118 + v52);
                      v79 = (__int64 **)(v151 + 8LL * (unsigned int)v52);
                      v75 = *v79;
                      if ( v147 == *v79 )
                        goto LABEL_112;
                      ++v118;
                    }
                    v75 = v147;
                    if ( v49 )
                      v79 = (__int64 **)v49;
                  }
                  goto LABEL_112;
                }
                if ( v61 )
                {
                  ++v143;
                  goto LABEL_161;
                }
LABEL_118:
                v54 = v155;
LABEL_119:
                if ( v154 == v54 )
                {
LABEL_120:
                  if ( v54 )
                    j_j___libc_free_0(v54, v156 - v54);
LABEL_122:
                  j___libc_free_0(v151);
                  v11 = sub_1A834A0(
                          a1,
                          v138,
                          (__int64)v140,
                          0xAAAAAAAAAAAAAAABLL * ((v141 - v140) >> 3),
                          (__int64)&v143,
                          a3,
                          a4,
                          a5,
                          a6,
                          v81,
                          v82,
                          a9,
                          a10);
                  j___libc_free_0(v144);
                  v83 = v141;
                  v84 = v140;
                  if ( v141 != v140 )
                  {
                    do
                    {
                      v85 = *((_QWORD *)v84 + 2);
                      if ( v85 != 0 && v85 != -8 && v85 != -16 )
                        sub_1649B30(v84);
                      v84 += 24;
                    }
                    while ( v83 != v84 );
                    v84 = v140;
                  }
                  if ( v84 )
                    j_j___libc_free_0(v84, v142 - v84);
                  return v11;
                }
              }
              v99 = 1;
              while ( v49 != -8 )
              {
                LODWORD(v39) = v99 + 1;
                v91 = v52 & (v99 + v91);
                v92 = v144 + 16LL * v91;
                v49 = *(_QWORD *)v92;
                if ( v89 == *(__int64 **)v92 )
                  goto LABEL_147;
                v99 = (int)v39;
              }
LABEL_168:
              v50 = *v89;
              v92 = v144 + 16LL * v146;
              if ( *(_BYTE *)(*v89 + 8) == 16 )
LABEL_245:
                v50 = **(_QWORD **)(v50 + 16);
              v52 = *(_DWORD *)(v50 + 8) >> 8;
              v93 = *(_DWORD *)(v50 + 8) >> 8;
              if ( !v146 )
                goto LABEL_170;
              v52 = v146 - 1;
              goto LABEL_341;
            }
          }
        }
        ++v143;
        goto LABEL_65;
      }
LABEL_49:
      v37 = v151;
      v36 = v153;
    }
    ++v150;
LABEL_52:
    sub_1353F00((__int64)&v150, 2 * v36);
    if ( !(_DWORD)v153 )
      goto LABEL_386;
    v44 = (v153 - 1) & (((unsigned int)v43 >> 9) ^ ((unsigned int)v43 >> 4));
    v39 = (__int64 *)(v151 + 8LL * v44);
    v45 = *v39;
    v46 = v152 + 1;
    if ( v43 != *v39 )
    {
      v130 = 1;
      v131 = 0;
      while ( v45 != -8 )
      {
        if ( !v131 && v45 == -16 )
          v131 = v39;
        v44 = (v153 - 1) & (v130 + v44);
        v39 = (__int64 *)(v151 + 8LL * v44);
        v45 = *v39;
        if ( v43 == *v39 )
          goto LABEL_54;
        ++v130;
      }
      if ( v131 )
        v39 = v131;
    }
    goto LABEL_54;
  }
  return v11;
}
