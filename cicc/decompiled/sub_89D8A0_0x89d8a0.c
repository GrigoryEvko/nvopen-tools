// Function: sub_89D8A0
// Address: 0x89d8a0
//
__int64 __fastcall sub_89D8A0(__int64 a1, __int64 a2, __int64 *a3)
{
  __int64 v4; // r15
  const __m128i *v5; // rdi
  __int64 v6; // r14
  char v7; // al
  __int64 v8; // rax
  __int64 v9; // r12
  __int64 *v10; // r11
  FILE *v11; // r10
  __m128i *v12; // rdi
  __int64 v13; // rsi
  unsigned __int64 v14; // rdi
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  __int64 v19; // rsi
  __int64 v20; // rdx
  __int64 v21; // r8
  __int64 v22; // r9
  char v23; // al
  __int64 v24; // rax
  char v25; // dl
  int v26; // eax
  int v27; // ecx
  _QWORD *v29; // rax
  unsigned int v30; // r11d
  int v31; // edx
  int v32; // edi
  __int64 v33; // rax
  __int16 v34; // si
  __int16 v35; // dx
  char v36; // dl
  char v37; // dl
  char v38; // al
  char v39; // al
  FILE *v40; // rsi
  char v41; // dl
  char v42; // dl
  char v43; // al
  int v44; // eax
  _QWORD *v45; // rax
  __m128i *v46; // rax
  __int64 v47; // rdx
  __int64 v48; // rcx
  __int64 v49; // r8
  __int64 v50; // r9
  int v51; // eax
  FILE *v52; // r10
  __int64 *v53; // r11
  __int64 v54; // rdi
  int v55; // eax
  __int64 v56; // rdx
  unsigned int v57; // edi
  FILE *v58; // rsi
  int v59; // eax
  __int64 v60; // r13
  char v61; // al
  char v62; // dl
  char v63; // al
  __int64 v64; // rax
  __int64 v65; // rdx
  int v66; // eax
  FILE **v67; // rax
  FILE *v68; // r10
  int v69; // eax
  __int64 v70; // rax
  int v71; // eax
  __int64 v72; // r12
  __int64 v73; // rax
  int v74; // ecx
  __int64 v75; // rdx
  __int64 v76; // rax
  _QWORD *v77; // rax
  int v78; // eax
  int v79; // eax
  _BOOL4 v80; // eax
  __int64 v81; // [rsp-10h] [rbp-240h]
  __int64 v82; // [rsp-8h] [rbp-238h]
  __int64 *v83; // [rsp+8h] [rbp-228h]
  __int64 v84; // [rsp+8h] [rbp-228h]
  __int64 v85; // [rsp+10h] [rbp-220h]
  __int64 *v86; // [rsp+18h] [rbp-218h]
  int v87; // [rsp+20h] [rbp-210h]
  unsigned int v88; // [rsp+24h] [rbp-20Ch]
  int v89; // [rsp+28h] [rbp-208h]
  unsigned int v90; // [rsp+28h] [rbp-208h]
  char v92; // [rsp+38h] [rbp-1F8h]
  unsigned int v93; // [rsp+38h] [rbp-1F8h]
  FILE *v94; // [rsp+38h] [rbp-1F8h]
  FILE *v95; // [rsp+38h] [rbp-1F8h]
  FILE *v96; // [rsp+38h] [rbp-1F8h]
  __m128i *v97; // [rsp+40h] [rbp-1F0h]
  _QWORD *v98; // [rsp+40h] [rbp-1F0h]
  unsigned int v99; // [rsp+40h] [rbp-1F0h]
  unsigned int v100; // [rsp+40h] [rbp-1F0h]
  __int64 *v101; // [rsp+40h] [rbp-1F0h]
  char v102; // [rsp+40h] [rbp-1F0h]
  int v103; // [rsp+48h] [rbp-1E8h]
  FILE *v104; // [rsp+48h] [rbp-1E8h]
  FILE *v105; // [rsp+48h] [rbp-1E8h]
  FILE *v106; // [rsp+48h] [rbp-1E8h]
  FILE *v107; // [rsp+48h] [rbp-1E8h]
  FILE *v108; // [rsp+48h] [rbp-1E8h]
  FILE *v109; // [rsp+48h] [rbp-1E8h]
  FILE *v110; // [rsp+48h] [rbp-1E8h]
  char v111; // [rsp+50h] [rbp-1E0h]
  __int64 v112; // [rsp+50h] [rbp-1E0h]
  __int64 v113; // [rsp+60h] [rbp-1D0h]
  int v114; // [rsp+60h] [rbp-1D0h]
  __m128i v116[2]; // [rsp+70h] [rbp-1C0h] BYREF
  _BYTE v117[352]; // [rsp+90h] [rbp-1A0h] BYREF
  int v118; // [rsp+1F0h] [rbp-40h] BYREF
  __int16 v119; // [rsp+1F4h] [rbp-3Ch]

  v4 = *(_QWORD *)a1;
  if ( (*(_BYTE *)(*(_QWORD *)a1 + 10LL) & 8) == 0 )
  {
LABEL_4:
    v6 = *(_QWORD *)(a2 + 24);
    if ( !v6 )
      goto LABEL_10;
    goto LABEL_5;
  }
  v5 = *(const __m128i **)(v4 + 288);
  if ( (v5[8].m128i_i8[12] & 0xFB) == 8 )
  {
    if ( (sub_8D4C10(v5, dword_4F077C4 != 2) & 1) != 0 )
      goto LABEL_4;
    v5 = *(const __m128i **)(v4 + 288);
  }
  *(_QWORD *)(v4 + 288) = sub_73C570(v5, 1);
  v6 = *(_QWORD *)(a2 + 24);
  if ( !v6 )
    goto LABEL_10;
LABEL_5:
  v88 = 0;
  v7 = *(_BYTE *)(v6 + 80);
  if ( v7 != 7 )
    goto LABEL_6;
  if ( (*(_BYTE *)(a2 + 18) & 1) == 0 )
  {
LABEL_16:
    v111 = 0;
    v9 = 0;
    goto LABEL_17;
  }
  v29 = sub_89B100(v6, a1, a2);
  v6 = (__int64)v29;
  if ( !v29 )
  {
LABEL_10:
    if ( (*(_BYTE *)(a2 + 17) & 0x20) == 0 )
    {
      if ( !(*(_BYTE *)(a2 + 16) & 0x18 | word_4D04A10 & 0x40) )
      {
        v72 = *(_QWORD *)a2;
        v73 = *(_QWORD *)&dword_4F063F8;
        v74 = dword_4F04C5C;
        v75 = *(_QWORD *)(*(_QWORD *)a2 + 64LL);
        *(_QWORD *)(*(_QWORD *)a2 + 64LL) = 0;
        *(_QWORD *)(a1 + 360) = v73;
        v114 = v74;
        v112 = v75;
        dword_4F04C5C = *(_DWORD *)(a1 + 200);
        v6 = sub_7CFB70((_QWORD *)a2, 0);
        dword_4F04C5C = v114;
        v76 = qword_4F063F0;
        *(_QWORD *)(a1 + 368) = qword_4F063F0;
        *(_QWORD *)(a1 + 384) = v76;
        *(_QWORD *)(v72 + 64) = v112;
        if ( v6 && *(_BYTE *)(v6 + 80) == 21 )
        {
          v88 = *(_DWORD *)(a1 + 116);
          if ( !v88 )
            goto LABEL_105;
          *(_BYTE *)(v4 + 127) |= 0x10u;
        }
        else
        {
          v77 = sub_898DA0(a1, a2, 0);
          *(_DWORD *)(a1 + 116) = 1;
          v6 = (__int64)v77;
          *(_BYTE *)(v4 + 127) |= 0x10u;
          if ( !v77 )
            goto LABEL_185;
        }
        v88 = 1;
        v7 = *(_BYTE *)(v6 + 80);
LABEL_6:
        if ( v7 != 21 )
          goto LABEL_7;
        goto LABEL_105;
      }
      sub_6851C0(0x1F6u, (_DWORD *)(a2 + 8));
    }
    *(__m128i *)a2 = _mm_loadu_si128(xmmword_4F06660);
    *(__m128i *)(a2 + 16) = _mm_loadu_si128(&xmmword_4F06660[1]);
    *(__m128i *)(a2 + 32) = _mm_loadu_si128(&xmmword_4F06660[2]);
    v8 = *(_QWORD *)dword_4F07508;
    *(__m128i *)(a2 + 48) = _mm_loadu_si128(&xmmword_4F06660[3]);
    *(_QWORD *)(a2 + 8) = v8;
    *(_BYTE *)(a2 + 17) |= 0x20u;
    LODWORD(v8) = *(_DWORD *)(a1 + 116);
    *(_DWORD *)(a1 + 52) = 1;
    LOBYTE(v88) = v8;
    if ( !(_DWORD)v8 )
    {
LABEL_14:
      v9 = 0;
      v92 = 0;
      v111 = 0;
      v113 = 0;
      v85 = *(_QWORD *)(v4 + 16) & 1LL;
      v87 = *(_DWORD *)(v4 + 16) & 1;
      v89 = 0;
LABEL_20:
      *(_BYTE *)(a2 + 17) |= 0x20u;
      *(_QWORD *)(a2 + 24) = 0;
      v6 = (__int64)sub_898DA0(a1, a2, 0);
      v12 = *(__m128i **)(v6 + 88);
      *(_DWORD *)(v6 + 40) = *(_DWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C64);
      sub_879080(v12, 0, *(_QWORD *)(a1 + 192));
      v13 = 0;
      v14 = *(_QWORD *)(v6 + 88) + 200LL;
      sub_879080((__m128i *)v14, 0, *(_QWORD *)(a1 + 192));
      v103 = 1;
      goto LABEL_21;
    }
    *(_BYTE *)(v4 + 127) |= 0x10u;
LABEL_185:
    LOBYTE(v88) = 1;
    goto LABEL_14;
  }
  v7 = *((_BYTE *)v29 + 80);
  if ( v7 != 21 )
  {
LABEL_7:
    switch ( v7 )
    {
      case 4:
      case 5:
        v9 = *(_QWORD *)(*(_QWORD *)(v6 + 96) + 80LL);
        break;
      case 6:
        v9 = *(_QWORD *)(*(_QWORD *)(v6 + 96) + 32LL);
        break;
      case 9:
      case 10:
        v9 = *(_QWORD *)(*(_QWORD *)(v6 + 96) + 56LL);
        break;
      case 19:
      case 20:
      case 22:
        v111 = 0;
        goto LABEL_106;
      default:
        goto LABEL_16;
    }
    v111 = 0;
    if ( v9 )
      goto LABEL_59;
LABEL_17:
    v113 = 0;
    v10 = 0;
    v11 = 0;
    goto LABEL_18;
  }
LABEL_105:
  v111 = 1;
LABEL_106:
  v9 = *(_QWORD *)(v6 + 88);
  if ( !v9 )
  {
    v113 = 0;
    v10 = 0;
    v11 = 0;
    goto LABEL_60;
  }
LABEL_59:
  v10 = *(__int64 **)(v9 + 232);
  v113 = *(_QWORD *)(v9 + 192);
  v11 = *(FILE **)v113;
LABEL_60:
  if ( v111 && *(_BYTE *)(v4 + 269) != 1 )
    *(_DWORD *)(a1 + 36) = 1;
LABEL_18:
  v85 = *(_QWORD *)(v4 + 16) & 1LL;
  v87 = *(_DWORD *)(v4 + 16) & 1;
  if ( (*(_BYTE *)(a2 + 17) & 0x20) != 0 )
  {
    v89 = 0;
    v92 = v111 & (v9 != 0);
    goto LABEL_20;
  }
  v39 = *(_BYTE *)(v6 + 80);
  if ( v39 != 21 && v39 != 9 )
  {
    v92 = v111 & (v9 != 0);
    v40 = (FILE *)(a2 + 8);
    if ( v39 == 8 )
    {
      sub_6851C0(0xF6u, v40);
      v89 = 0;
    }
    else
    {
      if ( v39 == 16 )
        sub_6851C0(0x12Au, v40);
      else
        sub_6854C0(0x93u, v40, v6);
      v89 = 0;
    }
    goto LABEL_20;
  }
  v101 = v10;
  v104 = v11;
  v13 = qword_4F04C68[0] + 776LL * dword_4F04C34;
  v51 = sub_85ED80(v6, v13);
  v52 = v104;
  v53 = v101;
  v89 = v51;
  if ( !v51 )
  {
    sub_6854E0(0x227u, v6);
    v92 = v111 & (v9 != 0);
    goto LABEL_20;
  }
  v16 = ((unsigned __int8)v88 ^ 1) & 1;
  v102 = (v113 != 0) & (v88 ^ 1);
  if ( v102 )
  {
    v54 = *(_QWORD *)(v4 + 288);
    v13 = *(_QWORD *)(v113 + 120);
    if ( v54 != v13 )
    {
      v83 = v53;
      v55 = sub_8DED30(v54, v13, 5);
      v16 = ((unsigned __int8)v88 ^ 1) & 1;
      v52 = v104;
      v89 = v55;
      v53 = v83;
      if ( !v55 )
      {
        v56 = v6;
        v57 = 147;
        v58 = (FILE *)(a2 + 8);
LABEL_155:
        sub_6854C0(v57, v58, v56);
        v92 = v111 & (v9 != 0);
        goto LABEL_20;
      }
    }
  }
  if ( v53 && (_BYTE)v16 )
  {
    v64 = v53[4];
    v65 = *(_QWORD *)(*(_QWORD *)(a1 + 192) + 32LL);
    if ( v64 )
    {
      if ( v65 )
      {
        v86 = v53;
        v95 = v52;
        v13 = *(_QWORD *)(v65 + 16);
        v84 = v13;
        v66 = sub_739400(*(__int128 **)(v64 + 16), (__int128 *)v13);
        v52 = v95;
        v53 = v86;
        v89 = v66;
        if ( !v66 )
        {
          v57 = 3051;
          v58 = (FILE *)(v13 + 8);
          v56 = v6;
          if ( !v84 )
            v58 = (FILE *)(a2 + 8);
          goto LABEL_155;
        }
      }
    }
    if ( *(_BYTE *)(v6 + 80) != 9 )
    {
      v96 = v52;
      v89 = sub_89B3C0(*v53, **(_QWORD **)(a1 + 192), 1, 0, (_DWORD *)(a2 + 8), 8u);
      if ( !v89
        || (v13 = a1,
            v80 = sub_89BD20(**(_QWORD **)(a1 + 192), a1, v6, (_DWORD *)(v6 + 48), 1, 0, 1, 8u),
            v17 = v81,
            v52 = v96,
            v89 = v80,
            v18 = v82,
            !v80) )
      {
        v92 = v111 & (v9 != 0);
        goto LABEL_20;
      }
    }
  }
  v14 = v88;
  if ( !v88 && (*(_BYTE *)(v6 + 81) & 0x10) != 0 )
  {
    v13 = *(unsigned int *)(a1 + 44);
    if ( !(_DWORD)v13 )
    {
      v16 = *(unsigned int *)(a1 + 48);
      if ( !(_DWORD)v16 && !*(_QWORD *)(a1 + 240) )
      {
        v13 = v6;
        v14 = a1;
        v94 = v52;
        v59 = sub_89BFC0(a1, v6, 1, (FILE *)dword_4F07508);
        v52 = v94;
        v89 = v59;
        if ( !v59 )
        {
          v92 = v111 & (v9 != 0);
          goto LABEL_20;
        }
      }
    }
  }
  if ( !v102 )
  {
    if ( (*(_BYTE *)(v6 + 81) & 2) == 0 || !v113 )
      goto LABEL_137;
    LOBYTE(v16) = *(_BYTE *)(v113 + 176);
    goto LABEL_178;
  }
  v14 = v113;
  v16 = *(unsigned __int8 *)(v113 + 176);
  if ( ((*(_BYTE *)(v113 + 176) & 8) != 0) != ((*(_QWORD *)(v4 + 8) & 0x400000LL) != 0) )
  {
    sub_6854F0(8u, ((v16 & 8) != 0) + 2502, (_DWORD *)(a2 + 8), (_QWORD *)(v113 + 64));
LABEL_158:
    v89 = 0;
    v92 = v111 & (v9 != 0);
    goto LABEL_20;
  }
  if ( (*(_BYTE *)(v6 + 81) & 2) != 0 )
  {
LABEL_178:
    v16 &= 1u;
    if ( !(_DWORD)v16 || (*(_BYTE *)(v113 + 172) & 0x28) == 0x20 )
    {
      sub_685920((_DWORD *)(a2 + 8), (FILE *)v6, 8u);
      v89 = 1;
      v92 = v111 & (v9 != 0);
      goto LABEL_20;
    }
  }
LABEL_137:
  if ( (*(_BYTE *)(v4 + 131) & 0x10) != 0 )
  {
    sub_6851C0(0xB8Cu, (_DWORD *)(a2 + 8));
    goto LABEL_158;
  }
  v15 = v88;
  if ( !v88 )
  {
    v60 = *(_QWORD *)(v4 + 288);
    if ( !v111 || *(_BYTE *)(v4 + 269) != 1 )
    {
      *(_BYTE *)(v4 + 122) |= 1u;
      *(_DWORD *)(a1 + 36) = 1;
    }
    v13 = a2;
    sub_657FD0(v4, a2, 1);
    switch ( *(_BYTE *)(v6 + 80) )
    {
      case 4:
      case 5:
        v9 = *(_QWORD *)(*(_QWORD *)(v6 + 96) + 80LL);
        break;
      case 6:
        v9 = *(_QWORD *)(*(_QWORD *)(v6 + 96) + 32LL);
        break;
      case 9:
      case 0xA:
        v9 = *(_QWORD *)(*(_QWORD *)(v6 + 96) + 56LL);
        break;
      case 0x13:
      case 0x14:
      case 0x15:
      case 0x16:
        v9 = *(_QWORD *)(v6 + 88);
        break;
      default:
        BUG();
    }
    v67 = *(FILE ***)(v9 + 192);
    v68 = *v67;
    if ( LOBYTE((*v67)->_IO_backup_base) == 9
      && (dword_4F077C4 != 2 || unk_4F07778 <= 201702 || !v113 || (*(_BYTE *)(v113 + 172) & 0x20) == 0)
      && (*(_BYTE *)(v9 + 168) & 1) != 0 )
    {
      v13 = (__int64)*v67;
      v108 = *v67;
      sub_685920((_DWORD *)(a2 + 8), v68, 8u);
      v68 = v108;
    }
    v105 = v68;
    v69 = sub_8D32B0(v60);
    v52 = v105;
    if ( v69 )
    {
      v70 = sub_8D46C0(v60);
      v71 = sub_8D2310(v70);
      v52 = v105;
      if ( v71 )
        goto LABEL_163;
    }
    v14 = v60;
    v107 = v52;
    v78 = sub_8D3D10(v60);
    v52 = v107;
    if ( v78 )
    {
      v14 = sub_8D4870(v60);
      v79 = sub_8D2310(v14);
      v52 = v107;
      if ( v79 )
      {
LABEL_163:
        v13 = v6;
        v14 = v60;
        v106 = v52;
        sub_6464A0(v60, v6, (unsigned int *)(a2 + 8), 1u);
        v52 = v106;
      }
    }
    if ( v111 )
    {
      if ( *(_DWORD *)(a1 + 24) )
      {
        if ( !*(_DWORD *)(a1 + 16) )
        {
          v15 = (*(_BYTE *)(v9 + 160) & 1) == 0;
          LOBYTE(v88) = (*(_BYTE *)(v9 + 160) & 1) == 0;
          if ( (*(_BYTE *)(v9 + 160) & 1) == 0 )
          {
            v13 = v9;
            v14 = v6;
            v110 = v52;
            sub_899910(v6, v9, (FILE *)(a2 + 8));
            v52 = v110;
          }
          *(_BYTE *)(v113 + 176) &= ~1u;
        }
      }
      else
      {
        LOBYTE(v88) = 0;
      }
    }
    else if ( *(_BYTE *)(v4 + 269) )
    {
      v14 = 80;
      v109 = v52;
      v13 = a2 + 8;
      sub_6851C0(0x50u, (_DWORD *)(a2 + 8));
      v52 = v109;
    }
  }
  *(_QWORD *)v4 = v52;
  v89 = 0;
  v92 = v111 & (v9 != 0);
  v103 = 0;
LABEL_21:
  if ( *(_DWORD *)(a1 + 72) )
  {
    v97 = *(__m128i **)(*(_QWORD *)(a1 + 464) + 88LL);
    while ( word_4F06418[0] != 9 )
      sub_7B8B50(v14, (unsigned int *)v13, v15, v16, v17, v18);
    sub_7B8B50(v14, (unsigned int *)v13, v15, v16, v17, v18);
    sub_7BC160((__int64)v97);
  }
  else
  {
    v97 = 0;
  }
  v19 = 1;
  sub_7ADF70((__int64)v116, 1);
  if ( word_4F06418[0] == 9 || word_4F06418[0] != 56 && !v85 && (word_4F06418[0] != 73 || !dword_4D04428) )
  {
    if ( !*(_DWORD *)(a1 + 72) )
      sub_8975E0((const __m128i *)a1, dword_4F0664C, 0);
    goto LABEL_31;
  }
  *(_DWORD *)(a1 + 36) = 1;
  *(_QWORD *)(a1 + 408) = *(_QWORD *)&dword_4F063F8;
  *(_BYTE *)(v4 + 127) |= 4u;
  v30 = dword_4F06650[0];
  if ( v113 && (*(_BYTE *)(v113 + 176) & 1) != 0 && !(*(_DWORD *)(a1 + 24) | v89) )
  {
    v90 = dword_4F06650[0];
    v19 = a2 + 8;
    sub_6854C0(0x94u, (FILE *)(a2 + 8), v6);
    v30 = v90;
  }
  if ( !*(_DWORD *)(a1 + 72) )
  {
    if ( word_4F06418[0] != 75 )
    {
      v100 = v30;
      memset(v117, 0, sizeof(v117));
      v117[75] = 1;
      v118 = 0;
      v119 = 0;
      sub_7B8B50((unsigned __int64)&v118, (unsigned int *)v19, v20, 0, v21, v22);
      sub_7C6880(0, (__int64)v117, v47, v48, v49, v50);
      v30 = v100;
    }
    v99 = v30;
    sub_8975E0((const __m128i *)a1, dword_4F0664C, 0);
    v31 = v99;
    v97 = v116;
    sub_7AE960(a1 + 288, (__int64)v116, v31, v87, v87, 0);
  }
  v32 = qword_4F063F0;
  v33 = v97->m128i_i64[1];
  v34 = WORD2(qword_4F063F0);
  while ( v33 )
  {
    v35 = *(_WORD *)(*(_QWORD *)v33 + 24LL);
    if ( v35 == 75 || v35 == 9 )
    {
      v32 = *(_DWORD *)(v33 + 16);
      v34 = *(_WORD *)(v33 + 20);
      break;
    }
    v33 = *(_QWORD *)v33;
  }
  *(_DWORD *)(a1 + 416) = v32;
  *(_WORD *)(a1 + 420) = v34;
  if ( !v103 )
  {
LABEL_31:
    if ( v92 )
      goto LABEL_80;
    goto LABEL_32;
  }
  sub_7AEA70(v116);
  v97 = 0;
  if ( v92 )
  {
LABEL_80:
    sub_88F9D0(**(__int64 ***)(a1 + 192), *(_DWORD *)(a1 + 28));
    v93 = dword_4F04C3C;
    if ( dword_4F04C3C )
    {
LABEL_81:
      v36 = *(_BYTE *)(a1 + 84);
      dword_4F04C3C = 1;
      v37 = (8 * (v36 & 1)) | *(_BYTE *)(v9 + 160) & 0xF7;
      *(_BYTE *)(v9 + 160) = v37;
      v38 = v37 & 0xEF | (16 * (*(_BYTE *)(a1 + 88) & 1));
      *(_BYTE *)(v9 + 160) = v38;
      *(_BYTE *)(v9 + 160) = (32 * (*(_BYTE *)(a1 + 128) & 1)) | v38 & 0xDF;
      if ( v111 && (v88 & 1) == 0 && (!v113 || (*(_BYTE *)(v113 + 176) & 1) != 0) )
        goto LABEL_40;
      goto LABEL_93;
    }
LABEL_34:
    v23 = *(_BYTE *)(v6 + 81) & 0x10;
    if ( *(_DWORD *)(a1 + 36) )
    {
      if ( !v23 )
      {
        if ( v113 )
        {
LABEL_37:
          v24 = *(_QWORD *)(v4 + 280);
          dword_4F04C3C = 1;
          *(_QWORD *)(v113 + 256) = v24;
          v25 = (8 * (*(_BYTE *)(a1 + 84) & 1)) | *(_BYTE *)(v9 + 160) & 0xF7;
          *(_BYTE *)(v9 + 160) = v25;
          LOBYTE(v24) = v25 & 0xEF | (16 * (*(_BYTE *)(a1 + 88) & 1));
          *(_BYTE *)(v9 + 160) = v24;
          *(_BYTE *)(v9 + 160) = (32 * (*(_BYTE *)(a1 + 128) & 1)) | v24 & 0xDF;
          if ( v111 && (v88 & 1) == 0 )
          {
            v93 = 0;
            if ( (*(_BYTE *)(v113 + 176) & 1) != 0 )
              goto LABEL_40;
            goto LABEL_93;
          }
          goto LABEL_146;
        }
LABEL_145:
        v61 = *(_BYTE *)(a1 + 84);
        dword_4F04C3C = 1;
        v62 = (8 * (v61 & 1)) | *(_BYTE *)(v9 + 160) & 0xF7;
        *(_BYTE *)(v9 + 160) = v62;
        v63 = (16 * (*(_BYTE *)(a1 + 88) & 1)) | v62 & 0xEF;
        *(_BYTE *)(v9 + 160) = v63;
        *(_BYTE *)(v9 + 160) = (32 * (*(_BYTE *)(a1 + 128) & 1)) | v63 & 0xDF;
        if ( v111 && (v88 & 1) == 0 )
        {
          v93 = 0;
          goto LABEL_40;
        }
LABEL_146:
        v93 = 0;
        goto LABEL_93;
      }
    }
    else if ( !v23 || (*(_BYTE *)(v4 + 122) & 1) == 0 )
    {
      goto LABEL_112;
    }
    if ( !v113 )
      goto LABEL_145;
    if ( (*(_BYTE *)(v113 + 176) & 1) == 0 )
      goto LABEL_37;
LABEL_112:
    v46 = sub_8921F0(*(_QWORD *)(a1 + 336));
    v93 = 0;
    v46[2].m128i_i64[0] = *(_QWORD *)(v4 + 280);
    v46[3].m128i_i8[8] = *(_BYTE *)(v4 + 268);
    goto LABEL_81;
  }
LABEL_32:
  if ( !v9 )
    goto LABEL_50;
  v93 = dword_4F04C3C;
  if ( !dword_4F04C3C )
    goto LABEL_34;
  v41 = *(_BYTE *)(a1 + 84);
  dword_4F04C3C = 1;
  v42 = (8 * (v41 & 1)) | *(_BYTE *)(v9 + 160) & 0xF7;
  *(_BYTE *)(v9 + 160) = v42;
  v43 = v42 & 0xEF | (16 * (*(_BYTE *)(a1 + 88) & 1));
  *(_BYTE *)(v9 + 160) = v43;
  *(_BYTE *)(v9 + 160) = (32 * (*(_BYTE *)(a1 + 128) & 1)) | v43 & 0xDF;
LABEL_93:
  if ( !*(_QWORD *)(v9 + 8) || !v103 )
    sub_879080((__m128i *)v9, v97, *(_QWORD *)(a1 + 192));
LABEL_40:
  sub_7AE340(v9);
  v98 = (_QWORD *)(a2 + 8);
  if ( *(_DWORD *)(a1 + 36) )
  {
    sub_8756F0(3, v6, (_QWORD *)(a2 + 8), 0);
    if ( *(_QWORD *)(v9 + 208) )
      goto LABEL_42;
  }
  else
  {
    sub_8756F0(1, v6, v98, 0);
    if ( *(_QWORD *)(v9 + 208) )
    {
LABEL_42:
      if ( !v103 )
      {
        if ( (*(_BYTE *)(v9 + 168) & 1) == 0 )
        {
          sub_879080((__m128i *)(v9 + 200), (const __m128i *)(a1 + 288), *(_QWORD *)(a1 + 192));
          v26 = *(_DWORD *)(v4 + 68);
          *(_BYTE *)(v9 + 168) |= 1u;
          *(_DWORD *)(v9 + 240) = v26;
        }
        goto LABEL_45;
      }
      goto LABEL_46;
    }
  }
  sub_879080((__m128i *)(v9 + 200), (const __m128i *)(a1 + 288), *(_QWORD *)(a1 + 192));
  v44 = *(_DWORD *)(v4 + 68);
  *(_BYTE *)(v9 + 168) |= 1u;
  *(_DWORD *)(v9 + 240) = v44;
  if ( !v103 )
  {
LABEL_45:
    sub_644920((_QWORD *)v4, 1);
    if ( v111
      && unk_4F072F3
      && v113
      && (*(_BYTE *)(v113 + 156) & 1) != 0
      && (unsigned int)sub_826000(*(const char **)(v113 + 8)) )
    {
      sub_6849F0(7u, 0xE4Au, v98, *(_QWORD *)(v113 + 8));
    }
  }
LABEL_46:
  sub_729470(v113, (const __m128i *)(a1 + 344));
  v27 = *(_DWORD *)(a1 + 60);
  *(_DWORD *)(a1 + 320) = 1;
  if ( v27 )
    *(_BYTE *)(*(_QWORD *)(v9 + 104) + 121LL) |= 1u;
  if ( (*(_BYTE *)(*(_QWORD *)(v9 + 104) + 121LL) & 1) != 0 )
  {
    v45 = sub_878440();
    v45[1] = v6;
    if ( !unk_4D03B70 )
      unk_4D03B70 = v45;
    if ( qword_4F601C0 )
      *(_QWORD *)qword_4F601C0 = v45;
    qword_4F601C0 = (__int64)v45;
  }
  dword_4F04C3C = v93;
  sub_65C210(v4);
LABEL_50:
  if ( (*(_QWORD *)(v4 + 8) & 8) != 0 )
  {
    v9 = 0;
    v6 = 0;
    sub_6851C0(0x115u, (_DWORD *)(a2 + 8));
  }
  else if ( v103 )
  {
    v9 = *(_QWORD *)(v4 + 8) & 8LL;
    v6 = v9;
  }
  *a3 = v9;
  return v6;
}
