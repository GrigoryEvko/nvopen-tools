// Function: sub_8AB5A0
// Address: 0x8ab5a0
//
__int64 __fastcall sub_8AB5A0(__int64 a1)
{
  __int64 v1; // r12
  __int64 v2; // r15
  __int64 v3; // rdi
  __int64 v4; // rax
  __m128i v5; // xmm4
  __m128i v6; // xmm5
  __m128i v7; // xmm6
  __m128i v8; // xmm7
  __int64 v10; // r15
  __int64 v11; // r8
  __int64 v12; // rdi
  __int64 v13; // r9
  __int64 v14; // rdi
  __int64 v15; // r15
  unsigned __int64 v16; // rsi
  __int64 v17; // rdi
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // rdx
  __int64 v22; // rcx
  int v23; // eax
  __int64 v24; // r15
  __m128i *v25; // rax
  char v26; // dl
  __int64 v27; // r10
  __int64 v28; // r15
  __int64 v29; // rax
  __int64 v30; // r15
  __int64 v31; // r9
  __int64 v32; // r11
  __int64 v33; // rax
  size_t v34; // rax
  size_t v35; // rax
  int v36; // eax
  char v37; // al
  __int64 v38; // rdx
  __int64 v39; // rcx
  const __m128i *v40; // rdi
  __int64 v41; // rsi
  __int64 v42; // r8
  unsigned int v43; // r9d
  __int64 v44; // r11
  __int64 v45; // r10
  char v46; // al
  __int64 *v47; // rcx
  __int64 *v48; // rax
  __int64 v49; // rax
  __int64 v50; // rax
  unsigned int v51; // r9d
  __int64 v52; // r8
  __int64 v53; // r10
  __int64 v54; // r10
  __int64 v55; // r10
  __int64 v56; // r8
  _QWORD *i; // rax
  char v58; // dl
  __int64 v59; // rdx
  __int64 v60; // rcx
  __int64 v61; // r8
  __int64 v62; // r9
  __int64 v63; // r10
  __int64 v64; // rdx
  __int64 v65; // rcx
  __int64 v66; // r8
  __int64 *v67; // r9
  __int64 v68; // rdx
  __int64 v69; // rcx
  __int64 v70; // r8
  __int64 v71; // r9
  __int64 v72; // r10
  int v73; // eax
  int v74; // eax
  int v75; // eax
  __int64 v76; // rax
  char v77; // al
  char v78; // al
  _QWORD *v79; // rax
  __int64 v80; // rax
  __int64 v81; // rax
  __m128i *v82; // rax
  __int64 v83; // [rsp+0h] [rbp-F0h]
  __int64 v84; // [rsp+8h] [rbp-E8h]
  __int64 v85; // [rsp+8h] [rbp-E8h]
  __int64 v86; // [rsp+8h] [rbp-E8h]
  __int64 v87; // [rsp+8h] [rbp-E8h]
  __int64 v88; // [rsp+8h] [rbp-E8h]
  __int64 v89; // [rsp+10h] [rbp-E0h]
  __int64 v90; // [rsp+10h] [rbp-E0h]
  __int64 v91; // [rsp+10h] [rbp-E0h]
  __int64 v92; // [rsp+10h] [rbp-E0h]
  __int64 v93; // [rsp+10h] [rbp-E0h]
  __int64 v94; // [rsp+18h] [rbp-D8h]
  unsigned int v95; // [rsp+18h] [rbp-D8h]
  unsigned int v96; // [rsp+18h] [rbp-D8h]
  __int64 v97; // [rsp+18h] [rbp-D8h]
  int v98; // [rsp+18h] [rbp-D8h]
  __int64 v99; // [rsp+18h] [rbp-D8h]
  __int64 v100; // [rsp+18h] [rbp-D8h]
  unsigned int v101; // [rsp+20h] [rbp-D0h]
  __int64 v102; // [rsp+20h] [rbp-D0h]
  __int64 v103; // [rsp+20h] [rbp-D0h]
  int v104; // [rsp+20h] [rbp-D0h]
  __int64 v105; // [rsp+20h] [rbp-D0h]
  unsigned int v106; // [rsp+20h] [rbp-D0h]
  int v107; // [rsp+20h] [rbp-D0h]
  __int64 v108; // [rsp+28h] [rbp-C8h]
  __int64 v109; // [rsp+28h] [rbp-C8h]
  __int64 v110; // [rsp+28h] [rbp-C8h]
  __int64 v111; // [rsp+28h] [rbp-C8h]
  __int64 v112; // [rsp+28h] [rbp-C8h]
  __int64 v113; // [rsp+28h] [rbp-C8h]
  __int64 v114; // [rsp+28h] [rbp-C8h]
  __int64 v115; // [rsp+28h] [rbp-C8h]
  __int64 v116; // [rsp+28h] [rbp-C8h]
  __int64 v117; // [rsp+28h] [rbp-C8h]
  __int64 v118; // [rsp+28h] [rbp-C8h]
  __int64 v119; // [rsp+30h] [rbp-C0h]
  __int64 v120; // [rsp+30h] [rbp-C0h]
  __int64 v121; // [rsp+30h] [rbp-C0h]
  __int64 v122; // [rsp+30h] [rbp-C0h]
  __int64 v123; // [rsp+30h] [rbp-C0h]
  unsigned __int64 v124; // [rsp+38h] [rbp-B8h]
  const char *v125; // [rsp+40h] [rbp-B0h]
  __int64 v126; // [rsp+40h] [rbp-B0h]
  unsigned __int64 v127; // [rsp+48h] [rbp-A8h]
  __int64 v128; // [rsp+48h] [rbp-A8h]
  __int64 v129; // [rsp+48h] [rbp-A8h]
  __int64 v130; // [rsp+48h] [rbp-A8h]
  int v131; // [rsp+50h] [rbp-A0h]
  __int64 v132; // [rsp+50h] [rbp-A0h]
  unsigned __int16 v133; // [rsp+5Eh] [rbp-92h]
  unsigned int v134; // [rsp+60h] [rbp-90h]
  int v135; // [rsp+64h] [rbp-8Ch]
  int v136; // [rsp+68h] [rbp-88h]
  unsigned __int16 v137; // [rsp+6Ch] [rbp-84h]
  __int16 v138; // [rsp+6Eh] [rbp-82h]
  __m128i *v139; // [rsp+78h] [rbp-78h] BYREF
  __m128i v140; // [rsp+80h] [rbp-70h] BYREF
  __m128i v141; // [rsp+90h] [rbp-60h] BYREF
  __m128i v142; // [rsp+A0h] [rbp-50h] BYREF
  __m128i v143[4]; // [rsp+B0h] [rbp-40h] BYREF

  v1 = a1;
  v134 = dword_4F063F8;
  v140 = _mm_loadu_si128((const __m128i *)&qword_4D04A00);
  v133 = word_4F063FC[0];
  v141 = _mm_loadu_si128((const __m128i *)&word_4D04A10);
  v142 = _mm_loadu_si128(&xmmword_4D04A20);
  v143[0] = _mm_loadu_si128((const __m128i *)&unk_4D04A30);
  v136 = dword_4F07508[0];
  v138 = dword_4F07508[1];
  v135 = dword_4F061D8;
  v137 = word_4F061DC[0];
  v2 = *(_QWORD *)(a1 + 72);
  if ( v2 )
  {
    v3 = *(_QWORD *)(v2 + 16);
    if ( v3 )
    {
      if ( v3 == qword_4D03FF0 )
      {
        if ( qword_4F074B0 )
          goto LABEL_9;
        v10 = *(_QWORD *)(v2 + 16);
        v131 = 0;
        v11 = sub_8CFEE0(*(_QWORD *)(v1 + 32), v3);
        if ( !v11 )
          goto LABEL_9;
        goto LABEL_14;
      }
    }
    else
    {
      if ( qword_4F074B0 )
        goto LABEL_9;
      v132 = qword_4D03FF0;
      sub_721540(*(_QWORD *)v2);
      sub_8D0BC0(*(char **)(v2 + 8));
      dword_4F601E0 = 1;
      sub_8D0910(v132);
      sub_721540((__int64)qword_4F076B0);
      v3 = *(_QWORD *)(v2 + 16);
    }
    sub_8D0A80(v3);
    v131 = 1;
  }
  else
  {
    v131 = sub_8D0B70(*(_QWORD *)(a1 + 32));
  }
  v4 = *(_QWORD *)(v1 + 72);
  if ( !v4 )
    goto LABEL_21;
  if ( qword_4F074B0 )
    goto LABEL_7;
  v10 = *(_QWORD *)(v4 + 16);
  v11 = sub_8CFEE0(*(_QWORD *)(v1 + 32), v10);
  if ( !v11 )
    goto LABEL_7;
LABEL_14:
  v12 = *(_QWORD *)(v1 + 24);
  if ( (*(_BYTE *)(v11 + 81) & 0x10) != 0 )
  {
    v13 = **(_QWORD **)(v12 + 64);
    if ( (unsigned __int8)(*(_BYTE *)(v13 + 80) - 4) <= 1u )
    {
      if ( *(_QWORD *)(*(_QWORD *)(v13 + 96) + 72LL) )
      {
        v129 = v11;
        v33 = sub_8D02C0(**(_QWORD **)(v12 + 64), v10);
        v11 = v129;
        if ( v33
          && dword_4F077C4 == 2
          && (v126 = v129, v130 = *(_QWORD *)(v33 + 88), v73 = sub_8D23B0(v130), v11 = v126, v73)
          && (v74 = sub_8D3A70(v130), v11 = v126, v74) )
        {
          sub_8AD220(v130, 0);
          v12 = *(_QWORD *)(v1 + 24);
          v11 = v126;
        }
        else
        {
          v12 = *(_QWORD *)(v1 + 24);
        }
      }
    }
  }
  if ( *(_BYTE *)(v11 + 80) == 20 )
  {
    v24 = *(_QWORD *)(v12 + 88);
    v128 = v11;
    v25 = sub_72F240(*(const __m128i **)(v24 + 240));
    v26 = *(_BYTE *)(v24 + 203);
    v139 = v25;
    v1 = *(_QWORD *)(sub_8B74F0(v128, &v139, (v26 & 0x10) != 0, &dword_4F077C8) + 96);
  }
  else
  {
    v14 = sub_8CFEE0(v12, v10);
    if ( !v14 )
      goto LABEL_7;
    v1 = sub_892240(v14);
  }
  if ( v1 )
  {
    *(_BYTE *)(v1 + 80) &= ~0x80u;
    if ( !(unsigned int)sub_899CC0(v1, 0, 0) )
    {
      v21 = *(_QWORD *)(v1 + 24);
      v22 = *(_QWORD *)(v21 + 88);
      if ( ((*(_BYTE *)(v21 + 80) - 7) & 0xFD) != 0 )
        v23 = (*(_BYTE *)(v22 + 195) & 2) != 0;
      else
        v23 = *(_BYTE *)(v22 + 170) >> 7;
      if ( v23 )
        sub_6854C0(0x43Cu, (FILE *)(v21 + 48), v21);
      goto LABEL_7;
    }
LABEL_21:
    if ( (unsigned int)sub_890B60(*(_QWORD *)(v1 + 32)) )
    {
      *(_BYTE *)(v1 + 80) |= 2u;
    }
    else
    {
      v15 = *(_QWORD *)(v1 + 24);
      if ( ((*(_BYTE *)(v15 + 80) - 7) & 0xFD) != 0 )
      {
        if ( (unsigned __int64)qword_4F60190 <= 0xFF )
        {
          v125 = 0;
          ++qword_4F60190;
          if ( dword_4D0460C )
          {
            v125 = (const char *)_libc_calloc(300000, 1);
            sub_87D390((__int64)v125, v15, 299997, 1);
            v34 = strlen(v125);
            v125[v34] = 32;
            v125[v34 + 1] = 91;
            sub_87D420((__int64)&v125[v34 + 2], *(_QWORD *)(v1 + 24), 299997 - v34, 0);
            v35 = strlen(v125);
            v125[v35] = 93;
            v125[v35 + 1] = 0;
          }
          v16 = (unsigned __int64)v125;
          v17 = (__int64)"Instantiating Template Function";
          unk_4D045D8("Instantiating Template Function", v125);
          v20 = *(_QWORD *)(v1 + 32);
          v124 = *(_QWORD *)(v1 + 24);
          v127 = *(_QWORD *)(v124 + 88);
          switch ( *(_BYTE *)(v20 + 80) )
          {
            case 4:
            case 5:
              v27 = *(_QWORD *)(*(_QWORD *)(v20 + 96) + 80LL);
              break;
            case 6:
              v27 = *(_QWORD *)(*(_QWORD *)(v20 + 96) + 32LL);
              break;
            case 9:
            case 0xA:
              v27 = *(_QWORD *)(*(_QWORD *)(v20 + 96) + 56LL);
              break;
            case 0x13:
            case 0x14:
            case 0x15:
            case 0x16:
              v27 = *(_QWORD *)(v20 + 88);
              break;
            default:
              BUG();
          }
          v28 = *(_QWORD *)(v27 + 104);
          if ( v28 )
          {
            v108 = v27;
            v119 = *(_QWORD *)(v1 + 32);
            if ( (unsigned int)sub_825090() )
            {
              v17 = v28;
              sub_8250A0();
              v20 = v119;
              v27 = v108;
            }
            else
            {
              v17 = v28;
              v36 = sub_8250B0();
              v20 = v119;
              v27 = v108;
              if ( v36 )
              {
                v17 = v28;
                sub_8250C0();
                v27 = v108;
                v20 = v119;
              }
            }
          }
          v29 = *(_QWORD *)(v27 + 88);
          if ( !v29 || (*(_BYTE *)(v27 + 160) & 1) != 0 || *(_QWORD *)(v27 + 240) )
            v30 = v27 + 184;
          else
            v30 = *(_QWORD *)(v29 + 88) + 184LL;
          *(_BYTE *)(*(_QWORD *)(v1 + 16) + 28LL) |= 1u;
          if ( (*(_BYTE *)(v127 + 193) & 0x20) == 0 )
          {
            v31 = *(unsigned int *)(v127 + 160);
            if ( !(_DWORD)v31 )
            {
              v32 = *(_QWORD *)(v127 + 344);
              if ( !v32 )
              {
                v37 = *(_BYTE *)(v30 + 64);
                if ( (v37 & 0x10) == 0 )
                {
                  if ( (v37 & 8) != 0 )
                  {
                    if ( !*(_QWORD *)(v127 + 240) )
                    {
                      *(_BYTE *)(v127 + 206) |= 8u;
                      v17 = v127;
                      sub_71D150(v127, (__int64)v125, v18, v19, v20, v31);
                    }
                  }
                  else
                  {
                    v38 = v27;
                    v39 = v20;
                    if ( *(_BYTE *)(v20 + 80) == 20 )
                    {
                      v76 = *(_QWORD *)(v20 + 88);
                      v39 = *(_QWORD *)(v76 + 88);
                      if ( v39 && (*(_BYTE *)(v76 + 160) & 1) == 0 )
                      {
                        switch ( *(_BYTE *)(v39 + 80) )
                        {
                          case 4:
                          case 5:
                            v38 = *(_QWORD *)(*(_QWORD *)(v39 + 96) + 80LL);
                            goto LABEL_65;
                          case 6:
                            v38 = *(_QWORD *)(*(_QWORD *)(v39 + 96) + 32LL);
                            goto LABEL_65;
                          case 9:
                          case 0xA:
                            v38 = *(_QWORD *)(*(_QWORD *)(v39 + 96) + 56LL);
                            goto LABEL_65;
                          case 0x13:
                          case 0x14:
                          case 0x15:
                          case 0x16:
                            goto LABEL_117;
                          default:
                            BUG();
                        }
                      }
                      v39 = v20;
LABEL_117:
                      v38 = *(_QWORD *)(v39 + 88);
                    }
LABEL_65:
                    v120 = *(_QWORD *)(v38 + 176);
                    *(_BYTE *)(v127 + 203) = *(_BYTE *)(v120 + 203) & 1 | *(_BYTE *)(v127 + 203) & 0xFE;
                    if ( (*(_BYTE *)(v38 + 424) & 2) == 0 )
                    {
                      v83 = v27;
                      v85 = v38;
                      v90 = v39;
                      v114 = v20;
                      v75 = sub_893570(v20);
                      v20 = v114;
                      LODWORD(v31) = 0;
                      v32 = 0;
                      v38 = v85;
                      v27 = v83;
                      if ( v75 )
                      {
                        sub_8950B0(v90);
                        v27 = v83;
                        v38 = v85;
                        v32 = 0;
                        LODWORD(v31) = 0;
                        v20 = v114;
                      }
                    }
                    if ( (unsigned __int64)*(unsigned int *)(v27 + 40) >= unk_4D042F0 )
                    {
                      v16 = v124;
                      v17 = 456;
                      sub_6854E0(0x1C8u, v124);
                    }
                    else
                    {
                      if ( (*(_BYTE *)(v30 + 64) & 2) != 0 )
                      {
                        v86 = v27;
                        v91 = v38;
                        v97 = v32;
                        v104 = v31;
                        v115 = v20;
                        sub_736C90(v127, 1);
                        LODWORD(v31) = v104;
                        v32 = v97;
                        v38 = v91;
                        v27 = v86;
                        v20 = v115;
                        if ( !dword_4D04824 )
                        {
                          v78 = *(_BYTE *)(v127 + 88);
                          *(_BYTE *)(v127 + 172) = 2;
                          *(_BYTE *)(v127 + 200) &= 0xF8u;
                          *(_BYTE *)(v127 + 88) = v78 & 0x8F | 0x10;
                        }
                        if ( (*(_BYTE *)(v30 + 64) & 2) != 0 )
                          *(_BYTE *)(v127 + 205) |= 1u;
                      }
                      *(_QWORD *)(v124 + 48) = *(_QWORD *)(*(_QWORD *)(v27 + 176) + 64LL);
                      v40 = *(const __m128i **)(v127 + 152);
                      if ( v40[8].m128i_i8[12] == 12 )
                      {
                        v88 = v27;
                        v93 = v38;
                        v100 = v32;
                        v107 = v31;
                        v118 = v20;
                        v82 = sub_73EDA0(v40, 1);
                        v27 = v88;
                        v32 = v100;
                        LODWORD(v31) = v107;
                        *(_QWORD *)(v127 + 152) = v82;
                        v20 = v118;
                        v38 = v93;
                      }
                      v41 = *(_QWORD *)(v1 + 112);
                      if ( !v41 )
                      {
                        if ( (*(_BYTE *)(v124 + 81) & 0x10) != 0 )
                          v41 = *(_QWORD *)(v124 + 64);
                        v87 = v38;
                        v92 = v32;
                        v98 = v31;
                        v105 = v20;
                        v116 = v27;
                        v79 = sub_88DE40(*(_QWORD *)(v27 + 176), v41);
                        v38 = v87;
                        v32 = v92;
                        *(_QWORD *)(v1 + 112) = v79;
                        LODWORD(v31) = v98;
                        v41 = (__int64)v79;
                        v20 = v105;
                        v27 = v116;
                      }
                      v84 = v27;
                      v89 = v38;
                      v94 = v32;
                      v101 = v31;
                      v109 = v20;
                      sub_64A300(v127, v41);
                      v42 = v109;
                      v43 = v101;
                      v44 = v94;
                      v45 = v84;
                      if ( unk_4D04734 == 3 )
                      {
                        v77 = *(_BYTE *)(v127 + 88);
                        *(_BYTE *)(v127 + 172) = 2;
                        *(_BYTE *)(v127 + 200) &= 0xF8u;
                        *(_BYTE *)(v127 + 88) = v77 & 0x8F | 0x10;
                      }
                      else if ( *(_BYTE *)(v127 + 172) != 2 )
                      {
                        v46 = *(_BYTE *)(v127 + 88);
                        *(_BYTE *)(v127 + 172) = 0;
                        *(_BYTE *)(v127 + 88) = v46 & 0x8F | 0x20;
                      }
                      v47 = *(__int64 **)(v120 + 104);
                      if ( v47 )
                      {
                        v48 = *(__int64 **)(v127 + 104);
                        if ( v48 )
                        {
                          while ( (*((_BYTE *)v48 + 11) & 1) == 0 )
                          {
                            v48 = (__int64 *)*v48;
                            if ( !v48 )
                              goto LABEL_96;
                          }
                        }
                        else
                        {
LABEL_96:
                          while ( (*((_BYTE *)v47 + 11) & 1) == 0 )
                          {
                            v47 = (__int64 *)*v47;
                            if ( !v47 )
                              goto LABEL_79;
                          }
                          v96 = v101;
                          v103 = v44;
                          sub_892760(v127, v109, v89, 1);
                          v42 = v109;
                          v45 = v84;
                          v44 = v103;
                          v43 = v96;
                        }
                      }
LABEL_79:
                      v49 = *(_QWORD *)(v45 + 400);
                      ++*(_DWORD *)(v45 + 40);
                      if ( v49 && (*(_BYTE *)(v49 + 81) & 2) != 0 )
                      {
                        switch ( *(_BYTE *)(v49 + 80) )
                        {
                          case 4:
                          case 5:
                            v44 = *(_QWORD *)(*(_QWORD *)(v49 + 96) + 80LL);
                            break;
                          case 6:
                            v44 = *(_QWORD *)(*(_QWORD *)(v49 + 96) + 32LL);
                            break;
                          case 9:
                          case 0xA:
                            v44 = *(_QWORD *)(*(_QWORD *)(v49 + 96) + 56LL);
                            break;
                          case 0x13:
                          case 0x14:
                          case 0x15:
                          case 0x16:
                            v44 = *(_QWORD *)(v49 + 88);
                            break;
                          default:
                            break;
                        }
                        v99 = v45;
                        v106 = v43;
                        v117 = v42;
                        v80 = sub_892400(v44);
                        v52 = v117;
                        v51 = v106;
                        v121 = v80;
                        v53 = v99;
                      }
                      else
                      {
                        v95 = v43;
                        v102 = v42;
                        v110 = v45;
                        v50 = sub_892400(v45);
                        v51 = v95;
                        v52 = v102;
                        v121 = v50;
                        v53 = v110;
                      }
                      v111 = v53;
                      if ( (*(_BYTE *)(v53 + 160) & 1) != 0 )
                        v51 = (unsigned int)&dword_400000;
                      sub_864700(*(_QWORD *)(v121 + 32), 0, v127, v124, v52, *(_QWORD *)(v127 + 240), 1, v51);
                      v54 = v111;
                      if ( (*(_BYTE *)(v124 + 81) & 2) == 0 )
                      {
                        sub_8756F0(32770, v124, (_QWORD *)(v124 + 48), 0);
                        v54 = v111;
                      }
                      v112 = v54;
                      sub_854C10(*(const __m128i **)(v54 + 56));
                      sub_64F530(v124);
                      v55 = v112;
                      if ( !dword_4D048B8 )
                      {
                        sub_64A410(v30);
                        v55 = v112;
                      }
                      v113 = v55;
                      sub_7BC160(v121);
                      for ( i = *(_QWORD **)(v113 + 56); i; i = (_QWORD *)*i )
                      {
                        v58 = *(_BYTE *)(i[1] + 8LL);
                        if ( v58 == 6 )
                        {
                          *(_BYTE *)(v127 + 197) |= 8u;
                        }
                        else if ( v58 == 7 )
                        {
                          *(_BYTE *)(v127 + 197) |= 0x10u;
                        }
                      }
                      sub_71E0E0(v127, v30, 28, v127, v56);
                      v63 = v113;
                      if ( word_4F06418[0] == 74 )
                      {
                        sub_7B8B50(v127, (unsigned int *)v30, v59, v60, v61, v62);
                        v63 = v113;
                      }
                      v16 = 0;
                      v122 = v63;
                      sub_854980(v124, 0);
                      sub_863FE0(v124, 0, v64, v65, v66, v67);
                      v72 = v122;
                      --*(_DWORD *)(v122 + 40);
                      if ( word_4F06418[0] != 9 )
                      {
                        do
                          sub_7B8B50(v124, 0, v68, v69, v70, v71);
                        while ( word_4F06418[0] != 9 );
                        v72 = v122;
                      }
                      v123 = v72;
                      sub_7B8B50(v124, 0, v68, v69, v70, v71);
                      *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v1 + 24) + 88LL) + 88LL) |= 4u;
                      if ( *(_QWORD *)(v123 + 72) )
                      {
                        if ( (*(_BYTE *)(v124 + 81) & 0x10) == 0 )
                        {
                          v81 = *(_QWORD *)(*(_QWORD *)(sub_892400(v123) + 32) + 16LL);
                          if ( *(_BYTE *)(v81 + 28) == 6 )
                          {
                            if ( *(_QWORD *)(v81 + 32) )
                              *(_BYTE *)(v127 + 202) |= 0x80u;
                          }
                        }
                      }
                      v17 = v127;
                      sub_8CBAA0(v127);
                    }
                  }
                }
              }
            }
          }
          if ( dword_4D0460C )
          {
            v17 = (__int64)v125;
            _libc_free(v125, v16);
          }
          unk_4D045D0(v17, v16);
          --qword_4F60190;
        }
      }
      else if ( (*(_BYTE *)(v15 + 81) & 2) == 0 )
      {
        sub_8AA320((_QWORD *)v1, 0, 1u);
      }
    }
  }
LABEL_7:
  if ( v131 )
    sub_8D0B10();
LABEL_9:
  v5 = _mm_loadu_si128(&v140);
  v6 = _mm_loadu_si128(&v141);
  v7 = _mm_loadu_si128(&v142);
  dword_4F07508[0] = v136;
  LOWORD(dword_4F07508[1]) = v138;
  v8 = _mm_loadu_si128(v143);
  *(__m128i *)&qword_4D04A00 = v5;
  dword_4F063F8 = v134;
  *(__m128i *)&word_4D04A10 = v6;
  word_4F063FC[0] = v133;
  xmmword_4D04A20 = v7;
  dword_4F061D8 = v135;
  unk_4D04A30 = v8;
  word_4F061DC[0] = v137;
  return v137;
}
