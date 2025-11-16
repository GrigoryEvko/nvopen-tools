// Function: sub_6DC380
// Address: 0x6dc380
//
__int64 __fastcall sub_6DC380(__int64 *a1, __int64 a2, unsigned int a3, _QWORD *a4, __m128i *a5, unsigned int a6)
{
  unsigned __int64 v6; // r15
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // r13
  __int64 v12; // rax
  __int64 v13; // rdi
  int v14; // edx
  unsigned int v15; // edx
  unsigned __int8 v16; // cl
  __int64 v17; // rdi
  __int16 v18; // ax
  _QWORD *v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r8
  __int64 v22; // r9
  unsigned __int64 v23; // rdi
  __int64 v24; // rcx
  __int64 v25; // r9
  __int64 v26; // rax
  __int64 result; // rax
  __int64 v28; // rdx
  __int64 v29; // rax
  __m128i v30; // xmm0
  __m128i v31; // xmm1
  __m128i v32; // xmm2
  __m128i v33; // xmm3
  __m128i v34; // xmm4
  __m128i v35; // xmm5
  __m128i v36; // xmm6
  __m128i v37; // xmm7
  char v38; // al
  __m128i v39; // xmm0
  bool v40; // zf
  __int64 v41; // rax
  __int64 v42; // rdx
  __int64 v43; // rdi
  __int64 v44; // r8
  __int64 v45; // r9
  char v46; // r11
  __int64 v47; // rcx
  __int64 v48; // rdx
  __int64 v49; // rsi
  char v50; // al
  __int64 v51; // r11
  _BOOL4 v52; // eax
  __int16 v53; // si
  __int64 v54; // r8
  char v55; // al
  __int64 v56; // r8
  __int64 v57; // rax
  __int64 v58; // rcx
  __int64 v59; // r8
  __int64 *v60; // r13
  __int64 v61; // rax
  __int64 v62; // rdx
  __int64 v63; // rdi
  unsigned int v64; // eax
  _BYTE *v65; // rdx
  __int64 v66; // rcx
  __int64 v67; // rax
  int v68; // r9d
  unsigned int v69; // r8d
  __int64 v70; // rdx
  __int64 v71; // rcx
  __int64 v72; // rax
  __int64 v73; // rax
  __int64 v74; // rax
  __int64 v75; // rax
  __int64 v76; // rax
  __int64 v77; // rax
  __int64 v78; // rax
  __int64 v79; // rax
  __int64 v80; // r8
  _QWORD *v81; // rdx
  __int64 v82; // r13
  int v83; // eax
  __int64 v84; // rax
  __int64 *v85; // rcx
  __int64 *v86; // r9
  __int64 v87; // rax
  __int64 v88; // rax
  _QWORD *v89; // rdx
  __int64 v90; // r13
  __int64 *v91; // rax
  __int64 v92; // rax
  __int64 v93; // rax
  unsigned int v94; // r13d
  __int64 v95; // rax
  __int64 v96; // rax
  __int64 v97; // rdx
  __int64 v98; // rsi
  int v99; // r8d
  __int64 *v100; // rdx
  __int64 v101; // rcx
  __int64 v102; // rdx
  __int64 v103; // rdi
  __int64 v104; // rax
  char v105; // dl
  __int64 v106; // r13
  __int64 *v107; // rax
  __int64 v108; // rax
  __int64 v109; // rdx
  _QWORD *v110; // rax
  unsigned int v111; // r11d
  __m128i v112; // xmm2
  __m128i v113; // xmm3
  __m128i v114; // xmm4
  __m128i v115; // xmm5
  __m128i v116; // xmm6
  __m128i v117; // xmm7
  __m128i v118; // xmm1
  __m128i v119; // xmm2
  __m128i v120; // xmm3
  __m128i v121; // xmm4
  __m128i v122; // xmm5
  __m128i v123; // xmm6
  __int64 v124; // [rsp-10h] [rbp-730h]
  __int64 v125; // [rsp-10h] [rbp-730h]
  __int64 v126; // [rsp-10h] [rbp-730h]
  __int64 v127; // [rsp-10h] [rbp-730h]
  __int64 v128; // [rsp-8h] [rbp-728h]
  __int64 *v129; // [rsp+8h] [rbp-718h]
  __int64 *v130; // [rsp+8h] [rbp-718h]
  _BOOL4 v131; // [rsp+10h] [rbp-710h]
  __int64 v132; // [rsp+10h] [rbp-710h]
  __int64 v133; // [rsp+10h] [rbp-710h]
  __int64 *v134; // [rsp+10h] [rbp-710h]
  int v135; // [rsp+10h] [rbp-710h]
  __int64 v136; // [rsp+18h] [rbp-708h]
  unsigned int v137; // [rsp+18h] [rbp-708h]
  unsigned __int16 v138; // [rsp+18h] [rbp-708h]
  __int64 v139; // [rsp+18h] [rbp-708h]
  __int64 v140; // [rsp+20h] [rbp-700h]
  unsigned int v141; // [rsp+20h] [rbp-700h]
  __int64 v142; // [rsp+28h] [rbp-6F8h]
  __int64 v145; // [rsp+38h] [rbp-6E8h]
  _BOOL4 v146; // [rsp+38h] [rbp-6E8h]
  __int64 v147; // [rsp+38h] [rbp-6E8h]
  __int64 v148; // [rsp+38h] [rbp-6E8h]
  __int64 v149; // [rsp+40h] [rbp-6E0h]
  __int64 v150; // [rsp+48h] [rbp-6D8h]
  int v151; // [rsp+48h] [rbp-6D8h]
  char v152; // [rsp+50h] [rbp-6D0h]
  unsigned int v153; // [rsp+54h] [rbp-6CCh]
  __int64 v154; // [rsp+58h] [rbp-6C8h]
  int v155; // [rsp+64h] [rbp-6BCh] BYREF
  unsigned int v156; // [rsp+68h] [rbp-6B8h] BYREF
  int v157; // [rsp+6Ch] [rbp-6B4h] BYREF
  __int64 v158; // [rsp+70h] [rbp-6B0h] BYREF
  __int64 v159; // [rsp+78h] [rbp-6A8h] BYREF
  _QWORD v160[2]; // [rsp+80h] [rbp-6A0h] BYREF
  unsigned int v161; // [rsp+90h] [rbp-690h]
  __int64 v162; // [rsp+94h] [rbp-68Ch]
  __int64 v163; // [rsp+A0h] [rbp-680h]
  char v164[18]; // [rsp+B0h] [rbp-670h] BYREF
  __int16 v165; // [rsp+C2h] [rbp-65Eh]
  _BYTE v166[352]; // [rsp+150h] [rbp-5D0h] BYREF
  __int64 v167[44]; // [rsp+2B0h] [rbp-470h] BYREF
  __m128i v168; // [rsp+410h] [rbp-310h] BYREF
  __m128i v169; // [rsp+420h] [rbp-300h]
  __m128i v170; // [rsp+430h] [rbp-2F0h]
  __m128i v171; // [rsp+440h] [rbp-2E0h]
  __m128i v172; // [rsp+450h] [rbp-2D0h]
  __m128i v173; // [rsp+460h] [rbp-2C0h]
  __m128i v174; // [rsp+470h] [rbp-2B0h]
  __m128i v175; // [rsp+480h] [rbp-2A0h]
  __m128i v176; // [rsp+490h] [rbp-290h]
  __m128i v177; // [rsp+4A0h] [rbp-280h]
  __m128i v178; // [rsp+4B0h] [rbp-270h]
  __m128i v179; // [rsp+4C0h] [rbp-260h]
  __m128i v180; // [rsp+4D0h] [rbp-250h]
  __m128i v181; // [rsp+4E0h] [rbp-240h]
  __m128i v182; // [rsp+4F0h] [rbp-230h]
  __m128i v183; // [rsp+500h] [rbp-220h]
  __m128i v184; // [rsp+510h] [rbp-210h]
  __m128i v185; // [rsp+520h] [rbp-200h]
  __m128i v186; // [rsp+530h] [rbp-1F0h]
  __m128i v187; // [rsp+540h] [rbp-1E0h]
  __m128i v188; // [rsp+550h] [rbp-1D0h]
  __m128i v189; // [rsp+560h] [rbp-1C0h]
  char v190[432]; // [rsp+570h] [rbp-1B0h] BYREF

  v6 = a2;
  v154 = *(_QWORD *)a2;
  v153 = *(_DWORD *)(a2 + 40);
  v142 = *a1;
  if ( unk_4F074B0 && (unsigned int)sub_893F30(*(_QWORD *)(a2 + 24)) )
  {
    result = sub_6E6260(a4);
    *(_BYTE *)(a2 + 56) = 1;
    return result;
  }
  if ( !a5 )
    a5 = (__m128i *)v166;
  v9 = sub_6E4240(a1, &v158);
  v10 = v158;
  v11 = v9;
  if ( !v158 )
    v10 = sub_6E3DA0(v9, v190);
  v150 = v10;
  sub_68A670(a2, (__int64)v160);
  v12 = qword_4D03C50;
  v13 = *(unsigned __int8 *)(v150 + 352);
  if ( qword_4D03C50 && (_BYTE)v13 == *(_BYTE *)(qword_4D03C50 + 16LL) && (*(_BYTE *)(qword_4D03C50 + 18LL) & 0x40) != 0 )
  {
    v151 = 0;
    v15 = *(_DWORD *)(a2 + 40);
  }
  else
  {
    sub_6E2140(v13, v164, 0, 0, a2);
    v14 = *(_DWORD *)(a2 + 40);
    v12 = qword_4D03C50;
    v151 = 1;
    v165 |= 0x2C0u;
    v15 = v14 & 0xFFFFFFFB;
    *(_DWORD *)(a2 + 40) = v15;
    if ( *(_BYTE *)(v12 + 16) > 2u )
    {
      v15 |= 4u;
      *(_DWORD *)(a2 + 40) = v15;
    }
  }
  v149 = *(_QWORD *)(v12 + 128);
  v152 = 0;
  if ( (v15 & 0x800) != 0 )
  {
    v16 = *(_BYTE *)(v12 + 19);
    v152 = v16 >> 7;
    *(_BYTE *)(v12 + 19) = v16 | 0x80;
    v15 = *(_DWORD *)(a2 + 40);
  }
  if ( (v15 & 0x86140) == 0 )
    *(_BYTE *)(v12 + 19) &= ~2u;
  if ( v158 )
    *(_QWORD *)(v12 + 128) = v158;
  *(_QWORD *)a2 = v11;
  v17 = v11;
  v18 = sub_687AF0(v11, &v155, &v156, 0);
  *(_WORD *)(a2 + 8) = v18;
  if ( v18 == 1 )
  {
    v159 = 0;
    v17 = v11;
    v140 = sub_6E3DA0(v11, 0);
    switch ( *(_BYTE *)(v11 + 24) )
    {
      case 1:
        v51 = **(_QWORD **)(*(_QWORD *)(*(_QWORD *)(v11 + 72) + 16LL) + 56LL);
        v52 = 0;
        goto LABEL_85;
      case 2:
        v43 = *(_QWORD *)(v11 + 56);
        v159 = 0;
        v44 = *(_QWORD *)(a2 + 48);
        v45 = *(unsigned int *)(a2 + 40);
        v46 = *(_BYTE *)(v43 + 176);
        v47 = v140 + 68;
        v48 = *(_QWORD *)(a2 + 32);
        v49 = *(_QWORD *)(a2 + 24);
        v50 = *(_BYTE *)(v140 + 19);
        if ( v46 == 4 )
        {
          v43 = *(_QWORD *)(v43 + 184);
          goto LABEL_74;
        }
        if ( v46 != 11 )
        {
LABEL_74:
          if ( v50 < 0 )
            v45 = (unsigned int)v45 | 0x20;
          v51 = sub_730C30(v43, v49, v48, v47, v44, v45);
          v52 = 0;
          goto LABEL_77;
        }
        v159 = *(_QWORD *)(v43 + 192);
        v103 = *(_QWORD *)(v43 + 184);
        if ( v50 < 0 )
          v45 = (unsigned int)v45 | 0x20;
        v104 = sub_730C30(v103, v49, v48, v47, v44, v45);
        v51 = v104;
        if ( !v104 )
        {
LABEL_192:
          *(_BYTE *)(v6 + 56) = 1;
          sub_6E6260(a4);
          *(_QWORD *)((char *)a4 + 68) = *(_QWORD *)(v140 + 68);
          *(_QWORD *)((char *)a4 + 76) = *(_QWORD *)(v140 + 76);
          goto LABEL_81;
        }
        v105 = *(_BYTE *)(v104 + 80);
        v106 = v104;
        if ( v105 == 16 )
        {
          v107 = *(__int64 **)(v104 + 88);
          v106 = *v107;
          v105 = *(_BYTE *)(*v107 + 80);
        }
        if ( v105 == 24 )
        {
          v106 = *(_QWORD *)(v106 + 88);
          v105 = *(_BYTE *)(v106 + 80);
        }
        v52 = 1;
        if ( v105 == 21 )
        {
          v168.m128i_i32[0] = 0;
          if ( !v159
            || (v139 = v51,
                v108 = sub_8A55D0(
                         v106,
                         v159,
                         **(_QWORD **)(*(_QWORD *)(v106 + 88) + 32LL),
                         0,
                         *(_QWORD *)(v6 + 24),
                         *(_QWORD *)(v6 + 32),
                         v140 + 68,
                         *(_DWORD *)(v6 + 40),
                         (__int64)&v168,
                         *(_QWORD *)(v6 + 48)),
                v51 = v139,
                v159 = v108,
                !v168.m128i_i32[0]) )
          {
            v51 = sub_8C0230(v106, &v159, 0, 1, 0);
          }
          v52 = 1;
LABEL_77:
          if ( v51 )
            goto LABEL_78;
          goto LABEL_192;
        }
        goto LABEL_78;
      case 3:
        v63 = *(_QWORD *)(v11 + 56);
        if ( (*(_BYTE *)(v63 + 170) & 0x40) == 0 )
        {
          v51 = *(_QWORD *)v63;
          if ( (*(_BYTE *)(v63 + 89) & 1) != 0 && *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 4) != 14 )
          {
            v64 = dword_4F04C60;
            v137 = dword_4F04C60;
            v65 = (_BYTE *)(qword_4F04C68[0] + 776LL * (int)dword_4F04C60 + 4);
            do
            {
              v66 = v64 - 1;
              if ( *v65 == 10 )
              {
                dword_4F04C60 = v64 - 1;
                goto LABEL_107;
              }
              --v64;
              v65 -= 776;
            }
            while ( v64 != -1 );
            v66 = dword_4F04C60;
LABEL_107:
            if ( (*(_BYTE *)(v63 + 172) & 1) != 0 )
            {
              v132 = v51;
              if ( (unsigned int)sub_830940(v167, &v168, v65, v66) )
              {
                sub_830B80(v167[0], v168.m128i_i64[0], (*(_BYTE *)(v11 + 27) & 2) != 0, v11 + 28, v11 + 28, a4);
              }
              else
              {
                *(_BYTE *)(a2 + 56) = 1;
                sub_6E6260(a4);
              }
              v51 = v132;
              *(_QWORD *)((char *)a4 + 68) = *(_QWORD *)(v140 + 68);
              *(_QWORD *)((char *)a4 + 76) = *(_QWORD *)(v140 + 76);
LABEL_111:
              dword_4F04C60 = v137;
              v52 = 0;
              goto LABEL_85;
            }
            sub_878710(v51, &v168);
            if ( (v169.m128i_i8[1] & 0x40) == 0 )
            {
              v169.m128i_i8[0] &= ~0x80u;
              v169.m128i_i64[1] = 0;
            }
            v51 = sub_7D5DD0(&v168, 0);
            if ( v51 )
              goto LABEL_111;
            *(_BYTE *)(a2 + 56) = 1;
            sub_6E6260(a4);
            *(_QWORD *)((char *)a4 + 68) = *(_QWORD *)(v140 + 68);
            *(_QWORD *)((char *)a4 + 76) = *(_QWORD *)(v140 + 76);
            dword_4F04C60 = v137;
LABEL_81:
            sub_6E4BC0(a4, v140);
            *((_BYTE *)a4 + 18) &= ~1u;
            a2 = v140;
            v23 = (unsigned __int64)a4;
            a4[16] = 0;
            sub_6E5010(a4, v140);
            if ( *((_BYTE *)a4 + 17) == 3 )
            {
              v55 = *((_BYTE *)a4 + 19);
              if ( (v55 & 2) != 0 )
              {
                *((_BYTE *)a4 + 19) = v55 & 0xFD;
                v23 = (unsigned __int64)a4;
                a2 = v140 + 120;
                sub_6F5FA0(a4, v140 + 120, 0, 0, v54, v25);
              }
            }
            goto LABEL_30;
          }
          v52 = 0;
LABEL_85:
          if ( v51 )
          {
LABEL_78:
            v53 = a3 & 4;
            if ( (a3 & 4) != 0 )
              v53 = 64;
            sub_6CC940(a4, v53, 0, v6, v51, v140, v52, v159, 0);
          }
          goto LABEL_81;
        }
        v96 = *(_QWORD *)a2;
        v97 = *(_QWORD *)(a2 + 32);
        v127 = *(_QWORD *)(a2 + 48);
        v98 = *(_QWORD *)(a2 + 24);
        v99 = *(_DWORD *)(v6 + 40);
        v168.m128i_i32[0] = 0;
        v51 = sub_8C2940(v63, v98, v97, (int)v96 + 28, v99, (unsigned int)&v168, v127);
        if ( v51 )
        {
          v100 = *(__int64 **)(v63 + 216);
          v52 = 0;
          if ( v100 )
          {
            if ( *(_BYTE *)(v51 + 80) == 2 )
            {
              v101 = *(_QWORD *)(v51 + 88);
              if ( v101 )
              {
                if ( *(_BYTE *)(v101 + 173) == 12 )
                {
                  v102 = *v100;
                  if ( v102 )
                  {
LABEL_178:
                    v159 = v102;
                    v52 = 1;
                  }
                }
              }
            }
          }
LABEL_98:
          if ( !v168.m128i_i32[0] )
            goto LABEL_85;
        }
        else
        {
LABEL_180:
          v168.m128i_i32[0] = 1;
          v52 = 0;
        }
        *(_BYTE *)(v6 + 56) = 1;
        v131 = v52;
        v136 = v51;
        sub_6E6260(a4);
        v51 = v136;
        *(_QWORD *)((char *)a4 + 68) = *(_QWORD *)(v140 + 68);
        v52 = v131;
        *(_QWORD *)((char *)a4 + 76) = *(_QWORD *)(v140 + 76);
        goto LABEL_85;
      case 0x14:
        v60 = *(__int64 **)(v11 + 56);
        if ( !v60[31] )
        {
          v51 = *v60;
          v52 = 0;
          goto LABEL_85;
        }
        v61 = *(_QWORD *)a2;
        v124 = *(_QWORD *)(a2 + 48);
        v62 = *(_QWORD *)(a2 + 32);
        v168.m128i_i32[0] = 0;
        v51 = sub_8C27C0(
                (_DWORD)v60,
                *(_QWORD *)(a2 + 24),
                v62,
                (int)v61 + 28,
                *(_DWORD *)(a2 + 40),
                (unsigned int)&v168,
                v124);
        if ( !v51 )
          goto LABEL_180;
        v52 = 0;
        if ( *(_BYTE *)(v51 + 80) != 2 )
          goto LABEL_98;
        v109 = *(_QWORD *)(v51 + 88);
        if ( !v109 )
          goto LABEL_98;
        if ( *(_BYTE *)(v109 + 173) != 12 )
          goto LABEL_98;
        v102 = v60[30];
        if ( !v102 )
          goto LABEL_98;
        goto LABEL_178;
      case 0x18:
        v56 = sub_730FF0(v11, 0, v42);
        if ( *(_DWORD *)(v56 + 56) )
          *(_BYTE *)(v56 + 25) = *(_BYTE *)(v56 + 25) & 0xFC | 1;
        v145 = v56;
        v57 = sub_866660(v56);
        v59 = v145;
        if ( !v57 )
        {
          v57 = sub_6E3F00(*(_QWORD *)v11, a2, v140, v58, v145);
          v59 = v145;
        }
        if ( (*(_BYTE *)(v59 + 25) & 3) == 0 )
        {
          *(_QWORD *)(v59 + 8) = v57;
          v148 = v59;
          v57 = sub_73D720(v57);
          v59 = v148;
        }
        *(_QWORD *)v59 = v57;
        sub_6E7170(v59, a4);
        if ( (unsigned int)sub_8D32E0(*a4) )
          sub_6F82C0(a4);
        goto LABEL_81;
      default:
        goto LABEL_48;
    }
  }
  if ( v18 == 4 )
  {
    v23 = *(_QWORD *)(v11 + 56);
    a2 = (__int64)a4;
    sub_6E6A50(v23, a4);
    if ( *((_BYTE *)a4 + 16) == 2 )
      a4[36] = 0;
    goto LABEL_30;
  }
  if ( !v155 )
  {
    switch ( v18 )
    {
      case 25:
        a2 = 0;
        v23 = 0;
        sub_6D30E0(0, 0, (_QWORD *)v6, (__int64)a4);
        goto LABEL_30;
      case 27:
        a2 = 0;
        v23 = 0;
        sub_6C0E20(0, 0, (_BYTE *)v6, (__m128i *)a4);
        goto LABEL_30;
      case 29:
      case 30:
        v23 = 0;
        sub_6D7FC0(0, a2, (a3 & 0x800) != 0, (a3 >> 7) & 1, a4, a5);
        goto LABEL_30;
      case 33:
      case 50:
      case 51:
        v23 = 0;
        sub_6B38B0(0, a2, (__int64)a4, v20);
        goto LABEL_30;
      case 34:
      case 39:
      case 40:
        v23 = 0;
        sub_6B1D00(0, a2, (__int64)a4, v20);
        goto LABEL_30;
      case 35:
      case 36:
        v23 = 0;
        sub_6B21E0(0, a2, (__int64)a4, v20);
        goto LABEL_30;
      case 41:
      case 42:
        v23 = 0;
        sub_6B2B40(0, a2, (__int64)a4, v20);
        goto LABEL_30;
      case 43:
      case 44:
      case 45:
      case 46:
        v23 = 0;
        sub_6B2F50(0, a2, (__int64)a4, v20);
        goto LABEL_30;
      case 47:
      case 48:
        v23 = 0;
        sub_6B3030(0, a2, a4, v20);
        goto LABEL_30;
      case 49:
        sub_6F8AB0(a2, (unsigned int)v167, (unsigned int)&v168, 0, (unsigned int)&v159, (unsigned int)&v157, 0);
        a2 = (__int64)&v168;
        v23 = (unsigned __int64)v167;
        sub_68FEF0(v167, &v168, &v159, v157, 0, (__int64)a4);
        goto LABEL_30;
      case 52:
      case 53:
        v23 = 0;
        sub_6B3BD0(0, a2, 0, (__int64)a4);
        goto LABEL_30;
      case 54:
        v23 = 0;
        sub_6B4800(0, a2, (__int64)a4);
        goto LABEL_30;
      case 56:
        v23 = 0;
        sub_6CEC90(0, (__int64 *)a2, &v168, a4, v21);
        goto LABEL_30;
      case 57:
      case 58:
      case 59:
      case 60:
      case 61:
      case 62:
      case 63:
      case 64:
      case 65:
      case 66:
        v23 = 0;
        sub_6CF140(0, (__int64 *)a2, &v168, a4, v21);
        goto LABEL_30;
      case 67:
        v23 = 0;
        sub_6B70D0(0, a2, (__int64)a4);
        goto LABEL_30;
      case 70:
      case 71:
        v23 = 0;
        sub_6B3200(0, a2, (__int64)a4, v20);
        goto LABEL_30;
      case 73:
        v147 = sub_6E3DA0(v11, 0);
        v90 = sub_690A60(*(_QWORD *)(v11 + 56), a2, v89);
        v91 = (__int64 *)sub_6E2F40(1);
        a2 = (__int64)a4;
        v91[3] = v90;
        v23 = (unsigned __int64)v91;
        v91[4] = *(_QWORD *)(v147 + 68);
        v91[5] = *(_QWORD *)(v147 + 76);
        sub_6E9FE0(v91, a4);
        goto LABEL_30;
      case 76:
        v80 = *(_QWORD *)(*(_QWORD *)a2 + 56LL);
        v81 = (_QWORD *)*(unsigned __int8 *)(*(_QWORD *)a2 + 66LL);
        v82 = *(_QWORD *)(*(_QWORD *)a2 + 80LL);
        v146 = *(_QWORD *)(v80 + 16) == 0;
        v141 = (unsigned __int8)v81 & 1;
        v138 = *(_WORD *)(*(_QWORD *)a2 + 64LL);
        v83 = *(_DWORD *)(a2 + 40);
        v129 = (__int64 *)(*(_QWORD *)a2 + 28LL);
        if ( (v83 & 0x40) != 0 )
        {
          v133 = *(_QWORD *)(*(_QWORD *)a2 + 56LL);
          *(_DWORD *)(a2 + 40) = v83 & 0xFFFFFFBF;
          v84 = sub_6E3060(*(_QWORD *)(v80 + 80));
          v85 = v129;
          v86 = (__int64 *)v84;
          v87 = *(_QWORD *)(v133 + 16);
          if ( v87 )
          {
            v130 = v86;
            v134 = v85;
            v88 = sub_6E3060(*(_QWORD *)(v87 + 80));
            v86 = v130;
            v85 = v134;
            *v130 = v88;
          }
          a2 = (__int64)a5;
          v23 = (unsigned __int64)a4;
          sub_6D02D0((__m128i *)a4, a5, (_QWORD *)(v82 + 68), v85, (_QWORD *)(v82 + 76), v86, v138, v146, v141, 1u);
          *(_DWORD *)(v6 + 40) |= 0x40u;
        }
        else
        {
          v135 = *(_DWORD *)(a2 + 40) & 0x40;
          v110 = (_QWORD *)sub_690A60(v80, a2, v81);
          v111 = v135;
          if ( v110 )
            v111 = v110[2] != 0;
          v23 = (unsigned __int64)a4;
          a2 = (__int64)a5;
          sub_6D02D0((__m128i *)a4, a5, (_QWORD *)(v82 + 68), v129, (_QWORD *)(v82 + 76), v110, v138, v146, v141, v111);
        }
        goto LABEL_30;
      case 112:
        a2 = (__int64)a4;
        v23 = v6;
        sub_6AF090((_QWORD *)v6, (__int64)a4, (__int64)v19, v20);
        goto LABEL_30;
      case 117:
        a2 = (__int64)a4;
        v23 = v6;
        sub_6DBAB0(v6, (__int64)a4, (__int64)v19, v20);
        goto LABEL_30;
      case 147:
      case 148:
        v23 = 0;
        sub_6B0A80(0, (__int64 *)a2, 0, (__int64)a4, a5, v22);
        goto LABEL_30;
      case 207:
      case 209:
      case 210:
      case 211:
      case 288:
      case 289:
      case 290:
      case 291:
      case 292:
      case 298:
      case 299:
        a2 = (__int64)a4;
        v23 = v6;
        sub_6A9D40((_QWORD *)v6, (__int64)a4);
        goto LABEL_30;
      case 225:
        v93 = sub_68AFD0(0x70u);
        v68 = 1;
        v69 = 1;
        v125 = (__int64)a4;
        v70 = v93;
        v71 = 1;
        a2 = 112;
        goto LABEL_113;
      case 226:
        v92 = sub_68AFD0(0x71u);
        v68 = 1;
        v69 = 1;
        v125 = (__int64)a4;
        v70 = v92;
        v71 = 1;
        a2 = 113;
        goto LABEL_113;
      case 227:
        v79 = sub_68AFD0(0x1Eu);
        v68 = 1;
        v69 = 1;
        v125 = (__int64)a4;
        v70 = v79;
        v71 = 1;
        a2 = 30;
        goto LABEL_113;
      case 228:
        v78 = sub_68AFD0(0x1Fu);
        v68 = 1;
        v69 = 1;
        v125 = (__int64)a4;
        v70 = v78;
        v71 = 1;
        a2 = 31;
        goto LABEL_113;
      case 233:
        v77 = sub_68AFD0(0x2Du);
        v68 = 0;
        v69 = 1;
        v125 = (__int64)a4;
        v70 = v77;
        v71 = 1;
        a2 = 45;
        goto LABEL_113;
      case 234:
        v76 = sub_68AFD0(0x2Eu);
        v68 = 0;
        v69 = 1;
        v125 = (__int64)a4;
        v70 = v76;
        v71 = 1;
        a2 = 46;
        goto LABEL_113;
      case 257:
      case 258:
        a2 = (__int64)a4;
        v23 = v6;
        sub_6D3EC0(v6, (__int64)a4, (__int64)v19, v20);
        goto LABEL_30;
      case 259:
        a2 = (__int64)a4;
        v23 = v6;
        sub_6AFBA0(v6, (__int64)a4, (__int64)v19, v20);
        goto LABEL_30;
      case 261:
        a2 = (__int64)a4;
        v23 = v6;
        sub_6B0100(v6, (__int64)a4, (__int64)v19, v20);
        goto LABEL_30;
      case 270:
        v75 = sub_68AFD0(0x3Cu);
        v68 = 0;
        v69 = 1;
        v125 = (__int64)a4;
        v70 = v75;
        v71 = 1;
        a2 = 60;
        goto LABEL_113;
      case 297:
        a2 = (__int64)a4;
        v23 = v6;
        sub_6AA570(v6, (__int64)a4, (__int64)v19, v20);
        goto LABEL_30;
      case 300:
      case 301:
        a2 = (__int64)a4;
        v23 = v6;
        sub_6A9E80((_QWORD *)v6, (__int64)a4);
        goto LABEL_30;
      case 302:
      case 303:
        a2 = (__int64)a4;
        v23 = v6;
        sub_6A9F70((_QWORD *)v6, (__int64)a4);
        goto LABEL_30;
      case 304:
        v94 = *(unsigned __int8 *)(*(_QWORD *)a2 + 56LL);
        v95 = sub_68AFD0(*(_BYTE *)(*(_QWORD *)a2 + 56LL));
        v68 = 0;
        a2 = v94;
        v125 = (__int64)a4;
        v70 = v95;
        v69 = 1;
        v71 = 2;
LABEL_113:
        v23 = v6;
        sub_6A9320((_QWORD *)v6, a2, v70, v71, v69, v68, v125);
        v25 = v126;
        if ( !dword_4D044B0 )
        {
          v23 = (unsigned __int64)a4;
          sub_6E6840(a4);
        }
        goto LABEL_30;
      default:
        goto LABEL_48;
    }
  }
  if ( (unsigned __int16)v18 > 0x11Cu )
    goto LABEL_48;
  if ( (unsigned __int16)v18 > 0xAFu )
  {
    switch ( v18 )
    {
      case 176:
        a2 = (__int64)a4;
        v23 = v6;
        sub_6AEB80((__int64 *)v6, (__int64)a4);
        goto LABEL_30;
      case 177:
        a2 = (__int64)a4;
        v23 = v6;
        sub_6C0800(v6, (__int64)a4);
        goto LABEL_30;
      case 178:
        a2 = (__int64)a4;
        v23 = v6;
        sub_6A2D50(v6, (__int64)a4, (__int64)v19, v20, v21, v22);
        goto LABEL_30;
      case 183:
        a2 = 0;
        v23 = v6;
        sub_6CA0E0(v6, 0, 0, 0, 0, 0, a4, a3);
        goto LABEL_30;
      case 195:
        a2 = (__int64)a4;
        v23 = v6;
        sub_6A98C0((_QWORD *)v6, (__int64)a4);
        goto LABEL_30;
      case 229:
        v74 = sub_68AFD0(0x29u);
        v68 = 1;
        v69 = 1;
        v125 = (__int64)a4;
        v70 = v74;
        v71 = 1;
        a2 = 41;
        goto LABEL_113;
      case 230:
        v73 = sub_68AFD0(0x2Au);
        v68 = 0;
        v69 = 0;
        v125 = (__int64)a4;
        v70 = v73;
        v71 = 1;
        a2 = 42;
        goto LABEL_113;
      case 231:
      case 232:
        v67 = sub_68AFD0(0x2Bu);
        v68 = 0;
        v69 = 0;
        v125 = (__int64)a4;
        v70 = v67;
        v71 = 1;
        a2 = 43;
        goto LABEL_113;
      case 235:
        v72 = sub_68AFD0(0x37u);
        v68 = 0;
        v69 = 0;
        v125 = (__int64)a4;
        v70 = v72;
        v71 = 1;
        a2 = 55;
        goto LABEL_113;
      case 243:
        a2 = (__int64)a4;
        v23 = v6;
        sub_6ABC20(v6, (__int64)a4, (__int64)v19, v20, v21, v22);
        goto LABEL_30;
      case 247:
        goto LABEL_116;
      case 271:
        a2 = (__int64)a4;
        v23 = v6;
        sub_6A8930(v6, (__m128i *)a4, (__int64)v19, v20);
        goto LABEL_30;
      case 284:
        goto LABEL_117;
      default:
        goto LABEL_48;
    }
  }
  if ( (unsigned __int16)v18 > 0xA7u )
    goto LABEL_48;
  if ( (unsigned __int16)v18 > 0x62u )
  {
    switch ( v18 )
    {
      case 99:
LABEL_117:
        a2 = (__int64)a4;
        v23 = v6;
        sub_6A6540((_QWORD *)v6, (__m128i *)a4, (__int64)v19, v20, v21, v22);
        goto LABEL_30;
      case 111:
LABEL_116:
        a2 = (__int64)a4;
        v23 = v6;
        sub_6A7EC0((_WORD *)v6, (__int64)a4, (__int64)v19, v20, v21, v22);
        goto LABEL_30;
      case 144:
      case 145:
        a2 = (__int64)a4;
        v23 = v6;
        sub_6AABA0((_WORD *)v6, (__m128i *)a4, (__int64)v19, v20);
        goto LABEL_30;
      case 152:
        a2 = (__int64)a4;
        v23 = v6;
        sub_6D53F0(v6, a4, (__int64)v19, v20, v21, v22);
        goto LABEL_30;
      case 155:
        a2 = (__int64)a4;
        v23 = v6;
        sub_6C6830(v6, (unsigned int *)a4, (__int64)v19, v20, v21, v22);
        goto LABEL_30;
      case 162:
        a2 = (__int64)a4;
        v23 = v6;
        sub_6AF3D0(v6, (__int64)a4, v19, v20, v21);
        goto LABEL_30;
      case 166:
        a2 = (__int64)a4;
        v23 = v6;
        sub_6AE340((__int64 *)v6, (__int64)a4);
        goto LABEL_30;
      case 167:
        a2 = (__int64)a4;
        v23 = v6;
        sub_6AD6A0((__int64 *)v6, (__int64)a4);
        goto LABEL_30;
      default:
        goto LABEL_48;
    }
  }
  if ( v18 == 34 )
  {
    a2 = (__int64)a4;
    v23 = v6;
    sub_6A59A0((_QWORD *)v6, (__int64)a4, (__int64)v19, v20);
    goto LABEL_30;
  }
  if ( (unsigned __int16)v18 <= 0x22u )
  {
    if ( (unsigned __int16)v18 > 0x20u )
    {
      a2 = (__int64)a4;
      v23 = v6;
      sub_6A49A0(v6, (unsigned __int64)a4, a3 & 0x100, v20, v21, v22);
      goto LABEL_30;
    }
    if ( (unsigned __int16)v18 > 0x1Eu )
    {
      if ( v156 )
      {
        v23 = 0;
        sub_69D0B0(0, a2, (__int64)a4, v156, v21, v22);
      }
      else
      {
        a2 = (__int64)a4;
        v23 = v6;
        sub_6A4340(v6, a4, (__int64)v19, 0, v21, v22);
      }
      goto LABEL_30;
    }
LABEL_48:
    sub_721090(v17);
  }
  if ( (unsigned __int16)(v18 - 35) > 3u )
    goto LABEL_48;
  a2 = (__int64)a4;
  v23 = v6;
  sub_6A5FF0(v6, (__int64)a4, (__int64)v19, v20);
LABEL_30:
  v26 = qword_4D03C50;
  *(_BYTE *)(qword_4D03C50 + 20LL) &= ~8u;
  if ( *(_BYTE *)(v6 + 56) )
  {
    v23 = (unsigned __int64)a4;
    sub_6E6260(a4);
    v26 = qword_4D03C50;
  }
  else if ( a5 == (__m128i *)v166 && (*((_BYTE *)a4 + 18) & 1) != 0 )
  {
    sub_6E6840(a4);
    *((_BYTE *)a4 + 18) &= ~1u;
    v23 = (unsigned __int64)v166;
    sub_6E6450(v166);
    *(_BYTE *)(v6 + 56) = 1;
    v26 = qword_4D03C50;
  }
  else if ( (*(_BYTE *)(v26 + 19) & 1) != 0 )
  {
    *(_BYTE *)(v6 + 56) = 1;
  }
  else if ( word_4D04898 )
  {
    v28 = *((unsigned __int8 *)a4 + 16);
    if ( (_BYTE)v28 != 2 )
    {
      v23 = a6;
      if ( a6 )
      {
        if ( *((_BYTE *)a4 + 17) != 3 && (_BYTE)v28 != 3 && *(_BYTE *)(v26 + 16) == 1 )
        {
          v29 = sub_724DC0(a6, a2, v28, v24, word_4D04898, v25);
          v30 = _mm_loadu_si128((const __m128i *)a4);
          v31 = _mm_loadu_si128((const __m128i *)a4 + 1);
          v167[0] = v29;
          v32 = _mm_loadu_si128((const __m128i *)a4 + 2);
          v33 = _mm_loadu_si128((const __m128i *)a4 + 3);
          v34 = _mm_loadu_si128((const __m128i *)a4 + 4);
          v168 = v30;
          v35 = _mm_loadu_si128((const __m128i *)a4 + 5);
          v36 = _mm_loadu_si128((const __m128i *)a4 + 6);
          v169 = v31;
          v37 = _mm_loadu_si128((const __m128i *)a4 + 7);
          v38 = *((_BYTE *)a4 + 16);
          v170 = v32;
          v39 = _mm_loadu_si128((const __m128i *)a4 + 8);
          v171 = v33;
          v172 = v34;
          v173 = v35;
          v174 = v36;
          v175 = v37;
          v176 = v39;
          if ( v38 == 2 )
          {
            v112 = _mm_loadu_si128((const __m128i *)a4 + 10);
            v113 = _mm_loadu_si128((const __m128i *)a4 + 11);
            v114 = _mm_loadu_si128((const __m128i *)a4 + 12);
            v115 = _mm_loadu_si128((const __m128i *)a4 + 13);
            v177 = _mm_loadu_si128((const __m128i *)a4 + 9);
            v116 = _mm_loadu_si128((const __m128i *)a4 + 14);
            v117 = _mm_loadu_si128((const __m128i *)a4 + 15);
            v178 = v112;
            v118 = _mm_loadu_si128((const __m128i *)a4 + 16);
            v119 = _mm_loadu_si128((const __m128i *)a4 + 17);
            v179 = v113;
            v120 = _mm_loadu_si128((const __m128i *)a4 + 18);
            v180 = v114;
            v121 = _mm_loadu_si128((const __m128i *)a4 + 19);
            v181 = v115;
            v122 = _mm_loadu_si128((const __m128i *)a4 + 20);
            v182 = v116;
            v123 = _mm_loadu_si128((const __m128i *)a4 + 21);
            v183 = v117;
            v184 = v118;
            v185 = v119;
            v186 = v120;
            v187 = v121;
            v188 = v122;
            v189 = v123;
          }
          else if ( v38 == 5 || v38 == 1 )
          {
            v177.m128i_i64[0] = a4[18];
          }
          v40 = (unsigned int)sub_8D3DE0(v142) == 0;
          v41 = 0;
          if ( v40 )
            v41 = v142;
          sub_697340(a4, v41, 0xC1u, 0, 0, 0, v167[0]);
          sub_6E6A50(v167[0], a4);
          sub_6E4BC0(a4, &v168);
          v23 = (unsigned __int64)v167;
          sub_724E30(v167);
          v26 = qword_4D03C50;
          a2 = v128;
        }
      }
    }
  }
  *(_QWORD *)(v26 + 128) = v149;
  if ( v151 )
  {
    sub_6E2B30(v23, a2);
  }
  else if ( (*(_BYTE *)(v6 + 41) & 8) != 0 )
  {
    *(_BYTE *)(v26 + 19) = (v152 << 7) | *(_BYTE *)(v26 + 19) & 0x7F;
  }
  if ( v163 )
  {
    sub_878D40();
    sub_6E1DF0(v160[0]);
    qword_4F06BC0 = v160[1];
    *(_QWORD *)&dword_4F061D8 = v162;
    sub_729730(v161);
  }
  *(_QWORD *)v6 = v154;
  *(_DWORD *)(v6 + 40) = v153;
  return v153;
}
