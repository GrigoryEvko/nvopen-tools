// Function: sub_8B6180
// Address: 0x8b6180
//
__int64 *__fastcall sub_8B6180(unsigned __int64 a1, __m128i *a2, int a3)
{
  __int64 v3; // r13
  __int64 v4; // rax
  __m128i *v5; // rbx
  __int64 v6; // rax
  __int64 v7; // rax
  __m128i *v8; // rax
  __int8 v9; // r12
  __m128i *v10; // r14
  unsigned int v11; // eax
  __int64 v12; // r9
  unsigned __int64 v13; // rdx
  char v14; // r15
  __int64 v15; // rdi
  __int64 v16; // rax
  __int64 *v17; // r15
  unsigned int *v18; // rsi
  __int64 *v19; // rax
  __int64 v20; // rdx
  __int64 v21; // r8
  __int64 v22; // r9
  __m128i *v23; // rcx
  __int64 v24; // rdi
  __int64 v25; // r12
  _BYTE *v26; // rax
  __int64 i; // rax
  __int64 v28; // rdx
  __int64 *v29; // rcx
  __int64 *j; // rax
  unsigned int v31; // edx
  __int64 *v32; // rax
  char v33; // al
  __int8 v34; // dl
  __int8 v35; // dl
  __int8 v36; // al
  __int8 v37; // al
  __int8 v38; // al
  __int8 v39; // al
  __int8 v40; // al
  __int8 v41; // al
  __int8 v42; // cl
  __int8 v43; // al
  __int8 v44; // al
  __int8 v45; // al
  __int8 v46; // al
  __int8 v47; // al
  __int64 v48; // rax
  __int32 v49; // edx
  __int64 v50; // rdx
  __int64 v51; // rcx
  __int64 v52; // r8
  __int64 v53; // r9
  __int8 v54; // al
  __int8 v55; // al
  __int8 v56; // al
  __int8 v57; // al
  __int64 v58; // r12
  _QWORD *k; // r15
  __int64 v60; // rdx
  __int64 v61; // r12
  __int64 **v62; // rax
  __int64 **v63; // r15
  __int64 v64; // r8
  __int64 v65; // r9
  __int8 v66; // al
  __int64 v67; // rdx
  __int64 v68; // rcx
  __int64 v69; // r8
  __int64 *v70; // r9
  __int64 v71; // rax
  bool v72; // zf
  int v74; // edi
  __int64 v75; // rax
  __int64 v76; // rax
  char v77; // dl
  __int64 v78; // rdx
  __int64 v79; // r12
  _QWORD *v80; // rax
  __int64 v81; // rdx
  __int64 v82; // r8
  __int64 v83; // r9
  __int64 v84; // rcx
  int v85; // eax
  __int8 v86; // al
  __int64 v87; // rdx
  __int64 v88; // rcx
  __int64 v89; // r8
  __int64 v90; // r9
  __int8 v91; // al
  __int64 **v92; // rax
  __int8 v93; // al
  __int64 v94; // rdx
  __int64 v95; // rcx
  __int64 v96; // r8
  __int64 v97; // r9
  __int64 v98; // rdx
  __int64 v99; // rcx
  __int64 v100; // r8
  __int64 v101; // r9
  __int64 v102; // [rsp-10h] [rbp-3B0h]
  __int64 v103; // [rsp-8h] [rbp-3A8h]
  _QWORD *v104; // [rsp+0h] [rbp-3A0h]
  int v105; // [rsp+14h] [rbp-38Ch]
  unsigned __int16 v106; // [rsp+18h] [rbp-388h]
  __int16 v107; // [rsp+1Ah] [rbp-386h]
  int v108; // [rsp+1Ch] [rbp-384h]
  __int64 v109; // [rsp+20h] [rbp-380h]
  unsigned __int16 v110; // [rsp+28h] [rbp-378h]
  __int64 v111; // [rsp+28h] [rbp-378h]
  int v112; // [rsp+30h] [rbp-370h]
  __int64 v114; // [rsp+38h] [rbp-368h]
  __int64 v115; // [rsp+40h] [rbp-360h]
  __int64 v116; // [rsp+50h] [rbp-350h]
  __int64 v118; // [rsp+60h] [rbp-340h]
  unsigned int v119; // [rsp+68h] [rbp-338h]
  __int64 *v120; // [rsp+68h] [rbp-338h]
  int v121; // [rsp+74h] [rbp-32Ch] BYREF
  __int64 v122; // [rsp+78h] [rbp-328h] BYREF
  __m128i v123; // [rsp+80h] [rbp-320h] BYREF
  _QWORD v124[12]; // [rsp+C0h] [rbp-2E0h] BYREF
  __int64 v125; // [rsp+120h] [rbp-280h] BYREF
  _QWORD *v126; // [rsp+128h] [rbp-278h] BYREF
  char v127; // [rsp+161h] [rbp-23Fh]
  __m128i v128[33]; // [rsp+190h] [rbp-210h] BYREF

  v118 = a1;
  sub_88E6E0(a2->m128i_i64, 0);
  v3 = *(_QWORD *)(a1 + 88);
  v4 = sub_880C60();
  *(_QWORD *)(v4 + 32) = a1;
  v5 = *(__m128i **)(v3 + 176);
  v114 = v4;
  v112 = sub_8D0B70(a1);
  sub_7296C0(&v121);
  v116 = 0;
  ++qword_4D03B78;
  v6 = qword_4F601B8;
  qword_4F601B8 = 0;
  v109 = v6;
  v7 = qword_4F601B0;
  qword_4F601B0 = 0;
  v104 = (_QWORD *)v7;
  if ( (*(_BYTE *)(a1 + 81) & 0x10) != 0 )
    v116 = *(_QWORD *)(a1 + 64);
  v8 = sub_725FD0();
  ++*(_DWORD *)(v3 + 388);
  v9 = v5[10].m128i_i8[14];
  v10 = v8;
  v11 = 4 * (v9 == 7) + 66560;
  if ( a3 )
  {
    v10[12].m128i_i8[3] |= 0x20u;
    if ( dword_4F04C44 != -1 || (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) & 2) != 0 )
      v11 |= 2u;
  }
  sub_864700(*(_QWORD *)(v3 + 328), 0, 0, 0, a1, (__int64)a2, 1, v11);
  sub_854C10(*(const __m128i **)(v3 + 56));
  v119 = dword_4F063F8;
  v110 = word_4F063FC[0];
  v108 = dword_4F07508[0];
  v107 = dword_4F07508[1];
  v105 = dword_4F061D8;
  v106 = word_4F061DC[0];
  if ( dword_4F077C4 == 2 )
    *(_BYTE *)(qword_4F04C68[0] + 776LL * (int)dword_4F04C40 + 7) |= 8u;
  sub_7BC160(v3 + 296);
  v13 = *(unsigned int *)(v3 + 388);
  v14 = *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v3 + 328) + 16LL) + 28LL);
  v122 = *(_QWORD *)&dword_4F063F8;
  if ( v13 >= unk_4D042F0 )
  {
    sub_6854E0(0x1C8u, a1);
    v80 = sub_88DE40((__int64)v5, v116);
    v84 = v114;
    v115 = (__int64)v80;
    v72 = word_4F06418[0] == 9;
    *(_QWORD *)(v114 + 112) = v80;
    if ( !v72 )
    {
      do
        sub_7B8B50((unsigned __int64)v5, (unsigned int *)v116, v81, v84, v82, v83);
      while ( word_4F06418[0] != 9 );
    }
    sub_7B8B50((unsigned __int64)v5, (unsigned int *)v116, v81, v84, v82, v83);
    v17 = (__int64 *)(a1 + 48);
  }
  else
  {
    if ( v9 != 7 && !v116 )
    {
      sub_87E3B0((__int64)&v125);
      memset(v124, 0, 0x58u);
      memset(v128, 0, 0x1D8u);
      v128[9].m128i_i64[1] = (__int64)v128;
      v128[1].m128i_i64[1] = *(_QWORD *)&dword_4F063F8;
      if ( dword_4F077BC && qword_4F077A8 <= 0x9F5Fu )
        v128[11].m128i_i8[2] |= 1u;
      v128[7].m128i_i8[11] = (8 * (word_4D04430 & 1)) | v128[7].m128i_i8[11] & 0xF7;
      sub_898140(
        (__int64)v128,
        0,
        v14 == 6,
        a1,
        0,
        0,
        0,
        &v123,
        (unsigned __int64)&v125,
        (unsigned int *)v5,
        v114,
        v124);
      v15 = v128[18].m128i_i64[0];
      *(_QWORD *)(v114 + 64) = v125;
      v115 = v15;
      *(_QWORD *)(v114 + 104) = v126;
      *(_QWORD *)(v114 + 112) = sub_624310(v15, (__int64)&v125);
      v126 = 0;
      if ( dword_4F04C64 == -1
        || (v16 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v16 + 7) & 1) == 0)
        || dword_4F04C44 == -1 && (*(_BYTE *)(v16 + 6) & 2) == 0 )
      {
        if ( (v127 & 8) == 0 )
          sub_87E280(&v126);
      }
      v122 = v123.m128i_i64[1];
      v17 = (__int64 *)(v118 + 48);
      goto LABEL_29;
    }
    *(_QWORD *)(v114 + 112) = v5[16].m128i_i64[1];
    if ( v9 == 7 )
      goto LABEL_24;
    if ( (v5[12].m128i_i8[1] & 0x10) != 0 )
    {
      if ( (v5[12].m128i_i8[2] & 0x40) != 0 )
      {
LABEL_24:
        v18 = (unsigned int *)v3;
        v17 = (__int64 *)(a1 + 48);
        v19 = sub_894B30(a1, v3, a2, 0x20000, 0, v12);
        v23 = v128;
        v115 = (__int64)v19;
        if ( !v19 )
        {
          LODWORD(v125) = 0;
          sub_892150(v128);
          a1 = v5[9].m128i_u64[1];
          v18 = (unsigned int *)a2;
          v92 = sub_8A2270(a1, a2, **(_QWORD **)(v3 + 328), v17, 0x20000, (int *)&v125, v128);
          v21 = v102;
          v22 = v103;
          v115 = (__int64)v92;
          v93 = v5[10].m128i_i8[14];
          if ( v93 == 3 )
          {
            if ( (*(_BYTE *)(*(_QWORD *)(v116 + 168) + 109LL) & 0x20) != 0 )
            {
              v18 = (unsigned int *)a2;
              a1 = *(_QWORD *)(v118 + 64);
              sub_8B1B50((_QWORD *)a1, a2, v115);
            }
          }
          else if ( v93 == 6 )
          {
            v18 = (unsigned int *)a2;
            a1 = *(_QWORD *)(v118 + 64);
            sub_8C0170(a1, a2, v115);
          }
        }
        if ( *(_QWORD *)(v3 + 304) )
        {
          while ( word_4F06418[0] != 9 )
            sub_7B8B50(a1, v18, v20, (__int64)v23, v21, v22);
          sub_7B8B50(a1, v18, v20, (__int64)v23, v21, v22);
        }
        goto LABEL_29;
      }
      v86 = v5[10].m128i_i8[14];
      if ( v86 == 3 )
      {
        if ( (*(_BYTE *)(*(_QWORD *)(v116 + 168) + 109LL) & 0x20) != 0 )
          goto LABEL_24;
      }
      else if ( v86 == 6 )
      {
        goto LABEL_24;
      }
    }
    ++*(_BYTE *)(qword_4F061C8 + 17LL);
    v128[0].m128i_i64[0] = sub_609B50(v116, (_QWORD *)v114);
    --*(_BYTE *)(qword_4F061C8 + 17LL);
    if ( word_4F06418[0] == 56 && (unsigned __int16)sub_7BE840(0, 0) == 4 )
    {
      sub_6851C0(0x723u, &dword_4F063F8);
      sub_7B8B50(0x723u, &dword_4F063F8, v94, v95, v96, v97);
      sub_7B8B50(0x723u, &dword_4F063F8, v98, v99, v100, v101);
    }
    sub_88DF70(v128[0].m128i_i64, (__int64)v5, 0, v116, v114);
    while ( word_4F06418[0] != 9 )
      sub_7B8B50((unsigned __int64)v128, (unsigned int *)v5, v87, v88, v89, v90);
    sub_7B8B50((unsigned __int64)v128, (unsigned int *)v5, v87, v88, v89, v90);
    v115 = v128[0].m128i_i64[0];
    v17 = (__int64 *)(a1 + 48);
  }
LABEL_29:
  dword_4F07508[0] = v108;
  LOWORD(dword_4F07508[1]) = v107;
  dword_4F063F8 = v119;
  word_4F063FC[0] = v110;
  dword_4F061D8 = v105;
  word_4F061DC[0] = v106;
  if ( (unsigned int)sub_8D2310(v115) )
  {
    v24 = v115;
    if ( *(_BYTE *)(v115 + 140) == 12 )
    {
      do
        v24 = *(_QWORD *)(v24 + 160);
      while ( *(_BYTE *)(v24 + 140) == 12 );
    }
    else
    {
      v24 = v115;
    }
    v25 = *(_QWORD *)(v24 + 168);
    v111 = *(_QWORD *)(v24 + 160);
    *(_BYTE *)(v25 + 20) = (2 * *(_BYTE *)(v3 + 160)) & 0x10 | *(_BYTE *)(v25 + 20) & 0xEF;
    v26 = *(_BYTE **)(v25 + 56);
    if ( v26 && (*v26 & 0x22) == 0 && (unsigned int)sub_8D76D0(v24) )
      v10[12].m128i_i8[3] |= 0x10u;
    if ( (v5[12].m128i_i8[2] & 0x40) != 0 )
      *(_QWORD *)(v25 + 8) = v10;
    if ( (v5[12].m128i_i8[14] & 2) != 0 )
      *(_BYTE *)(v25 + 17) |= 4u;
  }
  else
  {
    v111 = sub_72C930();
  }
  if ( (*(_BYTE *)(v118 + 81) & 2) != 0 )
  {
    for ( i = v5[9].m128i_i64[1]; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
      ;
    v28 = v115;
    if ( *(_BYTE *)(v115 + 140) == 12 )
    {
      do
        v28 = *(_QWORD *)(v28 + 160);
      while ( *(_BYTE *)(v28 + 140) == 12 );
    }
    else
    {
      v28 = v115;
    }
    v29 = **(__int64 ***)(v28 + 168);
    for ( j = **(__int64 ***)(i + 168); v29; v29 = (__int64 *)*v29 )
    {
      v31 = *((_DWORD *)v29 + 9);
      while ( *((_DWORD *)j + 9) < v31 )
      {
        j = (__int64 *)*j;
        if ( !j )
          goto LABEL_50;
      }
      if ( *((_DWORD *)j + 9) == v31 )
        *((_DWORD *)v29 + 8) = v29[4] & 0xFFFC07FF
                             | (((*((_DWORD *)v29 + 8) >> 11) & 0x7F | (*((_DWORD *)j + 8) >> 11) & 3) << 11);
    }
  }
LABEL_50:
  v32 = sub_87F4B0(v118, v17, v111);
  v32[11] = (__int64)v10;
  v120 = v32;
  v10[9].m128i_i64[1] = v115;
  v33 = v5[10].m128i_i8[12];
  if ( !v33 )
    v33 = 1;
  v10[10].m128i_i8[12] = v33;
  sub_725ED0((__int64)v10, v5[10].m128i_i8[14]);
  v34 = v10[12].m128i_i8[5];
  v10[11] = _mm_loadu_si128(v5 + 11);
  v35 = v5[12].m128i_i8[5] & 2 | v34 & 0xFD;
  v36 = v10[12].m128i_i8[6];
  v10[12].m128i_i8[5] = v35;
  v37 = v5[12].m128i_i8[6] & 8 | v36 & 0xF7;
  v10[12].m128i_i8[6] = v37;
  v38 = v5[12].m128i_i8[6] & 0x10 | v37 & 0xEF;
  v10[12].m128i_i8[6] = v38;
  v39 = v5[12].m128i_i8[6] & 0x20 | v38 & 0xDF;
  v10[12].m128i_i8[6] = v39;
  v40 = v5[12].m128i_i8[6] & 0x40 | v39 & 0xBF;
  v10[12].m128i_i8[6] = v40;
  v10[12].m128i_i8[6] = v5[12].m128i_i8[6] & 0x80 | v40 & 0x7F;
  v41 = v5[12].m128i_i8[7] & 4 | v10[12].m128i_i8[7] & 0xFB;
  v10[12].m128i_i8[7] = v41;
  v42 = v5[12].m128i_i8[5];
  if ( (v42 & 0x10) != 0 )
  {
    v10[12].m128i_i8[5] = v35 | 0x10;
    v42 = v5[12].m128i_i8[5];
  }
  if ( (v42 & 8) != 0 )
    v10[12].m128i_i8[5] |= 8u;
  if ( (v5[12].m128i_i8[7] & 0x10) != 0 )
  {
    v10[12].m128i_i8[7] = v41 | 0x10;
    v10[12].m128i_i8[4] = v5[12].m128i_i8[4] & 0x40 | v10[12].m128i_i8[4] & 0xBF;
    v10[12].m128i_i8[10] = v5[12].m128i_i8[10] & 1 | v10[12].m128i_i8[10] & 0xFE;
  }
  v43 = v5[12].m128i_i8[14] & 2 | v10[12].m128i_i8[14] & 0xFD;
  v10[12].m128i_i8[14] = v43;
  v44 = v5[12].m128i_i8[14] & 8 | v43 & 0xF7;
  v10[12].m128i_i8[14] = v44;
  v10[12].m128i_i8[14] = v5[12].m128i_i8[14] & 0x10 | v44 & 0xEF;
  v10[12].m128i_i8[15] = v5[12].m128i_i8[15] & 0x10 | v10[12].m128i_i8[15] & 0xEF;
  v45 = v5[12].m128i_i8[1];
  if ( (v45 & 1) != 0 )
  {
    v10[12].m128i_i8[1] |= 3u;
    v45 = v5[12].m128i_i8[1];
  }
  else if ( (v5[12].m128i_i8[14] & 2) != 0 && (v45 & 2) != 0 )
  {
    v10[12].m128i_i8[1] |= 2u;
    v45 = v5[12].m128i_i8[1];
  }
  if ( (v45 & 4) != 0 )
  {
    v10[12].m128i_i8[1] |= 6u;
    v45 = v5[12].m128i_i8[1];
  }
  v10[12].m128i_i8[1] = v10[12].m128i_i8[1] & 0xEF | v45 & 0x10;
  v46 = v5[12].m128i_i8[2] & 0x10 | v10[12].m128i_i8[2] & 0xEF;
  v10[12].m128i_i8[2] = v46;
  v47 = v5[12].m128i_i8[2] & 0x40 | v46 & 0xBF;
  v10[12].m128i_i8[2] = v47;
  if ( (v47 & 0x40) != 0 )
  {
    v48 = 0;
    if ( (v5[12].m128i_i8[2] & 0x40) != 0 )
      v48 = v5[14].m128i_i64[1];
    v10[14].m128i_i64[1] = v48;
  }
  v10[22].m128i_i64[1] = v5[22].m128i_i64[1];
  sub_736C90((__int64)v10, (unsigned __int8)v5[12].m128i_i8[0] >> 7);
  v49 = v10[12].m128i_i32[0];
  v10[12].m128i_i8[13] = v5[12].m128i_i8[13] & 1 | v10[12].m128i_i8[13] & 0xFE;
  v10[12].m128i_i32[0] = v49 & 0xFEFF7FFF | ((unsigned __int8)v5[12].m128i_i8[1] >> 7 << 15) | 0x1000000;
  sub_877D80((__int64)v10, v120);
  if ( (v5[5].m128i_i8[9] & 1) != 0 )
    v10[5].m128i_i8[9] |= 1u;
  sub_877F10((__int64)v10, (__int64)v120, v50, v51, v52, v53);
  v54 = v5[5].m128i_i8[8] & 0x70 | v10[5].m128i_i8[8] & 0x8F;
  v10[5].m128i_i8[8] = v54;
  if ( dword_4D047F8 && (v54 & 0x70) != 0x10 && (unsigned int)sub_88DB10(a2->m128i_i64) )
  {
    v91 = v10[5].m128i_i8[8];
    v10[12].m128i_i8[8] &= 0xF8u;
    v10[10].m128i_i8[12] = 2;
    v10[5].m128i_i8[8] = v91 & 0x8F | 0x10;
  }
  v10[5].m128i_i8[8] = v5[5].m128i_i8[8] & 3 | v10[5].m128i_i8[8] & 0xFC;
  v10[15].m128i_i64[0] = (__int64)a2;
  v55 = v10[12].m128i_i8[6];
  if ( (v55 & 0x20) != 0 )
  {
    unk_4F60218 = v10;
    sub_88E010((__int64)v10, 1, a2->m128i_i64);
    unk_4F60218 = 0;
    v55 = v10[12].m128i_i8[6];
  }
  if ( (v55 & 0x10) != 0 )
  {
    v76 = v10[15].m128i_i64[0];
    for ( v128[0].m128i_i64[0] = v76; v76; v128[0].m128i_i64[0] = v76 )
    {
      v77 = *(_BYTE *)(v76 + 8);
      if ( v77 == 3 )
      {
        sub_72F220((__int64 **)v128);
        v76 = v128[0].m128i_i64[0];
        if ( !v128[0].m128i_i64[0] )
          break;
        v77 = *(_BYTE *)(v128[0].m128i_i64[0] + 8);
      }
      if ( v77 == 1 )
      {
        v78 = *(_QWORD *)(v76 + 32);
        if ( v78 )
        {
          if ( (*(_BYTE *)(v76 + 24) & 1) == 0 )
          {
            v79 = *(_QWORD *)(v78 + 128);
            if ( v79 )
            {
              if ( (unsigned int)sub_8D2FB0(*(_QWORD *)(v78 + 128)) || (unsigned int)sub_8D2E30(v79) )
                v79 = sub_8D46C0(v79);
              if ( (*(_BYTE *)(v79 - 8) & 0x60) != 0 )
                sub_684AA0(7u, 0xE34u, dword_4F07508);
            }
          }
        }
      }
      v76 = *(_QWORD *)v128[0].m128i_i64[0];
    }
  }
  sub_826060((__int64)v10, v17);
  v10[15].m128i_i64[1] = *(_QWORD *)(v3 + 104);
  v56 = v10[10].m128i_i8[14];
  if ( v56 == 2 || v56 == 5 && ((v10[11].m128i_i8[0] - 2) & 0xFD) == 0 )
  {
    sub_5F93D0((__int64)v10, &v10[9].m128i_i64[1]);
    if ( (v5[12].m128i_i8[1] & 0x10) != 0 )
      goto LABEL_78;
  }
  else if ( (v5[12].m128i_i8[1] & 0x10) != 0 )
  {
    goto LABEL_78;
  }
  sub_8756F0(0x8000, (__int64)v120, v120 + 6, 0);
LABEL_78:
  if ( v5[6].m128i_i64[1] )
    sub_892760((__int64)v10, v118, v3, 0);
  sub_880310((__int64)v128);
  v128[0].m128i_i32[0] = v5[13].m128i_i32[1] & ~v10[13].m128i_i32[1];
  sub_648B00((__int64)v10, v128, (__int64)&v122);
  if ( dword_4F077BC )
  {
    if ( v10[10].m128i_i8[12] != 2 )
    {
      v57 = v5[12].m128i_i8[8] & 7 | v10[12].m128i_i8[8] & 0xF8;
      v10[12].m128i_i8[8] = v57;
      if ( (v10[5].m128i_i8[9] & 4) != 0 && (v57 & 7) == 0 )
        v10[12].m128i_i8[8] = *(_BYTE *)(*(_QWORD *)(v116 + 168) + 109LL) & 7 | v57 & 0xF8;
    }
  }
  if ( (v10[12].m128i_i8[2] & 0x40) != 0
    && (v10[12].m128i_i8[14] & 0x10) == 0
    && (unsigned int)sub_600C10(*(_QWORD *)(v10[2].m128i_i64[1] + 32)) )
  {
    v10[12].m128i_i8[14] |= 0x10u;
    v10[12].m128i_i8[1] |= 0x20u;
  }
  v58 = v3;
  sub_7362F0((__int64)v10, -1);
  while ( 1 )
  {
    for ( k = *(_QWORD **)(v58 + 72); k; k = (_QWORD *)*k )
      sub_5EDDD0((__int64)v10, k[1]);
    v60 = *(_QWORD *)(v58 + 88);
    if ( !v60 )
      break;
    switch ( *(_BYTE *)(v60 + 80) )
    {
      case 4:
      case 5:
        v58 = *(_QWORD *)(*(_QWORD *)(v60 + 96) + 80LL);
        break;
      case 6:
        v58 = *(_QWORD *)(*(_QWORD *)(v60 + 96) + 32LL);
        break;
      case 9:
      case 0xA:
        v58 = *(_QWORD *)(*(_QWORD *)(v60 + 96) + 56LL);
        break;
      case 0x13:
      case 0x14:
      case 0x15:
      case 0x16:
        v58 = *(_QWORD *)(v60 + 88);
        break;
      default:
        v58 = 0;
        break;
    }
  }
  sub_884800((__int64)v10);
  if ( dword_4F077C4 == 2 )
  {
    v74 = dword_4F04C40;
    v75 = 776LL * (int)dword_4F04C40;
    *(_BYTE *)(qword_4F04C68[0] + v75 + 7) &= ~8u;
    if ( *(_QWORD *)(qword_4F04C68[0] + v75 + 456) )
      sub_8845B0(v74);
  }
  *(_QWORD *)(v114 + 24) = v120;
  v120[12] = v114;
  *(_QWORD *)v114 = *(_QWORD *)(v3 + 168);
  *(_QWORD *)(v3 + 168) = v114;
  v61 = v10[9].m128i_i64[1];
  v62 = sub_8B1C20(v118, (__int64)a2, 0, 0, 0x20000u);
  v63 = v62;
  if ( !v62 || !(unsigned int)sub_8DD8B0(v62, v61, 1050624, 0) )
  {
    if ( !(unsigned int)sub_8D97B0(v61) )
    {
      v85 = sub_8D97B0(v5[9].m128i_i64[1]);
      if ( v63 )
      {
        if ( !v85 && !(unsigned int)sub_8DED30(v63, v61, 2049) )
          sub_6861A0(0x367u, &dword_4F063F8, v61, v5[9].m128i_i64[1]);
      }
    }
    v10[9].m128i_i64[1] = (__int64)sub_88DE40((__int64)v5, v116);
    *(_BYTE *)(v114 + 80) |= 2u;
  }
  sub_890140(v118, (_QWORD *)v3, (__int64)v120, (__int64)a2, v64, v65);
  sub_854980((__int64)v120, 0);
  if ( *(_QWORD *)(v3 + 288) )
    sub_88FC40(v5[9].m128i_i64[1], (__int64)v10, (__int64 *)v3);
  sub_878710((__int64)v120, v128);
  v66 = v10[10].m128i_i8[14];
  if ( v66 == 5 )
  {
    v128[1].m128i_i8[0] |= 8u;
    v128[3].m128i_i8[8] = v10[11].m128i_i8[0];
  }
  else if ( v66 == 3 )
  {
    v128[1].m128i_i8[0] |= 0x10u;
    v128[3].m128i_i64[1] = v111;
  }
  sub_646070(v115, v116, v128);
  sub_863FE0(v115, v116, v67, v68, v69, v70);
  --*(_DWORD *)(v3 + 388);
  sub_729730(v121);
  if ( !a3 )
    sub_8CCE20(v120, v3);
  if ( !v116 || (*(_BYTE *)(v116 + 177) & 0x20) == 0 )
    sub_8AD0D0((__int64)v120, 0, 0);
  if ( v112 )
    sub_8D0B10();
  --qword_4D03B78;
  if ( (v10[12].m128i_i8[1] & 2) != 0 && (*(_BYTE *)(*v120 + 73) & 2) != 0 )
    sub_64A920(v10, *v120);
  sub_5F94C0(1);
  if ( v109 )
  {
    v71 = qword_4F601B8;
    qword_4F601B8 = v109;
    v72 = qword_4F601B0 == 0;
    *v104 = v71;
    if ( v72 )
      qword_4F601B0 = (__int64)v104;
  }
  if ( v10[10].m128i_i8[14] != 7 && (v10[5].m128i_i8[9] & 4) == 0 )
    sub_894C00((__int64)v120);
  return v120;
}
