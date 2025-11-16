// Function: sub_71F5B0
// Address: 0x71f5b0
//
__int64 __fastcall sub_71F5B0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // r12
  __int64 v6; // r13
  __int64 i; // rax
  char v8; // al
  __int64 j; // rax
  __int64 v10; // r13
  __int64 k; // rax
  __int64 v12; // rbx
  __int64 v13; // r9
  unsigned __int64 v14; // rax
  FILE *v15; // rsi
  __int64 v16; // r8
  __int64 v17; // rax
  __int64 v18; // r14
  __int64 v19; // rax
  __m128i *v20; // rbx
  __int8 v21; // al
  unsigned __int64 v22; // r13
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // r9
  _BOOL8 v27; // r12
  __int64 v28; // r14
  __int64 v29; // rsi
  unsigned __int64 v30; // rsi
  __int64 v31; // rcx
  __int64 v32; // r8
  __int64 result; // rax
  _QWORD *jj; // r13
  char v35; // dl
  __int64 kk; // rax
  __int64 v37; // rsi
  __int64 v38; // rdi
  __int64 v39; // rdx
  __int64 v40; // rcx
  __int64 v41; // rax
  _QWORD *v42; // r13
  __int64 *v43; // rbx
  __int64 v44; // r14
  __int64 v45; // rdi
  __int64 v46; // rcx
  __int64 v47; // rax
  _QWORD *v48; // rdx
  _QWORD *v49; // rax
  __int64 v50; // rax
  __int64 v51; // rdx
  __int64 v52; // r8
  __int64 v53; // r12
  __int64 v54; // rax
  unsigned __int8 v55; // di
  __int64 v56; // rdx
  __int64 v57; // rsi
  __int64 v58; // rax
  __int64 v59; // r8
  __int64 v60; // r9
  char v61; // al
  bool v62; // sf
  _BYTE *v63; // rax
  __int64 v64; // rdx
  __int64 v65; // rax
  __int64 v66; // rax
  __int64 v67; // r9
  __int64 n; // rax
  __int64 v69; // rcx
  char v70; // al
  unsigned int v71; // edi
  FILE *v72; // rsi
  char v73; // al
  char v74; // al
  __int64 v75; // rsi
  __int64 v76; // rdx
  __int64 v77; // rcx
  __int64 v78; // r8
  __int64 v79; // r9
  __int64 v80; // rax
  __int64 v81; // rdi
  __int64 v82; // rdx
  char v83; // al
  __int64 ii; // rax
  int v85; // eax
  __int64 v86; // rax
  __int64 *v87; // rdx
  char v88; // cl
  __int64 m; // rax
  __int64 *v90; // [rsp+0h] [rbp-A0h]
  __int64 v91; // [rsp+8h] [rbp-98h]
  __int64 v92; // [rsp+8h] [rbp-98h]
  __int64 v93; // [rsp+8h] [rbp-98h]
  __int64 v94; // [rsp+8h] [rbp-98h]
  __int64 v95; // [rsp+8h] [rbp-98h]
  __int64 v96; // [rsp+8h] [rbp-98h]
  __int64 v97; // [rsp+8h] [rbp-98h]
  __int64 v98; // [rsp+8h] [rbp-98h]
  __int64 v99; // [rsp+8h] [rbp-98h]
  __int64 v100; // [rsp+10h] [rbp-90h]
  __int64 v101; // [rsp+10h] [rbp-90h]
  FILE *v102; // [rsp+10h] [rbp-90h]
  const __m128i *v103; // [rsp+10h] [rbp-90h]
  __int16 v104; // [rsp+18h] [rbp-88h]
  int v105; // [rsp+20h] [rbp-80h]
  __int64 v106; // [rsp+20h] [rbp-80h]
  __m128i v108; // [rsp+30h] [rbp-70h]
  int v109; // [rsp+30h] [rbp-70h]
  __m128i v110; // [rsp+30h] [rbp-70h]
  __int64 v111; // [rsp+30h] [rbp-70h]
  __int64 v112; // [rsp+40h] [rbp-60h]
  __int64 v113; // [rsp+40h] [rbp-60h]
  __int64 v114; // [rsp+40h] [rbp-60h]
  __int16 v115; // [rsp+40h] [rbp-60h]
  int v117; // [rsp+54h] [rbp-4Ch] BYREF
  __int64 v118; // [rsp+58h] [rbp-48h] BYREF
  __int64 v119; // [rsp+60h] [rbp-40h] BYREF
  __int64 v120[7]; // [rsp+68h] [rbp-38h] BYREF

  v5 = a2;
  v6 = *(_QWORD *)(a1 + 24);
  if ( *(char *)(a3 + 64) < 0 )
  {
    sub_6851C0(0x5Du, dword_4F07508);
    i = sub_73EDA0(*(_QWORD *)(a2 + 288), 1);
    for ( *(_QWORD *)(a2 + 288) = i; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
      ;
  }
  else
  {
    for ( i = *(_QWORD *)(a2 + 288); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
      ;
  }
  v112 = *(_QWORD *)(i + 168);
  if ( v6 && (*(_BYTE *)(v6 + 81) & 0x10) != 0 )
  {
    if ( (*(_BYTE *)(a3 + 64) & 8) != 0 && dword_4D048B8 && dword_4D048B0 && !*(_QWORD *)(v112 + 56) )
    {
      v8 = *(_BYTE *)(a1 + 16);
      if ( (v8 & 0x20) != 0 )
      {
        if ( *(_BYTE *)(v6 + 80) == 10 )
        {
          for ( j = *(_QWORD *)(*(_QWORD *)(v6 + 88) + 152LL); *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
            ;
          *(_QWORD *)(v112 + 56) = *(_QWORD *)(*(_QWORD *)(j + 168) + 56LL);
        }
      }
      else if ( (v8 & 8) != 0 && ((*(_BYTE *)(a1 + 56) - 2) & 0xFD) == 0 )
      {
        sub_5F1D90(v112);
      }
    }
    v10 = *(_QWORD *)(a2 + 288);
    for ( k = v10; *(_BYTE *)(k + 140) == 12; k = *(_QWORD *)(k + 160) )
      ;
    v12 = *(_QWORD *)(k + 168);
    v119 = k;
    v13 = *(_QWORD *)(a1 + 24);
    v113 = *(_QWORD *)(v13 + 64);
    v105 = dword_4F077C8;
    v104 = unk_4F077CC;
    v14 = *(unsigned __int8 *)(v13 + 80);
    if ( (*(_BYTE *)(v13 + 81) & 0x10) == 0 )
      goto LABEL_18;
    v15 = (FILE *)(a1 + 8);
    if ( (unsigned __int8)v14 > 0x14u )
      goto LABEL_114;
    v56 = 1180672;
    if ( !_bittest64(&v56, v14) )
    {
LABEL_18:
      v15 = (FILE *)(a1 + 8);
      if ( (_BYTE)v14 == 16 )
      {
        sub_6851C0(0x12Au, v15);
LABEL_20:
        v16 = *(_QWORD *)a1;
        *(_QWORD *)(v12 + 40) = v113;
        v100 = v16;
        *(_BYTE *)(v12 + 21) |= 1u;
        *(__m128i *)a1 = _mm_loadu_si128(xmmword_4F06660);
        *(__m128i *)(a1 + 16) = _mm_loadu_si128(&xmmword_4F06660[1]);
        *(__m128i *)(a1 + 32) = _mm_loadu_si128(&xmmword_4F06660[2]);
        v17 = *(_QWORD *)dword_4F07508;
        v108 = _mm_loadu_si128(&xmmword_4F06660[3]);
        *(_BYTE *)(a1 + 17) |= 0x20u;
        *(_QWORD *)(a1 + 8) = v17;
        *(__m128i *)(a1 + 48) = v108;
        v18 = sub_647630(0xBu, a1, 0, 1);
        *(_QWORD *)v18 = v100;
        v19 = sub_646F50(v10, 0, 0xFFFFFFFF);
        *(_QWORD *)(v18 + 88) = v19;
        v20 = (__m128i *)v19;
        sub_877D80(v19, v18);
        sub_877E20(v18, v20, v113);
        goto LABEL_21;
      }
LABEL_114:
      sub_6854C0(0x93u, v15, v13);
      goto LABEL_20;
    }
    v101 = *(_QWORD *)(a1 + 24);
    if ( !(dword_4F077BC | (unsigned int)sub_85ED80(v13, qword_4F04C68[0] + 776LL * dword_4F04C64)) )
    {
      sub_6854E0(0x227u, v101);
      goto LABEL_20;
    }
    v57 = v5;
    v91 = v101;
    v58 = sub_5EDAE0(v101, v5, 0, v120);
    v60 = v101;
    v18 = v58;
    v102 = (FILE *)(a1 + 8);
    if ( !v58 )
    {
      if ( !qword_4D0495C )
        goto LABEL_165;
      if ( !*(_QWORD *)(v12 + 40) )
        goto LABEL_165;
      *(_BYTE *)(v12 + 21) &= ~1u;
      *(_QWORD *)(v12 + 40) = 0;
      v80 = sub_5EDAE0(*(_QWORD *)(a1 + 24), v5, 0, v120);
      *(_BYTE *)(v12 + 18) &= 0x80u;
      v60 = v91;
      v18 = v80;
      if ( !v80 )
        goto LABEL_165;
      v57 = a1 + 8;
      sub_685490(0x93u, v102, *(_QWORD *)(a1 + 24));
      v60 = v91;
    }
    v61 = *(_BYTE *)(v18 + 80);
    if ( v61 != 16 )
      goto LABEL_98;
    v57 = a1 + 8;
    v93 = v60;
    sub_6851C0(0x12Au, v102);
    v61 = *(_BYTE *)(v18 + 80);
    v60 = v93;
    if ( v61 == 16 )
    {
      v18 = **(_QWORD **)(v18 + 88);
      v61 = *(_BYTE *)(v18 + 80);
    }
    if ( v61 != 24 )
    {
LABEL_98:
      if ( v61 == 20 )
      {
        sub_6854C0(0x31Bu, v102, v18);
        goto LABEL_20;
      }
      if ( (*(_BYTE *)(*(_QWORD *)(v18 + 88) + 193LL) & 0x10) != 0 )
      {
        v57 = a1 + 8;
        sub_6851C0(0x14Du, v102);
        if ( (*(_BYTE *)(v18 + 81) & 2) == 0 )
        {
          v57 = 0;
          *(_BYTE *)(*(_QWORD *)(v18 + 88) + 193LL) &= ~0x10u;
          sub_736C90(*(_QWORD *)(v18 + 88), 0);
        }
      }
      else if ( v120[0] )
      {
        sub_6854C0(0x10Au, v102, v18);
        goto LABEL_20;
      }
      if ( dword_4F077BC )
      {
        if ( qword_4F077A8 <= 0x9EFBu )
        {
          if ( HIDWORD(qword_4D04464) )
          {
            v86 = *(_QWORD *)(v18 + 88);
            if ( (*(_BYTE *)(v86 + 206) & 8) != 0 && !*(_DWORD *)(v86 + 160) )
            {
              v99 = *(_QWORD *)(v18 + 88);
              v90 = *(__int64 **)(v86 + 80);
              sub_86AA40(v99, v57, v90, dword_4F077BC, v59, v60);
              v87 = v90;
              *(_QWORD *)(v99 + 264) = 0;
              if ( v90 )
              {
                while ( 1 )
                {
                  v88 = *((_BYTE *)v87 + 33);
                  if ( (v88 & 0x10) == 0 )
                    break;
                  v87 = (__int64 *)*v87;
                  if ( !v87 )
                    goto LABEL_196;
                }
                *((_BYTE *)v87 + 33) = v88 & 0xEF;
              }
LABEL_196:
              *(_BYTE *)(v18 + 81) &= ~2u;
              *(_BYTE *)(v99 + 193) &= ~0x20u;
              *(_DWORD *)(v99 + 204) &= 0xFFF7FEFF;
              for ( m = *(_QWORD *)(v99 + 152); *(_BYTE *)(m + 140) == 12; m = *(_QWORD *)(m + 160) )
                ;
              *(_QWORD *)(*(_QWORD *)(m + 168) + 56LL) = *(_QWORD *)(v12 + 56);
              sub_685920(v102, (FILE *)v18, 5u);
            }
          }
        }
      }
      if ( (*(_BYTE *)(v18 + 81) & 2) == 0 )
      {
        v67 = *(_QWORD *)(v18 + 88);
        v105 = *(_DWORD *)(v18 + 48);
        v104 = *(_WORD *)(v18 + 52);
        for ( n = *(_QWORD *)(v67 + 152); *(_BYTE *)(n + 140) == 12; n = *(_QWORD *)(n + 160) )
          ;
        v120[0] = n;
        *(_QWORD *)(v5 + 296) = n;
        if ( unk_4D03CB8 )
        {
          v94 = v67;
          sub_858370(v5 + 184);
          v67 = v94;
        }
        v69 = *(_QWORD *)(v5 + 8);
        v70 = *(_BYTE *)(v67 + 193);
        if ( (v70 & 1) != ((v69 & 0x80000) != 0) || ((v70 & 4) != 0) != ((v69 & 0x100000) != 0) )
        {
          v71 = 2930;
          if ( (v70 & 4) == 0 )
          {
            v71 = 2383;
            if ( (v70 & 2) == 0 )
              v71 = (v69 & 0x100000) == 0 ? 2384 : 2931;
          }
          v72 = (FILE *)(v5 + 112);
          v95 = v67;
          if ( (v70 & 1) != 0 )
            v72 = (FILE *)(v5 + 48);
          sub_6854C0(v71, v72, v18);
          v67 = v95;
          v73 = *(_BYTE *)(v95 + 193);
          if ( (v73 & 2) == 0 )
          {
            v74 = v73 | 1;
            if ( (*(_BYTE *)(v5 + 10) & 8) == 0 )
              v74 = *(_BYTE *)(v95 + 193) | 4;
            *(_BYTE *)(v95 + 193) = v74 | 2;
          }
        }
        v96 = v67;
        sub_71CE00(v119, v120[0]);
        if ( dword_4D048B4 && !*(_QWORD *)(v12 + 56) )
        {
          v83 = *(_BYTE *)(v96 + 174);
          if ( v83 == 2 )
          {
            sub_5F93D0(v96, &v119);
          }
          else if ( v83 == 5 && ((*(_BYTE *)(v96 + 176) - 2) & 0xFD) == 0 )
          {
            for ( ii = v120[0]; *(_BYTE *)(ii + 140) == 12; ii = *(_QWORD *)(ii + 160) )
              ;
            *(_QWORD *)(v12 + 56) = *(_QWORD *)(*(_QWORD *)(ii + 168) + 56LL);
          }
        }
        sub_6464A0(v119, v18, (unsigned int *)(a3 + 24), 1u);
        v75 = *(_QWORD *)(a3 + 80);
        if ( v75 )
          sub_73BC00(v10, v75);
        v20 = *(__m128i **)(v18 + 88);
        sub_649200((__int64)v20, v10, 0, 1, v5);
        if ( v20[10].m128i_i8[14] == 1 )
        {
          v79 = *(_QWORD *)(*(_QWORD *)v113 + 96LL);
          if ( (*(_BYTE *)(v79 + 176) & 1) == 0 )
          {
            v98 = *(_QWORD *)(*(_QWORD *)v113 + 96LL);
            v85 = sub_72F310(v20, 1, v76, v77, v78, v79);
            v79 = v98;
            if ( v85 )
            {
              *(_BYTE *)(v98 + 176) |= 1u;
              if ( dword_4F077C4 == 2 && unk_4F07778 > 201401 )
              {
                sub_6851C0(0xAEBu, v102);
                v79 = v98;
              }
            }
          }
          v97 = v79;
          if ( (*(_WORD *)(v79 + 176) & 0x4010) != 0x10 )
          {
            if ( (unsigned int)sub_72F500(v20, v113, &v118, 1, 0) )
            {
              *(_WORD *)(v97 + 176) = *(_WORD *)(v97 + 176) & 0xBFE7 | (16 * (v118 & 1)) | 8;
              if ( dword_4F077C4 == 2 && unk_4F07778 > 201401 && !(unsigned int)sub_72F3C0(v120[0], v113, &v118, 1, 0) )
                sub_6851C0(0xAEBu, v102);
            }
          }
        }
        if ( *(_QWORD *)(v18 + 96) )
        {
          sub_648C10(v18, (__int64)v102);
          *(_BYTE *)(*(_QWORD *)(v18 + 88) + 195LL) |= 2u;
          *(_BYTE *)(*(_QWORD *)(v18 + 88) + 195LL) |= 4u;
          *(_BYTE *)(*(_QWORD *)(v18 + 96) + 80LL) &= ~1u;
        }
        v20[12].m128i_i8[11] |= 1u;
        sub_8756F0(3, v18, v102, *(_QWORD *)(a3 + 72));
        sub_64A300((__int64)v20, *(_QWORD *)(a3 + 80));
        v20[4].m128i_i64[0] = *(_QWORD *)(a1 + 8);
        sub_729470(v20, a4);
LABEL_22:
        if ( !dword_4D048B8 )
          sub_64A410(a3);
        if ( (*(_BYTE *)(a3 + 64) & 2) != 0 )
        {
          if ( v20[12].m128i_i8[0] < 0 )
          {
            if ( !dword_4D04824 )
            {
LABEL_27:
              v21 = v20[5].m128i_i8[8];
              v20[10].m128i_i8[12] = 2;
              v20[5].m128i_i8[8] = v21 & 0x8F | 0x10;
              goto LABEL_28;
            }
            goto LABEL_111;
          }
          sub_736C90(v20, 1);
          if ( (v20[12].m128i_i8[1] & 0x40) != 0 )
            sub_685460(0x1DFu, (FILE *)(a1 + 8), v18);
        }
        if ( !dword_4D04824 )
        {
          if ( v20[12].m128i_i8[0] < 0 )
            goto LABEL_27;
          if ( (v20[5].m128i_i8[8] & 0x70) == 0x20 )
          {
            v20[10].m128i_i8[12] = 0;
LABEL_110:
            v20[5].m128i_i8[8] |= 4u;
          }
LABEL_28:
          *(_QWORD *)v5 = v18;
          sub_650FD0(v5);
          if ( (*(_BYTE *)(a1 + 17) & 0x20) == 0 )
          {
            sub_644920((_QWORD *)v5, (*(_BYTE *)(a3 + 64) & 4) != 0);
            v115 = *(_WORD *)(v18 + 52);
            v109 = *(_DWORD *)(v18 + 48);
            *(_DWORD *)(v18 + 48) = v105;
            *(_WORD *)(v18 + 52) = v104;
            sub_648B00((__int64)v20, (_BYTE *)(v5 + 224), a1 + 8);
            *(_DWORD *)(v18 + 48) = v109;
            *(_WORD *)(v18 + 52) = v115;
          }
          if ( dword_4F04C40 != -1 && *(_QWORD *)(qword_4F04C68[0] + 776LL * (int)dword_4F04C40 + 456) )
            sub_884800(v20);
          if ( (*(_BYTE *)(a3 + 65) & 1) != 0 )
            sub_85E870(v10, v18);
          sub_64F530(v18);
          sub_854980(v18, 0);
          v118 = 0;
          v117 = 2;
          if ( !dword_4D041AC )
            goto LABEL_36;
LABEL_52:
          if ( (*(_BYTE *)(a3 + 64) & 0x10) == 0 )
          {
            for ( jj = *(_QWORD **)(a3 + 8); jj; jj = (_QWORD *)*jj )
            {
              v35 = *(_BYTE *)(jj[2] + 140LL);
              for ( kk = jj[2]; v35 == 12; v35 = *(_BYTE *)(kk + 140) )
                kk = *(_QWORD *)(kk + 160);
              if ( (unsigned __int8)(v35 - 9) <= 2u && (*(_BYTE *)(kk + 176) & 0x20) != 0 )
                sub_5EB950(8u, 603, jj[2], (__int64)(jj + 4));
            }
          }
          goto LABEL_36;
        }
LABEL_111:
        if ( (v20[5].m128i_i8[8] & 0x70) == 0x20 )
        {
          v62 = v20[12].m128i_i8[0] < 0;
          v20[10].m128i_i8[12] = 0;
          if ( !v62 )
            goto LABEL_110;
        }
        goto LABEL_28;
      }
      v92 = *(_QWORD *)a1;
      sub_685920(v102, (FILE *)v18, 8u);
      v103 = *(const __m128i **)(v18 + 88);
      v64 = *(_QWORD *)(v103[9].m128i_i64[1] + 168);
      *(_QWORD *)(v12 + 40) = *(_QWORD *)(v64 + 40);
      *(_BYTE *)(v12 + 21) = *(_BYTE *)(v64 + 21) & 1 | *(_BYTE *)(v12 + 21) & 0xFE;
      *(_BYTE *)(v12 + 18) = *(_BYTE *)(v64 + 18) & 0x7F | *(_BYTE *)(v12 + 18) & 0x80;
      *(_WORD *)(v12 + 18) = *(_WORD *)(v64 + 18) & 0x3F80 | *(_WORD *)(v12 + 18) & 0xC07F;
      *(__m128i *)a1 = _mm_loadu_si128(xmmword_4F06660);
      *(__m128i *)(a1 + 16) = _mm_loadu_si128(&xmmword_4F06660[1]);
      *(__m128i *)(a1 + 32) = _mm_loadu_si128(&xmmword_4F06660[2]);
      v65 = *(_QWORD *)dword_4F07508;
      v110 = _mm_loadu_si128(&xmmword_4F06660[3]);
      *(_BYTE *)(a1 + 17) |= 0x20u;
      *(_QWORD *)(a1 + 8) = v65;
      *(__m128i *)(a1 + 48) = v110;
      v18 = sub_647630(0xBu, a1, 0, 1);
      *(_QWORD *)v18 = v92;
      v66 = sub_646F50(v10, 0, 0xFFFFFFFF);
      *(_QWORD *)(v18 + 88) = v66;
      v20 = (__m128i *)v66;
      sub_877D80(v66, v18);
      sub_877E20(v18, v20, v113);
      sub_725ED0(v20, v103[10].m128i_u8[14]);
      v20[11] = _mm_loadu_si128(v103 + 11);
LABEL_21:
      v120[0] = v10;
      *(_QWORD *)(v5 + 296) = v10;
      goto LABEL_22;
    }
    v18 = *(_QWORD *)(v18 + 88);
    if ( v18 )
    {
      v61 = *(_BYTE *)(v18 + 80);
      goto LABEL_98;
    }
LABEL_165:
    v111 = v60;
    if ( (unsigned int)sub_8BFC10(v60, *(_QWORD *)(v5 + 288), *(_QWORD *)(a1 + 40)) )
    {
      v81 = 795;
      sub_6854C0(0x31Bu, v102, v111);
    }
    else
    {
      v81 = 493;
      v82 = *(_QWORD *)(a1 + 24);
      if ( *(_BYTE *)(v82 + 80) != 17 )
        v81 = 147;
      sub_6854C0(v81, v102, v82);
    }
    if ( (*(_BYTE *)(v5 + 125) & 8) != 0 )
      *(_QWORD *)(v119 + 160) = sub_72C930(v81);
    goto LABEL_20;
  }
  if ( (*(_BYTE *)(v112 + 16) & 2) != 0 )
  {
    sub_854430();
  }
  else
  {
    v37 = *(unsigned int *)(a3 + 40);
    sub_8600D0(1, v37, *(_QWORD *)(v5 + 288), 0);
    *(_DWORD *)(a3 + 40) = *(_DWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C64);
    while ( 1 )
    {
      if ( word_4F06418[0] != 1 )
      {
        v38 = 2;
        if ( !(unsigned int)sub_651B00(2u) )
          break;
      }
      v37 = 1;
      sub_660E20(0, 1, 0, 0, *(_QWORD *)(a3 + 8), 0, 0);
    }
    if ( dword_4F077C0 && word_4F06418[0] == 76 )
    {
      *(_BYTE *)(v112 + 16) |= 1u;
      sub_7B8B50(2, v37, v39, v40);
    }
    v41 = qword_4F04C68[0] + 776LL * dword_4F04C64;
    *(_QWORD *)(a3 + 88) = *(_QWORD *)(v41 + 328);
    *(_QWORD *)(v41 + 328) = 0;
    if ( *(_QWORD *)(a3 + 8) )
    {
      v106 = v5;
      v42 = 0;
      v43 = *(__int64 **)(a3 + 8);
      v44 = 0;
      while ( 1 )
      {
        v45 = v43[2];
        if ( !v45 )
        {
          v53 = v43[1];
          v54 = sub_72BA30(5);
          v43[2] = v54;
          v43[3] = v54;
          *((_WORD *)v43 + 20) = v43[5] & 0x7F00 | 0x8003;
          v43[4] = *(_QWORD *)(v53 + 48);
          sub_885FF0(v53, (unsigned int)dword_4F04C5C, (*(_BYTE *)(v53 + 82) & 4) != 0);
          if ( dword_4F077C4 != 2 && unk_4F07778 > 199900 )
          {
            v55 = 5;
            if ( dword_4D04964 )
              v55 = unk_4F07471;
            sub_6853B0(v55, 0x41Bu, (FILE *)(v53 + 48), v53);
          }
          v45 = v43[2];
        }
        v37 = (__int64)(v43 + 4);
        v38 = sub_72B0C0(v45, v43 + 4);
        *(_QWORD *)(v38 + 16) = v43[3];
        sub_73D590(v38);
        if ( v44 )
        {
          *v42 = v38;
          v43 = (__int64 *)*v43;
          if ( !v43 )
            goto LABEL_73;
        }
        else
        {
          v43 = (__int64 *)*v43;
          v44 = v38;
          if ( !v43 )
          {
LABEL_73:
            v5 = v106;
            v46 = v44;
            goto LABEL_74;
          }
        }
        v42 = (_QWORD *)v38;
      }
    }
    v46 = 0;
LABEL_74:
    *(_QWORD *)v112 = v46;
    if ( dword_4F077C4 == 2 )
    {
      *(_BYTE *)(v112 + 16) |= 2u;
      if ( *(_QWORD *)(a3 + 80) )
      {
        v38 = *(_QWORD *)(v5 + 288);
        v37 = 0;
        *(_QWORD *)(a3 + 80) = sub_73EDA0(v38, 0);
      }
    }
    else
    {
      *(_BYTE *)(v112 + 16) |= 4u;
    }
    v47 = qword_4F04C68[0] + 776LL * dword_4F04C64;
    v48 = *(_QWORD **)(v47 + 24);
    v49 = (_QWORD *)(v47 + 32);
    if ( !v48 )
      v48 = v49;
    *(_QWORD *)a3 = *v48;
    sub_854430();
    v50 = qword_4F04C68[0] + 776LL * dword_4F04C64;
    v51 = *(_QWORD *)(v50 + 232);
    *(_QWORD *)(a3 + 48) = v51;
    *(_QWORD *)(v50 + 232) = 0;
    sub_863FC0(v38, v37, v51, qword_4F04C68, v52);
  }
  sub_6523A0(a1, v5, a3, 3, &v117, v120, &v118, a4);
  if ( dword_4D041AC )
    goto LABEL_52;
LABEL_36:
  v22 = *(_QWORD *)(*(_QWORD *)v5 + 88LL);
  sub_65C210(v5);
  *(_BYTE *)(v22 + 173) = *(_BYTE *)(v5 + 268);
  sub_643EB0(v5, 0);
  if ( word_4F06418[0] != 75
    || !dword_4F077BC
    || qword_4F077A8 > 0x76BFu
    || (result = a1, (*(_DWORD *)(a1 + 16) & 0x22000) == 0) )
  {
    if ( (*(_BYTE *)(a3 + 64) & 0x18) != 0 )
    {
      sub_71DEE0((_BYTE *)v5, a3, v23, v24);
      return sub_7BE280(75, 65, 0, 0);
    }
    else
    {
      v27 = (*(_QWORD *)(v5 + 8) & 1) == 0;
      if ( (*(_BYTE *)(a1 + 17) & 0x20) == 0 )
        sub_737310(v22, 11);
      v28 = 0;
      if ( dword_4D0460C )
      {
        v29 = *(_QWORD *)v22;
        v114 = *(_QWORD *)v22;
        if ( *(_QWORD *)v22 )
        {
          v28 = malloc(100000, v29, v23, dword_4D0460C, v25, v26);
          sub_87D420(v28, v114, 100000, 0);
        }
        else
        {
          v63 = (_BYTE *)malloc(1, v29, v23, dword_4D0460C, v25, v26);
          *v63 = 0;
          v28 = (__int64)v63;
        }
      }
      unk_4D045D8("Scanning Function Body", v28);
      v30 = a3;
      sub_71E0E0(v22, a3, v27, v31, v32);
      if ( (*(_BYTE *)(a1 + 17) & 0x20) == 0 )
      {
        v30 = v22;
        sub_826A90(a1 + 8, v22);
      }
      if ( dword_4D0460C )
        _libc_free(v28, v30);
      return unk_4D045D0();
    }
  }
  return result;
}
