// Function: sub_7BF840
// Address: 0x7bf840
//
_QWORD *__fastcall sub_7BF840(__int64 a1, unsigned int a2, int *a3)
{
  __int64 v3; // r15
  int v4; // r13d
  __int16 v5; // cx
  unsigned int v6; // r12d
  char v7; // al
  bool v8; // di
  int v9; // r8d
  bool v10; // dl
  bool v11; // dl
  int v12; // r13d
  __int64 v13; // rdx
  int v14; // eax
  char v15; // dl
  __int16 v16; // cx
  int v17; // r13d
  int v18; // eax
  _QWORD *v19; // r14
  _BYTE *v20; // rax
  char v22; // al
  __int64 i; // rax
  __int64 v24; // rax
  _BYTE *v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rcx
  __int64 v28; // r8
  __int64 v29; // r9
  _BYTE *v30; // rax
  unsigned int *v31; // rsi
  __int64 v32; // rdx
  __int64 v33; // rcx
  __int64 v34; // r8
  __int64 v35; // r9
  int v36; // ecx
  int v37; // ecx
  _BYTE *v38; // rax
  unsigned int v39; // ecx
  bool v40; // zf
  int v41; // ecx
  __m128i v42; // xmm5
  __m128i v43; // xmm6
  __m128i v44; // xmm7
  __int64 v45; // rax
  int v46; // eax
  _QWORD *v47; // r10
  unsigned __int64 v48; // rsi
  __int64 v49; // rdx
  char v50; // al
  __int64 v51; // rdi
  int v52; // esi
  __int64 j; // rcx
  __int64 v54; // r8
  __int64 v55; // rcx
  __int64 v56; // r8
  int v57; // r9d
  _QWORD *v58; // r10
  __int64 v59; // rax
  __int64 v60; // rax
  _QWORD *v61; // r13
  int *v62; // rax
  __int64 v63; // rcx
  __int64 v64; // r8
  __int64 v65; // r13
  __int64 v66; // r14
  __int64 v67; // rax
  _BOOL4 v68; // eax
  __int64 v69; // [rsp+8h] [rbp-F8h]
  _BOOL4 v70; // [rsp+10h] [rbp-F0h]
  _QWORD *v71; // [rsp+10h] [rbp-F0h]
  char v72; // [rsp+18h] [rbp-E8h]
  int v73; // [rsp+18h] [rbp-E8h]
  __int16 v74; // [rsp+30h] [rbp-D0h]
  __int16 v75; // [rsp+30h] [rbp-D0h]
  __int16 v76; // [rsp+38h] [rbp-C8h]
  __int16 v77; // [rsp+38h] [rbp-C8h]
  __int64 v78; // [rsp+38h] [rbp-C8h]
  char *v79; // [rsp+38h] [rbp-C8h]
  __int16 v80; // [rsp+38h] [rbp-C8h]
  __int64 v81; // [rsp+38h] [rbp-C8h]
  __int16 v82; // [rsp+38h] [rbp-C8h]
  _QWORD *v83; // [rsp+40h] [rbp-C0h]
  __int64 v84; // [rsp+48h] [rbp-B8h]
  int v87; // [rsp+5Ch] [rbp-A4h]
  int v88; // [rsp+64h] [rbp-9Ch] BYREF
  int v89; // [rsp+68h] [rbp-98h] BYREF
  int v90; // [rsp+6Ch] [rbp-94h] BYREF
  __int64 *v91; // [rsp+70h] [rbp-90h]
  _QWORD *v92; // [rsp+78h] [rbp-88h] BYREF
  __int64 v93; // [rsp+80h] [rbp-80h] BYREF
  __int64 v94; // [rsp+88h] [rbp-78h] BYREF
  __m128i v95; // [rsp+90h] [rbp-70h] BYREF
  __m128i v96; // [rsp+A0h] [rbp-60h] BYREF
  __m128i v97; // [rsp+B0h] [rbp-50h] BYREF
  __m128i v98[4]; // [rsp+C0h] [rbp-40h] BYREF

  v3 = a1;
  *a3 = 0;
  v4 = a2 & 0x4000;
  v91 = 0;
  v88 = 0;
  v93 = -1;
  v5 = sub_7BE840(0, 0);
  v94 = *(_QWORD *)&dword_4F063F8;
  v95 = _mm_loadu_si128((const __m128i *)&qword_4D04A00);
  v6 = dword_4F077BC;
  v96 = _mm_loadu_si128((const __m128i *)&word_4D04A10);
  v97 = _mm_loadu_si128(&xmmword_4D04A20);
  v98[0] = _mm_loadu_si128((const __m128i *)&unk_4D04A30);
  if ( !dword_4F077BC )
    goto LABEL_4;
  v6 = qword_4F077B4;
  if ( (_DWORD)qword_4F077B4 )
  {
    v83 = 0;
    v6 = 0;
    v84 = 0;
    if ( !a1 )
      goto LABEL_18;
    goto LABEL_6;
  }
  if ( qword_4F077A8 > 0x9C3Fu || !a1 || v5 != 43 )
  {
LABEL_4:
    v83 = 0;
    v84 = 0;
    goto LABEL_5;
  }
  v22 = sub_877F80(a1);
  v5 = 43;
  v83 = 0;
  v84 = 0;
  if ( v22 == 1 )
  {
    for ( i = *(_QWORD *)(a1 + 64); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
      ;
    v78 = *(_QWORD *)(a1 + 64);
    v84 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)i + 96LL) + 72LL);
    if ( v84 )
    {
      v6 = 1;
      v24 = sub_892920();
      v83 = (_QWORD *)a1;
      v5 = 43;
      v3 = v24;
      v84 = v78;
LABEL_5:
      if ( v3 )
        goto LABEL_6;
LABEL_18:
      if ( (a2 & 0x4300) != 0 )
      {
LABEL_32:
        v18 = v88;
        v19 = (_QWORD *)v3;
        goto LABEL_33;
      }
      v12 = a2 & 0x8000;
      if ( (a2 & 0x8000) != 0 )
      {
        v82 = v5;
        v12 = 0;
        sub_6851C0(0x348u, &v94);
        v5 = v82;
        v92 = 0;
        goto LABEL_103;
      }
      if ( (a2 & 0x10000) != 0 || (word_4D04A10 & 0x2000) != 0 )
      {
        v92 = 0;
        goto LABEL_103;
      }
      if ( !dword_4F077BC
        || qword_4F077A8 > 0x76BFu
        || (v75 = v5, v68 = sub_67D810((unsigned int *)&v94), v5 = v75, !v68)
        || (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) & 2) == 0
        || (a2 & 0x40000) != 0 )
      {
        v76 = v5;
        v12 = 0;
        sub_6851A0(0x360u, &v94, *(_QWORD *)(qword_4D04A00 + 8));
        v5 = v76;
        v92 = 0;
        goto LABEL_103;
      }
LABEL_16:
      v92 = 0;
      v12 = 1;
LABEL_103:
      if ( v5 != 43 )
      {
        v3 = 0;
        if ( (a2 & 0x1000) != 0 )
        {
LABEL_108:
          v18 = v88;
          v19 = 0;
          goto LABEL_42;
        }
LABEL_105:
        if ( (a2 & 0x80001) != 0 )
          goto LABEL_32;
        goto LABEL_106;
      }
      v72 = 0;
      v79 = 0;
      goto LABEL_77;
    }
    v83 = 0;
  }
LABEL_6:
  v7 = *(_BYTE *)(v3 + 80);
  if ( v7 != 19 )
  {
    if ( v7 != 3 )
    {
      v8 = 1;
      v9 = a2 & 0x300;
      v10 = v9 != 0;
      if ( (*(_DWORD *)(v3 + 80) & 0x42000) != 0 )
      {
        if ( dword_4F077C4 != 2 )
          goto LABEL_10;
      }
      else
      {
        v8 = v7 == 13;
        if ( dword_4F077C4 != 2 )
          goto LABEL_10;
      }
      if ( (unsigned __int8)(v7 - 4) <= 2u )
        goto LABEL_73;
LABEL_10:
      v11 = v4 != 0 || v8 || v10;
      if ( !dword_4F077BC
        || qword_4F077A8 > 0x76BFu
        || (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) & 2) == 0
        || v7 == 3
        || dword_4F077C4 == 2 && (unsigned __int8)(v7 - 4) <= 2u )
      {
        if ( !v11 )
        {
          v12 = dword_4F04D80;
          if ( dword_4F04D80 )
            goto LABEL_32;
          v80 = v5;
          sub_6854C0(0x37Eu, (FILE *)&v94, v3);
          v5 = v80;
          v92 = 0;
          goto LABEL_103;
        }
        goto LABEL_111;
      }
      if ( v11 )
      {
LABEL_111:
        if ( v9 | v4 || !v8 )
          goto LABEL_32;
        v92 = 0;
        v12 = 0;
        goto LABEL_103;
      }
      goto LABEL_16;
    }
    if ( !*(_BYTE *)(v3 + 104)
      || (v13 = *(_QWORD *)(v3 + 88), (*(_BYTE *)(v13 + 177) & 0x10) == 0)
      || !*(_QWORD *)(*(_QWORD *)(v13 + 168) + 168LL) )
    {
      v9 = a2 & 0x300;
      v10 = v9 != 0;
      v8 = (*(_DWORD *)(v3 + 80) & 0x42000) != 0;
LABEL_73:
      if ( v5 == 43 && !v9 )
      {
        sub_6854C0(0x207u, (FILE *)&v94, v3);
        v72 = 0;
        v92 = 0;
        v79 = 0;
LABEL_76:
        v12 = 0;
LABEL_77:
        v3 = 0;
LABEL_78:
        sub_7296C0(&v89);
        v30 = (_BYTE *)qword_4F061C8;
        v31 = (unsigned int *)dword_4F07770;
        ++*(_BYTE *)(qword_4F061C8 + 52LL);
        if ( (_DWORD)v31 )
          ++v30[50];
        ++v30[81];
        ++v30[83];
        sub_7B8B50((unsigned __int64)&dword_4F07770, v31, v26, v27, v28, v29);
        sub_7B8B50((unsigned __int64)&dword_4F07770, v31, v32, v33, v34, v35);
        ++*(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 632);
        if ( v3 && (*(_BYTE *)(*(_QWORD *)(v3 + 88) + 160LL) & 6) == 0 )
          v91 = (__int64 *)sub_7C6D90(v3, 0, &v88, a2, &v93);
        else
          v91 = (__int64 *)sub_7BF3A0(1u, &v88);
        *(_QWORD *)dword_4F07508 = *(_QWORD *)&dword_4F063F8;
        if ( word_4F06418[0] != 44 )
          sub_7BC0A0(&v88);
        v36 = v88;
        --*(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 632);
        if ( !v36 && v72 )
        {
          v46 = sub_7AB730(v3, &v90);
          v48 = (unsigned __int64)v47;
          HIDWORD(v69) = v46;
          LODWORD(v69) = a2 & 0x400;
          v70 = (v46 | (unsigned int)v69) != 0;
          v19 = (_QWORD *)sub_8A0370(v3, (_DWORD)v47, v70, (_DWORD)v92, 0, (a2 & 0x80000) != 0, 0);
          v49 = v70;
          if ( (*((_BYTE *)v19 + 81) & 0x20) != 0 )
            v88 = 1;
          v50 = *((_BYTE *)v19 + 80);
          v51 = v19[11];
          if ( (a2 & 0x8000) == 0 && (v79[265] & 1) != 0 )
          {
            v52 = *(unsigned __int8 *)(v51 + 140);
            for ( j = v19[11]; (_BYTE)v52 == 12; v52 = *(unsigned __int8 *)(j + 140) )
              j = *(_QWORD *)(j + 160);
            v48 = (unsigned int)(v52 - 9);
            if ( (unsigned __int8)v48 <= 2u && (*(_BYTE *)(j + 177) & 0x20) != 0 )
            {
              v48 = *(_QWORD *)j;
              v54 = *(_QWORD *)(*(_QWORD *)j + 96LL);
              if ( *(_QWORD *)(v54 + 72) )
              {
                v48 = (unsigned __int64)&v90;
                HIDWORD(v69) = sub_7AB730(*(_QWORD *)(v54 + 72), &v90);
                v49 = HIDWORD(v69) | (unsigned int)v69;
                if ( v69 )
                {
                  v71 = v58;
                  v73 = v57;
                  v81 = v56;
                  v91 = (__int64 *)sub_72F240(*(const __m128i **)(*(_QWORD *)(v55 + 168) + 168LL));
                  v48 = (unsigned __int64)v71;
                  v59 = sub_8A0370(*(_QWORD *)(v81 + 72), (_DWORD)v71, 1, (_DWORD)v92, 0, v73, 0);
                  v49 = 1;
                  v19 = (_QWORD *)v59;
                  v50 = *(_BYTE *)(v59 + 80);
                }
                else
                {
                  HIDWORD(v69) = 0;
                  v50 = *((_BYTE *)v19 + 80);
                }
                v51 = v19[11];
              }
            }
          }
          if ( dword_4F077BC )
          {
            if ( !(_DWORD)qword_4F077B4 && qword_4F077A8 <= 0x9D6Bu )
            {
              v49 &= 1u;
              if ( (_DWORD)v49 )
              {
                v49 = qword_4F04C68[0] + 776LL * dword_4F04C64;
                if ( *(_BYTE *)(v49 + 4) == 8 )
                {
                  v48 = HIDWORD(v69);
                  if ( (!HIDWORD(v69) || !v90) && (a2 & 0x18000) == 0 && (unsigned __int8)(v50 - 4) <= 1u )
                  {
                    v49 = *(_BYTE *)(v51 + 177) & 0x30;
                    if ( (_BYTE)v49 == 48 && *(char *)(v51 + 177) >= 0 )
                    {
                      v60 = *(_QWORD *)(v3 + 88);
                      v61 = *(_QWORD **)(v60 + 176);
                      if ( v61 )
                      {
                        if ( HIDWORD(v69) )
                        {
                          v48 = (unsigned __int64)&v94;
                          sub_686C80(0x67Fu, (FILE *)&v94, (__int64)v19, *(_QWORD *)(v60 + 176));
                        }
                        v51 = v61[11];
                        v19 = v61;
                      }
                    }
                  }
                }
              }
            }
          }
          v62 = (int *)sub_8E3660(v51, v48, v49);
          if ( v93 >= 0 )
          {
            v63 = *v62;
            if ( (_DWORD)v63 == -1 || v93 < v63 )
              *v62 = v93;
          }
          if ( v6 )
          {
            v65 = v19[11];
            v66 = v19[12];
            if ( dword_4F077C4 == 2 && (unsigned int)sub_8D23B0(v65) && (unsigned int)sub_8D3A70(v65) )
              sub_8AD220(v65, 0);
            if ( v65 == v84 || (unsigned int)sub_8D97D0(v65, v84, 0, v63, v64) )
            {
              v19 = *(_QWORD **)(v66 + 8);
            }
            else
            {
              sub_6861A0(0x375u, dword_4F07508, v65, v84);
              v19 = v83;
            }
          }
          v37 = 0;
          LOBYTE(v12) = 1;
        }
        else if ( v12 )
        {
          v37 = 0;
          LOBYTE(v12) = 0;
          v19 = 0;
        }
        else
        {
          if ( v91 )
            sub_725130(v91);
          sub_885B10(&qword_4D04A00);
          v19 = qword_4D04A18;
          v37 = 1;
        }
        v87 = v37;
        sub_729730(v89);
        v38 = (_BYTE *)qword_4F061C8;
        v39 = dword_4F07770;
        --*(_BYTE *)(qword_4F061C8 + 52LL);
        v40 = v39 == 0;
        v41 = v87;
        if ( !v40 )
          --v38[50];
        --v38[81];
        --v38[83];
        if ( word_4F06418[0] != 44 )
        {
          sub_7BEC40();
          v41 = v87;
        }
        word_4F06418[0] = 1;
        if ( v41 )
        {
          v18 = v88;
          goto LABEL_107;
        }
        v41 = 1;
LABEL_97:
        v42 = _mm_loadu_si128(&v96);
        v43 = _mm_loadu_si128(&v97);
        v44 = _mm_loadu_si128(v98);
        v18 = v88;
        *(__m128i *)&qword_4D04A00 = _mm_loadu_si128(&v95);
        *(__m128i *)&word_4D04A10 = v42;
        xmmword_4D04A20 = v43;
        unk_4D04A30 = v44;
        goto LABEL_98;
      }
      goto LABEL_10;
    }
  }
  v77 = v5;
  v92 = (_QWORD *)v3;
  v14 = sub_85F9C0(&v92);
  v15 = *(_BYTE *)(v3 + 80);
  v16 = v77;
  v17 = v14;
  switch ( v15 )
  {
    case 4:
    case 5:
      v79 = *(char **)(*(_QWORD *)(v3 + 96) + 80LL);
      goto LABEL_63;
    case 6:
      v79 = *(char **)(*(_QWORD *)(v3 + 96) + 32LL);
      goto LABEL_63;
    case 9:
    case 10:
      v79 = *(char **)(*(_QWORD *)(v3 + 96) + 56LL);
      goto LABEL_63;
    case 19:
    case 20:
    case 21:
    case 22:
      v79 = *(char **)(v3 + 88);
LABEL_63:
      if ( !v79 || (v79[265] & 3) != 1 )
        goto LABEL_54;
      v74 = v16;
      sub_6854C0(0x760u, (FILE *)&v94, v3);
      v79[266] |= 0x10u;
      if ( v74 == 43 )
      {
        v72 = 0;
        goto LABEL_76;
      }
      if ( (a2 & 0x1000) != 0 )
        goto LABEL_108;
      v3 = 0;
      if ( !v17 )
        goto LABEL_105;
      goto LABEL_117;
    default:
      v79 = 0;
LABEL_54:
      if ( v16 == 43 )
      {
        if ( v15 == 3 )
        {
          if ( *(_BYTE *)(v3 + 104) )
          {
            v67 = *(_QWORD *)(v3 + 88);
            if ( (*(_BYTE *)(v67 + 177) & 0x10) != 0 )
            {
              if ( *(_QWORD *)(*(_QWORD *)(v67 + 168) + 168LL) )
              {
                v3 = sub_880FE0(v3);
                switch ( *(_BYTE *)(v3 + 80) )
                {
                  case 4:
                  case 5:
                    v79 = *(char **)(*(_QWORD *)(v3 + 96) + 80LL);
                    break;
                  case 6:
                    v79 = *(char **)(*(_QWORD *)(v3 + 96) + 32LL);
                    break;
                  case 9:
                  case 0xA:
                    v79 = *(char **)(*(_QWORD *)(v3 + 96) + 56LL);
                    break;
                  case 0x13:
                  case 0x14:
                  case 0x15:
                  case 0x16:
                    v79 = *(char **)(v3 + 88);
                    break;
                  default:
                    v79 = 0;
                    break;
                }
              }
            }
          }
        }
        if ( dword_4F04C64 == -1 )
        {
          v72 = 1;
          v12 = 0;
        }
        else
        {
          v25 = (_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64);
          v72 = v25[7] & 1;
          if ( v72 )
          {
            if ( dword_4F04C44 != -1 || (v25[6] & 6) != 0 || (v12 = 0, v25[4] == 12) )
            {
              sub_867130(v3, &v94, 0, 0);
              v12 = 0;
            }
          }
          else
          {
            v72 = 1;
            v12 = 0;
          }
        }
        goto LABEL_78;
      }
      if ( (a2 & 0x1000) != 0 )
        goto LABEL_32;
      if ( v15 == 3 )
      {
        if ( *(_BYTE *)(v3 + 104) )
        {
          v45 = *(_QWORD *)(v3 + 88);
          if ( (*(_BYTE *)(v45 + 177) & 0x10) != 0 )
          {
            if ( *(_QWORD *)(*(_QWORD *)(v45 + 168) + 168LL) )
            {
              v19 = (_QWORD *)v3;
LABEL_35:
              if ( dword_4F04C64 != -1 )
              {
                v20 = (_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64);
                if ( (v20[7] & 1) != 0 && (dword_4F04C44 != -1 || (v20[6] & 6) != 0 || v20[4] == 12) )
                  sub_867130(v3, &v94, 0, 0);
              }
              v18 = v88;
              goto LABEL_42;
            }
          }
        }
      }
      if ( !v17 )
      {
        if ( (a2 & 0x80001) != 0 )
          goto LABEL_32;
        goto LABEL_120;
      }
LABEL_117:
      if ( (word_4D04A10 & 1) == 0 && (v79[160] >= 0 || *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 4) != 9) )
      {
        v41 = 0;
        LOBYTE(v12) = 0;
        v19 = v92;
        word_4F06418[0] = 1;
        goto LABEL_97;
      }
      if ( (a2 & 0x80001) != 0 )
        goto LABEL_32;
      if ( v3 )
LABEL_120:
        sub_6854C0(0x1B9u, (FILE *)&v94, v3);
LABEL_106:
      LOBYTE(v12) = 0;
      sub_885B10(&qword_4D04A00);
      v19 = qword_4D04A18;
      v41 = 0;
      v88 = 1;
      word_4F06418[0] = 1;
      v18 = 1;
LABEL_107:
      qword_4D04A08 = v95.m128i_i64[1];
LABEL_98:
      if ( v19 )
      {
        qword_4D04A18 = v19;
        HIBYTE(word_4D04A10) = ((v12 & 1) << 6) | HIBYTE(word_4D04A10) & 0xBF;
        qword_4D04A00 = *v19;
      }
      unk_4D04A12 = (v6 ^ 1) & 1 | unk_4D04A12 & 0xFE;
      *(_QWORD *)dword_4F07508 = v94;
      if ( !v41 )
      {
LABEL_33:
        if ( v18 || !v3 )
          goto LABEL_42;
        goto LABEL_35;
      }
LABEL_42:
      *a3 = v18;
      return v19;
  }
}
