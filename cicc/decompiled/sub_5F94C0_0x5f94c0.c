// Function: sub_5F94C0
// Address: 0x5f94c0
//
_QWORD *__fastcall sub_5F94C0(int a1)
{
  __int64 v1; // rdx
  _QWORD *result; // rax
  int v3; // r15d
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 *v6; // rbx
  __int64 v7; // r12
  __int64 *v8; // rbx
  __int64 v9; // rsi
  __int64 v10; // r12
  __int64 v11; // rcx
  __int64 v12; // rcx
  int v13; // eax
  __int64 v14; // rbx
  __int64 v15; // rdx
  __int64 v16; // r8
  __int64 v17; // r12
  __int64 i; // rbx
  __int64 v19; // rcx
  __int64 v20; // rax
  _QWORD *v21; // rax
  _QWORD *v22; // rbx
  unsigned __int64 v23; // rcx
  __int64 v24; // r14
  __int64 v25; // rdi
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // r13
  bool v30; // r12
  __int64 v31; // r15
  __int64 v32; // rdx
  __int64 v33; // rax
  char v34; // al
  __int64 v35; // rax
  int v36; // eax
  __int64 v37; // rax
  char v38; // dl
  __int64 v39; // r12
  __int64 v40; // rdi
  __int64 v41; // r12
  __int64 v42; // rdi
  __int64 v43; // rdx
  __int64 v44; // rcx
  char v45; // r9
  __int64 v46; // rax
  __int64 v47; // rdx
  __int64 *j; // rax
  __int64 *v49; // rcx
  int v50; // eax
  __int64 v51; // rax
  __int64 **v52; // r13
  __int64 **v53; // r14
  __int64 v54; // r12
  __int64 *v55; // rbx
  __int64 *v56; // r12
  __int64 v57; // rax
  __int64 v58; // rsi
  __int64 v59; // rdi
  char v60; // al
  __int64 v61; // r8
  __int64 k; // rax
  _BYTE *v63; // rax
  __int64 v64; // rcx
  _QWORD *v65; // rdx
  __int64 v66; // rdi
  __int64 *v67; // r12
  _QWORD *v68; // rbx
  __int64 v69; // r13
  unsigned int v70; // [rsp+4h] [rbp-7Ch]
  _QWORD *v71; // [rsp+10h] [rbp-70h]
  __int64 v72; // [rsp+18h] [rbp-68h]
  int v73; // [rsp+20h] [rbp-60h]
  char v74; // [rsp+26h] [rbp-5Ah]
  bool v75; // [rsp+27h] [rbp-59h]
  char v76; // [rsp+27h] [rbp-59h]
  int v77; // [rsp+28h] [rbp-58h]
  unsigned int v78; // [rsp+2Ch] [rbp-54h]
  __int64 *v79; // [rsp+30h] [rbp-50h]
  __int64 *v80; // [rsp+38h] [rbp-48h]
  _BOOL4 v82; // [rsp+44h] [rbp-3Ch]
  int v83; // [rsp+48h] [rbp-38h]
  unsigned int v84; // [rsp+48h] [rbp-38h]
  unsigned int v85; // [rsp+48h] [rbp-38h]

  v1 = qword_4F04C68[0];
  if ( a1 || dword_4F04C58 == -1 )
  {
    result = (_QWORD *)(qword_4F04C68[0] + 776LL * unk_4F04C24);
    if ( result[90] )
      return result;
  }
  else
  {
    result = (_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C58);
    if ( result[90] )
      return result;
  }
  if ( !*((_DWORD *)result + 178) )
  {
    v3 = 0;
LABEL_8:
    while ( 1 )
    {
      v4 = a1 || dword_4F04C58 == -1 ? 776LL * unk_4F04C24 : 776LL * dword_4F04C58;
      v5 = v1 + v4;
      v6 = *(__int64 **)(v5 + 728);
      v80 = v6;
      if ( !v6 )
        break;
      ++*(_DWORD *)(v5 + 712);
      *(_QWORD *)(v5 + 728) = 0;
      *(_QWORD *)(v5 + 736) = 0;
      ++qword_4D03B78;
      do
      {
        v7 = sub_880F80(*(_QWORD *)v6[1]);
        if ( v7 != unk_4D03FF0 )
        {
          if ( v3 && (sub_8D0B10(), v7 == unk_4D03FF0) )
          {
            v3 = 0;
          }
          else
          {
            v3 = 1;
            sub_8D0A80(v7);
          }
        }
        sub_5EAAC0(v6[1], *((_DWORD *)v6 + 4), 0);
        v6 = (__int64 *)*v6;
      }
      while ( v6 );
      v8 = v80;
      do
      {
        v10 = sub_880F80(*(_QWORD *)v8[1]);
        if ( v10 != unk_4D03FF0 )
        {
          if ( v3 && (sub_8D0B10(), v10 == unk_4D03FF0) )
          {
            v3 = 0;
          }
          else
          {
            v3 = 1;
            sub_8D0A80(v10);
          }
        }
        v9 = *((unsigned int *)v8 + 4);
        sub_5EAAC0(v8[1], v9, 1u);
        v8 = (__int64 *)*v8;
      }
      while ( v8 );
      v1 = qword_4F04C68[0];
      if ( a1 || dword_4F04C58 == -1 )
        v11 = 776LL * unk_4F04C24;
      else
        v11 = 776LL * dword_4F04C58;
      v12 = qword_4F04C68[0] + v11;
      v13 = *(_DWORD *)(v12 + 712) - 1;
      *(_DWORD *)(v12 + 712) = v13;
      --qword_4D03B78;
      if ( !v13 )
      {
        if ( unk_4D04424 )
        {
          v67 = v80;
          do
          {
            v9 = dword_4D04420;
            if ( !dword_4D04420 )
            {
              v68 = (_QWORD *)v67[1];
              v69 = sub_880F80(*v68);
              if ( v69 != unk_4D03FF0 )
              {
                if ( v3 && (sub_8D0B10(), v69 == unk_4D03FF0) )
                {
                  v3 = 0;
                }
                else
                {
                  v3 = 1;
                  sub_8D0A80(v69);
                }
              }
              if ( (*((_BYTE *)v68 + 141) & 0x20) == 0 )
              {
                v9 = *((unsigned int *)v67 + 4);
                sub_5E8530(v68, v9);
              }
            }
            v67 = (__int64 *)*v67;
          }
          while ( v67 );
        }
        v73 = v3;
        v79 = v80;
        while ( 1 )
        {
          v14 = v79[1];
          v17 = sub_880F80(*(_QWORD *)v14);
          if ( v17 != unk_4D03FF0 )
          {
            if ( v73 && (sub_8D0B10(), v17 == unk_4D03FF0) )
            {
              v73 = 0;
            }
            else
            {
              sub_8D0A80(v17);
              v73 = 1;
            }
          }
          if ( (*(_BYTE *)(v14 + 177) & 0x20) != 0 )
            goto LABEL_45;
          v15 = dword_4F077BC;
          if ( dword_4F077BC )
          {
            if ( !dword_4F077B4 )
            {
              if ( qword_4F077A8 <= 0x1869Fu )
                goto LABEL_39;
              goto LABEL_45;
            }
          }
          else if ( !dword_4F077B4 )
          {
            goto LABEL_45;
          }
          if ( unk_4F077A0 <= 0x15F8Fu )
          {
LABEL_39:
            for ( i = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v14 + 168) + 152LL) + 144LL); i; i = *(_QWORD *)(i + 112) )
            {
              while ( (*(_BYTE *)(i + 206) & 0x18) != 8 || sub_5EB8C0(i) )
              {
                i = *(_QWORD *)(i + 112);
                if ( !i )
                  goto LABEL_45;
              }
              sub_5F9380(i, v9, v15, v19, v16);
            }
          }
LABEL_45:
          v20 = v79[1];
          v72 = v20;
          if ( *(_BYTE *)(v20 + 140) == 12 )
          {
            do
              v20 = *(_QWORD *)(v20 + 160);
            while ( *(_BYTE *)(v20 + 140) == 12 );
          }
          else
          {
            v20 = v79[1];
          }
          v21 = *(_QWORD **)(*(_QWORD *)v20 + 96LL);
          v22 = (_QWORD *)v21[7];
          v71 = v21;
          if ( !v22 )
            goto LABEL_147;
          v23 = *((unsigned int *)v79 + 4);
          v78 = *((_DWORD *)v79 + 4);
          if ( (*(_BYTE *)(v72 + 177) & 0x30) == 0x30 )
          {
            v77 = 1;
            v82 = 0;
          }
          else
          {
            v77 = 0;
            v82 = v23 != 0;
          }
          v24 = 0;
          v70 = unk_4F066AC;
          v74 = v77 ^ 1;
          do
          {
            v25 = v22[1];
            v29 = (__int64)v22;
            v22 = (_QWORD *)*v22;
            v30 = (*(_BYTE *)(v25 + 178) & 4) != 0;
            if ( *(_QWORD *)(v29 + 160) )
            {
              v31 = *(_QWORD *)(v29 + 16);
            }
            else
            {
              if ( (*(_BYTE *)(v29 + 184) & 2) == 0 )
                goto LABEL_57;
              v31 = *(_QWORD *)(v29 + 16);
              if ( (*(_BYTE *)(v31 + 81) & 2) == 0 )
                goto LABEL_57;
            }
            v9 = v31;
            v16 = (unsigned int)sub_5E9520(v25, v31);
            if ( v25 != v24 )
            {
              if ( v24 )
              {
                if ( dword_4F07588 )
                {
                  v33 = *(_QWORD *)(v24 + 32);
                  if ( *(_QWORD *)(v25 + 32) == v33 )
                  {
                    if ( v33 )
                      goto LABEL_72;
                  }
                }
                v83 = v16;
                sub_866010(v25, v31, v32, v23, v16);
                v25 = *(_QWORD *)(v29 + 8);
                LODWORD(v16) = v83;
              }
              v9 = v78;
              v84 = v16;
              sub_866000(v25, v78, 1);
              v24 = *(_QWORD *)(v29 + 8);
              v16 = v84;
            }
LABEL_72:
            if ( v82 && !(_DWORD)v16 && (*(_BYTE *)(v29 + 184) & 1) == 0 && !v30
              || !dword_4D047B0
              && (dword_4F04C64 == -1
               || (v23 = (unsigned __int64)qword_4F04C68,
                   (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 7) & 1) == 0))
              && v77
              && ((_DWORD)v16 || (*(_BYTE *)(v29 + 184) & 1) != 0) )
            {
              v25 = v29 + 152;
              sub_7AEA70(v29 + 152);
              goto LABEL_57;
            }
            v25 = dword_4D047A0;
            if ( !dword_4D047A0 )
            {
              v34 = *(_BYTE *)(v29 + 184);
              v23 = v82;
              v38 = v34;
              if ( v82 )
                goto LABEL_110;
              v15 = *(_BYTE *)(v29 + 184) & 2;
LABEL_80:
              if ( (_BYTE)v15 )
                goto LABEL_81;
              goto LABEL_113;
            }
            v9 = v82;
            if ( !v82 )
            {
              v34 = *(_BYTE *)(v29 + 184);
              v15 = v34 & 2;
              goto LABEL_80;
            }
            v50 = *(unsigned __int8 *)(v31 + 80);
            v15 = (unsigned int)(v50 - 10);
            if ( (unsigned __int8)(v50 - 10) > 1u && (_BYTE)v50 != 17 )
            {
LABEL_154:
              v38 = *(_BYTE *)(v29 + 184);
LABEL_110:
              v34 = v38;
              v15 = v38 & 2;
              if ( (_DWORD)v15 )
              {
                if ( !(_DWORD)v16 )
                {
                  if ( !(_BYTE)v15 )
                    goto LABEL_113;
LABEL_81:
                  if ( unk_4D047AC )
                  {
                    switch ( *(_BYTE *)(v31 + 80) )
                    {
                      case 4:
                      case 5:
                        v35 = *(_QWORD *)(*(_QWORD *)(v31 + 96) + 80LL);
                        break;
                      case 6:
                        v35 = *(_QWORD *)(*(_QWORD *)(v31 + 96) + 32LL);
                        break;
                      case 9:
                      case 0xA:
                        v35 = *(_QWORD *)(*(_QWORD *)(v31 + 96) + 56LL);
                        break;
                      case 0x13:
                      case 0x14:
                      case 0x15:
                      case 0x16:
                        v35 = *(_QWORD *)(v31 + 88);
                        break;
                      default:
                        BUG();
                    }
                    v23 = v70;
                    *(_DWORD *)(*(_QWORD *)(v35 + 32) + 44LL) = v70;
                  }
                  v25 = v31;
                  v85 = v16;
                  v36 = sub_893570(v31, v9, v15, v23, v16);
                  v16 = v85;
                  if ( v36 && (!unk_4D047AC || *(_QWORD *)(*(_QWORD *)(v72 + 168) + 176LL) && unk_4D047A4) )
                  {
                    if ( (*(_BYTE *)(v29 + 184) & 4) != 0 )
                    {
                      v25 = v31;
                      sub_8950B0(v31);
                      v16 = v85;
                    }
                    if ( (_DWORD)v16 )
                    {
                      switch ( *(_BYTE *)(v31 + 80) )
                      {
                        case 4:
                        case 5:
                          v37 = *(_QWORD *)(*(_QWORD *)(v31 + 96) + 80LL);
                          goto LABEL_98;
                        case 6:
                          v37 = *(_QWORD *)(*(_QWORD *)(v31 + 96) + 32LL);
                          goto LABEL_98;
                        case 9:
                        case 0xA:
                          v37 = *(_QWORD *)(*(_QWORD *)(v31 + 96) + 56LL);
                          if ( (*(_BYTE *)(v29 + 184) & 4) == 0 )
                            break;
                          goto LABEL_99;
                        case 0x13:
                        case 0x14:
                        case 0x15:
                        case 0x16:
                          v37 = *(_QWORD *)(v31 + 88);
                          goto LABEL_98;
                        default:
                          v37 = 0;
LABEL_98:
                          if ( (*(_BYTE *)(v29 + 184) & 4) == 0 )
                            break;
LABEL_99:
                          *(_BYTE *)(*(_QWORD *)(v37 + 176) + 202LL) |= 0x80u;
                          v26 = dword_4F04C64;
                          if ( dword_4F04C64 == -1 )
                            goto LABEL_100;
                          goto LABEL_58;
                      }
                    }
                  }
                }
LABEL_57:
                v26 = dword_4F04C64;
                if ( dword_4F04C64 == -1 )
                {
LABEL_100:
                  if ( (*(_BYTE *)(v29 + 89) & 8) != 0 )
                    goto LABEL_60;
                  v25 = v29 + 32;
                  sub_87E280(v29 + 32);
                  if ( (*(_BYTE *)(v29 + 184) & 2) != 0 )
                  {
LABEL_61:
                    v28 = qword_4CF8000;
                    *(_QWORD *)(v29 + 128) = 0;
                    *(_QWORD *)v29 = v28;
                    qword_4CF8000 = v29;
                    continue;
                  }
                }
                else
                {
LABEL_58:
                  v23 = (unsigned __int64)qword_4F04C68;
                  v27 = qword_4F04C68[0] + 776 * v26;
                  if ( (*(_BYTE *)(v27 + 7) & 1) == 0 )
                    goto LABEL_100;
                  v15 = (__int64)&dword_4F04C44;
                  if ( dword_4F04C44 == -1 && (*(_BYTE *)(v27 + 6) & 2) == 0 )
                    goto LABEL_100;
LABEL_60:
                  if ( (*(_BYTE *)(v29 + 184) & 2) != 0 )
                    goto LABEL_61;
                }
                v25 = *(_QWORD *)(v29 + 128);
                sub_679050(v25);
                goto LABEL_61;
              }
LABEL_113:
              if ( v77
                && !(_DWORD)v16
                && !(v34 & 1 | *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 5) & 8)
                && !v30 )
              {
                switch ( *(_BYTE *)(v31 + 80) )
                {
                  case 4:
                  case 5:
                    v39 = *(_QWORD *)(*(_QWORD *)(v31 + 96) + 80LL);
                    break;
                  case 6:
                    v39 = *(_QWORD *)(*(_QWORD *)(v31 + 96) + 32LL);
                    break;
                  case 9:
                  case 0xA:
                    v39 = *(_QWORD *)(*(_QWORD *)(v31 + 96) + 56LL);
                    break;
                  case 0x13:
                  case 0x14:
                  case 0x15:
                  case 0x16:
                    v39 = *(_QWORD *)(v31 + 88);
                    break;
                  default:
                    MEMORY[0] = _mm_loadu_si128((const __m128i *)(v29 + 152));
                    MEMORY[0x10] = _mm_loadu_si128((const __m128i *)(v29 + 168));
                    BUG();
                }
                v25 = v29 + 152;
                v9 = 1;
                *(__m128i *)v39 = _mm_loadu_si128((const __m128i *)(v29 + 152));
                *(__m128i *)(v39 + 16) = _mm_loadu_si128((const __m128i *)(v29 + 168));
                sub_7ADF70(v29 + 152, 1);
                *(__m128i *)(v39 + 184) = _mm_loadu_si128((const __m128i *)(v29 + 24));
                *(__m128i *)(v39 + 200) = _mm_loadu_si128((const __m128i *)(v29 + 40));
                *(__m128i *)(v39 + 216) = _mm_loadu_si128((const __m128i *)(v29 + 56));
                *(__m128i *)(v39 + 232) = _mm_loadu_si128((const __m128i *)(v29 + 72));
                *(__m128i *)(v39 + 248) = _mm_loadu_si128((const __m128i *)(v29 + 88));
                *(__m128i *)(v39 + 264) = _mm_loadu_si128((const __m128i *)(v29 + 104));
                *(_QWORD *)(v39 + 280) = *(_QWORD *)(v29 + 120);
                *(_QWORD *)(v29 + 32) = 0;
                if ( !unk_4D047AC || *(_QWORD *)(*(_QWORD *)(v72 + 168) + 176LL) && unk_4D047A4 )
                {
                  v25 = v31;
                  if ( (unsigned int)sub_893570(v31, 1, v15, v23, v16) )
                  {
                    v25 = v31;
                    sub_8950B0(v31);
                    if ( *(_BYTE *)(*(_QWORD *)(v39 + 104) + 136LL) )
                    {
                      v23 = (unsigned __int64)v71;
                      *(_BYTE *)(*(_QWORD *)(v71[10] + 104LL) + 136LL) = 1;
                      *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v71[10] + 104LL) + 208LL) + 136LL) = 1;
                    }
                  }
                }
                goto LABEL_57;
              }
              v75 = (_DWORD)v16 == 0;
              v40 = v29 + 152;
              v41 = *(_QWORD *)(*(_QWORD *)(v29 + 16) + 88LL);
              if ( *(_BYTE *)(v29 + 176) )
                sub_7BC160(v40);
              else
                sub_7BC000(v40);
              v42 = v41;
              v9 = v29 + 24;
              sub_71E0E0(v41, v29 + 24, 22);
              v45 = v74 | v75;
              if ( word_4F06418[0] == 74 )
              {
                v76 = v74 | v75;
                sub_7B8B50(v41, v9, v43, v44);
                if ( word_4F06418[0] != 9 )
                {
                  v9 = (__int64)&dword_4F063F8;
                  v42 = 3095;
                  sub_6851C0(3095, &dword_4F063F8);
                  v45 = v76;
                  goto LABEL_131;
                }
                if ( v76 )
                {
LABEL_132:
                  if ( *(_BYTE *)(v29 + 176) )
                  {
                    v42 = v29 + 152;
                    sub_7AEA70(v29 + 152);
                  }
                  goto LABEL_134;
                }
              }
              else
              {
LABEL_131:
                if ( v45 )
                  goto LABEL_132;
LABEL_134:
                while ( word_4F06418[0] != 9 )
                  sub_7B8B50(v42, v9, v43, v44);
              }
              sub_7B8B50(v42, v9, v43, v44);
              v25 = *(_QWORD *)(v29 + 8);
              if ( v25 == v24 )
                goto LABEL_57;
              if ( v24 )
              {
                if ( v25 )
                {
                  v16 = dword_4F07588;
                  if ( dword_4F07588 )
                  {
                    v46 = *(_QWORD *)(v24 + 32);
                    if ( *(_QWORD *)(v25 + 32) == v46 )
                    {
                      if ( v46 )
                        goto LABEL_57;
                    }
                  }
                }
                sub_866010(v25, v9, v15, v23, v16);
                v25 = *(_QWORD *)(v29 + 8);
              }
              v9 = v78;
              sub_866000(v25, v78, 1);
              v24 = *(_QWORD *)(v29 + 8);
              goto LABEL_57;
            }
            if ( (_DWORD)v16 )
            {
              v51 = *(_QWORD *)(v31 + 88);
              if ( (*(_BYTE *)(v51 + 88) & 4) != 0 )
                goto LABEL_154;
            }
            else
            {
              v34 = *(_BYTE *)(v29 + 184);
              if ( (v34 & 1) == 0 )
              {
                v15 = v34 & 2;
                if ( !v30 )
                  goto LABEL_80;
              }
              v51 = *(_QWORD *)(v31 + 88);
            }
            if ( (*(_BYTE *)(v51 + 193) & 0x20) != 0 )
              goto LABEL_57;
            *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v29 + 16) + 88LL) + 344LL) = v29;
            *(_QWORD *)v29 = 0;
          }
          while ( v22 );
          if ( v24 )
            sub_866010(v25, v9, v15, v23, v16);
          v71[7] = 0;
LABEL_147:
          v79 = (__int64 *)*v79;
          if ( !v79 )
          {
            v3 = v73;
            v47 = qword_4CF7FF8;
            for ( j = v80; ; j = v49 )
            {
              v49 = (__int64 *)*j;
              *j = v47;
              v47 = (__int64)j;
              qword_4CF7FF8 = (__int64)j;
              if ( !v49 )
                break;
            }
            v1 = qword_4F04C68[0];
            goto LABEL_8;
          }
        }
      }
    }
    v52 = *(__int64 ***)(v5 + 744);
    if ( !v52 )
      goto LABEL_186;
    *(_QWORD *)(v5 + 744) = 0;
    v53 = v52;
    while ( 1 )
    {
      v54 = sub_880F80(*v53[1]);
      if ( v54 == unk_4D03FF0 )
        goto LABEL_180;
      if ( v3 && (sub_8D0B10(), v54 == unk_4D03FF0) )
      {
        v53 = (__int64 **)*v53;
        v3 = 0;
        if ( !v53 )
        {
LABEL_185:
          sub_878510(v52);
LABEL_186:
          v55 = (__int64 *)qword_4CF7FC8;
          v56 = &qword_4CF7FC8;
          if ( qword_4CF7FC8 )
          {
            do
            {
              while ( 1 )
              {
                v58 = v55[1];
                v59 = *(_QWORD *)(v58 + 88);
                if ( (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v59 + 40) + 32LL) + 141LL) & 0x20) == 0 )
                  break;
                v56 = v55;
                v55 = (__int64 *)*v55;
                if ( !v55 )
                  goto LABEL_193;
              }
              v57 = v55[2];
              if ( v57 )
              {
                if ( (unsigned int)sub_8DADD0(*(_QWORD *)(v59 + 152), *(_QWORD *)(*(_QWORD *)(v57 + 88) + 152LL)) )
                {
                  v61 = v55[2];
                  for ( k = *(_QWORD *)(*(_QWORD *)(v61 + 88) + 152LL); *(_BYTE *)(k + 140) == 12; k = *(_QWORD *)(k + 160) )
                    ;
                  v63 = *(_BYTE **)(*(_QWORD *)(k + 168) + 56LL);
                  if ( !v63 || (*v63 & 0x10) == 0 )
                  {
                    v64 = v55[1];
                    v65 = v55 + 4;
                    if ( (*(_BYTE *)(*(_QWORD *)(v64 + 88) + 193LL) & 0x10) != 0 )
                    {
                      v66 = 5;
                      if ( dword_4D048B8 )
                      {
                        if ( dword_4F077C4 != 2 || (v66 = 7, unk_4F07778 <= 201102) && !dword_4F07774 )
                        {
                          v66 = 5;
                          if ( unk_4D04964 )
                            v66 = unk_4F07471;
                        }
                      }
                      sub_686B60(v66, 768, v65, v64, v61);
                    }
                    else
                    {
                      sub_686B60(dword_4D048B8 == 0 ? 5 : 7, 766, v65, v64, v61);
                    }
                  }
                }
              }
              else
              {
                v60 = *(_BYTE *)(v59 + 174);
                if ( v60 == 2 || v60 == 5 && ((*(_BYTE *)(v59 + 176) - 2) & 0xFD) == 0 )
                {
                  sub_5F93D0(v59, v55 + 3);
                  v58 = v55[1];
                }
                sub_6464A0(v55[3], v58, v55 + 4, 1);
              }
              *v56 = *v55;
              *v55 = qword_4CF7FD0;
              qword_4CF7FD0 = (__int64)v55;
              v55 = (__int64 *)*v56;
            }
            while ( *v56 );
          }
LABEL_193:
          if ( v3 )
            sub_8D0B10();
          result = &qword_4D03B78;
          if ( !qword_4D03B78 )
            return (_QWORD *)sub_8ACAD0();
          return result;
        }
      }
      else
      {
        v3 = 1;
        sub_8D0A80(v54);
LABEL_180:
        v53 = (__int64 **)*v53;
        if ( !v53 )
          goto LABEL_185;
      }
    }
  }
  return result;
}
