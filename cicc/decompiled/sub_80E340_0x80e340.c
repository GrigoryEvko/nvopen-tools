// Function: sub_80E340
// Address: 0x80e340
//
__int64 __fastcall sub_80E340(__int64 a1, int a2, __int64 a3)
{
  const __m128i *v3; // r15
  int v6; // ebx
  __int8 v7; // al
  __int8 v8; // cl
  __int64 *v9; // rax
  __int64 result; // rax
  unsigned __int8 v11; // al
  __m128i *v12; // r13
  __int8 v13; // al
  char v14; // dl
  bool v15; // si
  char v16; // al
  __int8 v17; // al
  size_t v18; // rdx
  char *v19; // rsi
  char v20; // r13
  const __m128i *v21; // rdi
  __int8 v22; // al
  _QWORD *v23; // rdi
  __int64 v24; // rax
  int v25; // eax
  __int8 v26; // al
  const __m128i *v27; // r13
  _QWORD *v28; // rdi
  __int64 v29; // rax
  char *v30; // rsi
  _QWORD *v31; // rdi
  char v32; // al
  __int8 v33; // dl
  unsigned __int8 v34; // r13
  char *v35; // rax
  unsigned __int64 v36; // rax
  __int64 v37; // rdx
  _QWORD *v38; // rdi
  __int64 v39; // rax
  __int8 v40; // al
  __int64 *v41; // r13
  __int64 v42; // rdx
  char *v43; // r8
  char v44; // al
  __int64 v45; // rdx
  _QWORD *v46; // rdi
  __int64 v47; // rax
  __int64 i; // r13
  __int64 v49; // rax
  char *v50; // rax
  __int64 v51; // rax
  __int64 *v52; // rax
  __int64 *v53; // rax
  __int64 v54; // rdx
  const char *v55; // rsi
  __int64 v56; // rax
  char *v57; // rax
  const char *v58; // rax
  _QWORD *v59; // rdi
  char *v60; // r13
  __int64 v61; // rax
  _QWORD *v62; // rdi
  _QWORD *v63; // rdi
  unsigned __int8 v64; // [rsp+0h] [rbp-80h]
  const __m128i **v65; // [rsp+0h] [rbp-80h]
  _DWORD v67[28]; // [rsp+10h] [rbp-70h] BYREF

  v3 = (const __m128i *)a1;
  v6 = 0;
  v7 = *(_BYTE *)(a1 + 140);
  if ( v7 == 12 )
  {
    while ( 1 )
    {
      v6 |= v3[11].m128i_i8[9] & 0x7F;
      if ( v3[11].m128i_i64[0] )
      {
        v3 = (const __m128i *)v3[11].m128i_i64[0];
LABEL_12:
        if ( dword_4F077BC && (unsigned int)sub_8D2310(v3) && (*(_BYTE *)(v3[10].m128i_i64[1] + 20) & 1) != 0 )
        {
LABEL_39:
          v6 |= 2u;
          v12 = (__m128i *)sub_7259C0(7);
          sub_73C230(v3, v12);
          v13 = v3[-1].m128i_i8[8];
          v3 = v12;
          v12[-1].m128i_i8[8] = v13 & 8 | v12[-1].m128i_i8[8] & 0xF7;
          *(_BYTE *)(v12[10].m128i_i64[1] + 20) &= ~1u;
LABEL_40:
          sub_80C190(v6, (_QWORD *)a3);
          result = sub_80C5A0((__int64)v3, 6, 0, 0, v67, (_QWORD *)a3);
          if ( !(_DWORD)result )
          {
            v7 = v3[8].m128i_i8[12];
            goto LABEL_15;
          }
LABEL_33:
          if ( (a2 & 1) == 0 && v6 && !*(_QWORD *)(a3 + 40) )
            return sub_80A250(a1, 6, 0, a3);
          return result;
        }
        if ( v6 )
          goto LABEL_40;
LABEL_14:
        v7 = v3[8].m128i_i8[12];
        v6 = 0;
        goto LABEL_15;
      }
      v8 = v3[11].m128i_i8[8];
      if ( v8 != 1 )
        break;
      if ( !dword_4D0425C )
        goto LABEL_26;
      if ( (unsigned int)sub_809230((__int64)v3) )
        goto LABEL_12;
      if ( (v3[11].m128i_i8[10] & 8) == 0 )
        goto LABEL_27;
      v9 = sub_746BE0((__int64)v3);
      if ( !v9 )
        goto LABEL_12;
      v3 = (const __m128i *)*v9;
      result = sub_80C5A0(*v9, 6, 0, 0, v67, (_QWORD *)a3);
      if ( (_DWORD)result )
        return result;
LABEL_9:
      if ( v3[8].m128i_i8[12] != 12 )
        goto LABEL_12;
    }
    if ( v8 == 5 )
    {
      v14 = v3[11].m128i_i8[10] & 8;
      if ( v14 )
        goto LABEL_12;
    }
    else
    {
      if ( v8 == 3 || v8 == 2 )
      {
        if ( *(_DWORD *)(a3 + 52) )
          goto LABEL_12;
        v11 = v8 - 6;
        goto LABEL_25;
      }
      v14 = v3[11].m128i_i8[10] & 8;
      if ( (unsigned __int8)v8 > 0xAu )
      {
        v15 = 1;
        goto LABEL_46;
      }
    }
    v15 = ((0x71DuLL >> v8) & 1) == 0;
LABEL_46:
    v11 = v8 - 6;
    if ( (unsigned __int8)(v8 - 6) > 1u && (unsigned __int8)(v8 - 11) > 1u && v15 && v14 != 0 )
      goto LABEL_12;
LABEL_25:
    if ( v11 <= 1u )
    {
LABEL_26:
      if ( (v3[11].m128i_i8[10] & 8) != 0 )
        goto LABEL_12;
    }
LABEL_27:
    v3 = (const __m128i *)v3[10].m128i_i64[0];
    goto LABEL_9;
  }
  if ( dword_4F077BC )
  {
    if ( (unsigned int)sub_8D2310(a1) && (*(_BYTE *)(*(_QWORD *)(a1 + 168) + 20LL) & 1) != 0 )
      goto LABEL_39;
    goto LABEL_14;
  }
LABEL_15:
  if ( !v3->m128i_i64[1] )
  {
LABEL_18:
    switch ( v7 )
    {
      case 0:
      case 21:
        v18 = 1;
        v19 = "?";
        goto LABEL_57;
      case 1:
        v18 = 1;
        v19 = "v";
        goto LABEL_57;
      case 2:
        goto LABEL_49;
      case 3:
        switch ( v3[10].m128i_i8[0] )
        {
          case 0:
          case 0xA:
            v18 = 5;
            v19 = "DF16_";
            goto LABEL_57;
          case 1:
            v18 = 2;
            v19 = "Dh";
            goto LABEL_57;
          case 2:
            v18 = 1;
            v19 = "f";
            goto LABEL_57;
          case 3:
            v18 = 5;
            v19 = "DF32x";
            goto LABEL_57;
          case 4:
            v18 = 1;
            v19 = "d";
            goto LABEL_57;
          case 5:
            v18 = 5;
            v19 = "DF64x";
            goto LABEL_57;
          case 6:
            v18 = 1;
            v19 = (char *)&unk_3F885D4;
            goto LABEL_57;
          case 7:
            v18 = 9;
            v19 = "u7float80";
            goto LABEL_57;
          case 8:
            v18 = 1;
            v19 = "g";
            goto LABEL_57;
          case 9:
            v18 = 8;
            v19 = "u6__bf16";
            if ( !(_DWORD)qword_4F077B4 )
            {
              v18 = 5;
              v19 = "DF16b";
              if ( HIDWORD(qword_4F077B4) )
              {
                v18 = (-(__int64)(qword_4F06A78 == 0) & 0xFFFFFFFFFFFFFFFDLL) + 8;
                if ( qword_4F06A78 )
                  v19 = "u6__bf16";
              }
            }
            goto LABEL_57;
          case 0xB:
            v18 = 5;
            v19 = "DF32_";
            goto LABEL_57;
          case 0xC:
            v18 = 5;
            v19 = "DF64_";
            goto LABEL_57;
          case 0xD:
            v18 = 6;
            v19 = "DF128_";
            goto LABEL_57;
          default:
            goto LABEL_117;
        }
      case 5:
        switch ( v3[10].m128i_i8[0] )
        {
          case 0:
          case 0xA:
            v18 = 6;
            v19 = "CDF16_";
            goto LABEL_57;
          case 2:
            v18 = 2;
            v19 = "Cf";
            goto LABEL_57;
          case 3:
            v18 = 6;
            v19 = "CDF32x";
            goto LABEL_57;
          case 4:
            v18 = 2;
            v19 = "Cd";
            goto LABEL_57;
          case 5:
            v18 = 6;
            v19 = "CDF64x";
            goto LABEL_57;
          case 6:
            v18 = 2;
            v19 = "Ce";
            goto LABEL_57;
          case 7:
            v18 = 10;
            v19 = "Cu7float80";
            goto LABEL_57;
          case 8:
            v18 = 2;
            v19 = "Cg";
            goto LABEL_57;
          case 9:
            v18 = 9;
            v19 = "Cu6__bf16";
            if ( !(_DWORD)qword_4F077B4 )
            {
              v18 = 6;
              v19 = "CDF16b";
              if ( HIDWORD(qword_4F077B4) )
              {
                v18 = (-(__int64)(qword_4F06A78 == 0) & 0xFFFFFFFFFFFFFFFDLL) + 9;
                if ( qword_4F06A78 )
                  v19 = "Cu6__bf16";
              }
            }
            goto LABEL_57;
          case 0xB:
            v18 = 6;
            v19 = "CDF32_";
            goto LABEL_57;
          case 0xC:
            v18 = 6;
            v19 = "CDF64_";
            goto LABEL_57;
          case 0xD:
            v18 = 7;
            v19 = "CDF128_";
            goto LABEL_57;
          default:
            goto LABEL_117;
        }
      case 6:
        v25 = sub_7E1E50((__int64)v3);
        v18 = 2;
        v19 = "Dn";
        if ( !v25 )
        {
          v26 = v3[10].m128i_i8[8];
          v18 = 1;
          v19 = "P";
          if ( (v26 & 1) != 0 )
          {
            v19 = "O";
            if ( (v26 & 2) == 0 )
              v19 = "R";
          }
        }
        goto LABEL_57;
      case 7:
        v27 = v3;
        sub_80C2B0((__int64)v3, (_QWORD *)a3);
        if ( (v3[-1].m128i_i8[8] & 8) != 0 )
          v27 = (const __m128i *)v3[11].m128i_i64[0];
        v28 = (_QWORD *)qword_4F18BE0;
        if ( !dword_4F06978 )
          goto LABEL_81;
        v43 = *(char **)(v27[10].m128i_i64[1] + 56);
        if ( !v43 )
          goto LABEL_81;
        v44 = *v43;
        if ( (*v43 & 2) != 0 )
          goto LABEL_81;
        if ( (v44 & 1) != 0 )
        {
          v45 = *((_QWORD *)v43 + 1);
          if ( v45 )
          {
            if ( *(_BYTE *)(v45 + 173) == 12 )
            {
              *(_QWORD *)a3 += 2LL;
              v65 = (const __m128i **)v43;
              sub_8238B0(v28, "DO", 2);
              sub_80D8A0(v65[1], 0, 0, (_QWORD *)a3);
              v28 = (_QWORD *)qword_4F18BE0;
              ++*(_QWORD *)a3;
              if ( (unsigned __int64)(v28[2] + 1LL) > v28[1] )
              {
                sub_823810(v28);
                v28 = (_QWORD *)qword_4F18BE0;
              }
              *(_BYTE *)(v28[4] + v28[2]++) = 69;
              goto LABEL_81;
            }
          }
        }
        v28 = (_QWORD *)qword_4F18BE0;
        if ( (v44 & 4) != 0 )
          goto LABEL_81;
        if ( (unsigned int)sub_8D7650(v43) )
        {
          *(_QWORD *)a3 += 2LL;
          sub_8238B0(qword_4F18BE0, "Do", 2);
          v28 = (_QWORD *)qword_4F18BE0;
LABEL_81:
          ++*(_QWORD *)a3;
          v29 = v28[2];
          if ( (unsigned __int64)(v29 + 1) > v28[1] )
          {
            sub_823810(v28);
            v28 = (_QWORD *)qword_4F18BE0;
            v29 = *(_QWORD *)(qword_4F18BE0 + 16);
          }
          *(_BYTE *)(v28[4] + v29) = 70;
          ++v28[2];
          if ( unk_4F06904 && (*(_BYTE *)(v27[10].m128i_i64[1] + 17) & 0x70) == 0x30 )
          {
            ++*(_QWORD *)a3;
            v51 = v28[2];
            if ( (unsigned __int64)(v51 + 1) > v28[1] )
            {
              sub_823810(v28);
              v28 = (_QWORD *)qword_4F18BE0;
              v51 = *(_QWORD *)(qword_4F18BE0 + 16);
            }
            *(_BYTE *)(v28[4] + v51) = 89;
            ++v28[2];
          }
          sub_80F5E0(v27[10].m128i_i64[0], 0, a3);
          sub_80FC70(v27[10].m128i_i64[1], a3);
          if ( (*(_BYTE *)(v27[10].m128i_i64[1] + 19) & 0xC0) == 0x40 )
          {
            v30 = "R";
          }
          else
          {
            v30 = "O";
            if ( (*(_BYTE *)(v27[10].m128i_i64[1] + 19) & 0xC0) != 0x80 )
              goto LABEL_87;
          }
          ++*(_QWORD *)a3;
          sub_8238B0(qword_4F18BE0, v30, 1);
LABEL_87:
          v31 = (_QWORD *)qword_4F18BE0;
          ++*(_QWORD *)a3;
          result = v31[2];
          if ( (unsigned __int64)(result + 1) <= v31[1] )
          {
LABEL_88:
            *(_BYTE *)(v31[4] + result) = 69;
            ++v31[2];
            goto LABEL_31;
          }
LABEL_189:
          sub_823810(v31);
          v31 = (_QWORD *)qword_4F18BE0;
          result = *(_QWORD *)(qword_4F18BE0 + 16);
          goto LABEL_88;
        }
        goto LABEL_117;
      case 8:
        v18 = 1;
        v19 = "A";
        goto LABEL_57;
      case 9:
      case 10:
      case 11:
        goto LABEL_30;
      case 12:
        v32 = v3[11].m128i_i8[8];
        switch ( v32 )
        {
          case 1:
            v41 = sub_746BE0((__int64)v3);
            if ( (v3[11].m128i_i8[10] & 2) != 0 )
            {
              *(_QWORD *)a3 += 2LL;
              sub_8238B0(qword_4F18BE0, "Dt", 2);
            }
            else
            {
              *(_QWORD *)a3 += 2LL;
              sub_8238B0(qword_4F18BE0, "DT", 2);
            }
            if ( v41 )
            {
              sub_816460(v41, 1, 0, a3);
              v31 = (_QWORD *)qword_4F18BE0;
            }
            else
            {
              v31 = (_QWORD *)qword_4F18BE0;
              ++*(_QWORD *)a3;
              if ( (unsigned __int64)(v31[2] + 1LL) > v31[1] )
              {
                sub_823810(v31);
                v31 = (_QWORD *)qword_4F18BE0;
              }
              *(_BYTE *)(v31[4] + v31[2]++) = 63;
            }
            ++*(_QWORD *)a3;
            result = v31[2];
            if ( (unsigned __int64)(result + 1) <= v31[1] )
              goto LABEL_88;
            goto LABEL_189;
          case 5:
            *(_QWORD *)a3 += 5LL;
            v54 = 5;
            v55 = "U3eut";
            break;
          case 3:
            *(_QWORD *)a3 += 2LL;
            result = sub_8238B0(qword_4F18BE0, "Da", 2);
            goto LABEL_31;
          case 2:
            *(_QWORD *)a3 += 2LL;
            result = sub_8238B0(qword_4F18BE0, "Dc", 2);
            goto LABEL_31;
          default:
            if ( (unsigned __int8)(v32 - 6) > 1u )
            {
              if ( (((unsigned __int8)(v32 - 8) > 2u) & ((v32 & 0xFB) != 0)
                                                      & ((unsigned __int8)v3[11].m128i_i8[10] >> 3)) == 0
                || (unsigned __int8)(v32 - 11) <= 1u )
              {
                v19 = 0;
                v18 = strlen(0);
                goto LABEL_57;
              }
              v58 = sub_746810(v32);
              v59 = (_QWORD *)qword_4F18BE0;
              ++*(_QWORD *)a3;
              v60 = (char *)v58;
              v61 = v59[2];
              if ( (unsigned __int64)(v61 + 1) > v59[1] )
              {
                sub_823810(v59);
                v59 = (_QWORD *)qword_4F18BE0;
                v61 = *(_QWORD *)(qword_4F18BE0 + 16);
              }
              *(_BYTE *)(v59[4] + v61) = 117;
              ++v59[2];
              sub_80BC40(v60, (_QWORD *)a3);
              v62 = (_QWORD *)qword_4F18BE0;
              ++*(_QWORD *)a3;
              if ( (unsigned __int64)(v62[2] + 1LL) > v62[1] )
              {
                sub_823810(v62);
                v62 = (_QWORD *)qword_4F18BE0;
              }
              *(_BYTE *)(v62[4] + v62[2]++) = 73;
              sub_80F5E0(v3[10].m128i_i64[0], 0, a3);
              v63 = (_QWORD *)qword_4F18BE0;
              ++*(_QWORD *)a3;
              if ( (unsigned __int64)(v63[2] + 1LL) > v63[1] )
              {
                sub_823810(v63);
                v63 = (_QWORD *)qword_4F18BE0;
              }
              result = v63[2];
              *(_BYTE *)(v63[4] + result) = 69;
              ++v63[2];
              goto LABEL_31;
            }
            if ( v32 != 7 )
            {
              v52 = sub_746BE0((__int64)v3);
              *(_QWORD *)a3 += 2LL;
              if ( v52 )
              {
                sub_8238B0(qword_4F18BE0, "DY", 2);
                v53 = sub_746BE0((__int64)v3);
                sub_816460(v53, 1, 0, a3);
              }
              else
              {
                sub_8238B0(qword_4F18BE0, "Dy", 2);
                sub_80BC40("?", (_QWORD *)a3);
              }
              goto LABEL_87;
            }
            *(_QWORD *)a3 += 2LL;
            v54 = 2;
            v55 = "Dy";
            break;
        }
        sub_8238B0(qword_4F18BE0, v55, v54);
        sub_80F5E0(v3[10].m128i_i64[0], 0, a3);
        goto LABEL_87;
      case 13:
        v18 = 1;
        v19 = "M";
        goto LABEL_57;
      case 14:
        if ( (unsigned int)sub_8D3EA0(v3) )
        {
          v19 = "Dc";
          v18 = 2;
          if ( *(_DWORD *)(v3[10].m128i_i64[1] + 24) != 2 )
            v19 = "Da";
          goto LABEL_57;
        }
        v40 = v3[10].m128i_i8[0];
        if ( v40 != 1 )
        {
          if ( v40 == 2 )
          {
            result = sub_80BC40("?", (_QWORD *)a3);
            goto LABEL_31;
          }
          if ( !v40 )
          {
            result = sub_812B60(v3[10].m128i_i64[1] + 24, 0, a3);
            goto LABEL_31;
          }
LABEL_117:
          sub_721090();
        }
        if ( !v3->m128i_i64[1] )
          sub_815F90(v3, a3);
        break;
      case 15:
        v33 = v3[11].m128i_i8[1];
        if ( v33 == 4 )
        {
          v56 = sub_8D4620(v3);
          v57 = (char *)sub_88A410(v3[10].m128i_i64[0], v56);
          result = sub_80BC40(v57, (_QWORD *)a3);
          goto LABEL_31;
        }
        if ( (unsigned __int8)(v33 - 2) > 1u )
        {
          v18 = 2;
          v19 = "Dv";
          if ( HIDWORD(qword_4F077B4) )
          {
            if ( !(_DWORD)qword_4F077B4 )
            {
              v18 = unk_4D04250 < 0xC350u ? 10LL : 2LL;
              if ( unk_4D04250 < 0xC350u )
                v19 = "U8__vector";
            }
          }
          goto LABEL_57;
        }
        for ( i = v3[10].m128i_i64[0]; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
          ;
        v64 = v3[11].m128i_u8[1];
        v49 = sub_8D4620(v3);
        v50 = (char *)sub_88A220(i, v49, v64);
        result = sub_80BC40(v50, (_QWORD *)a3);
        goto LABEL_31;
      case 16:
        v34 = v3[10].m128i_u8[8];
        if ( v34 == 1 )
        {
          v46 = (_QWORD *)qword_4F18BE0;
          ++*(_QWORD *)a3;
          v47 = v46[2];
          if ( (unsigned __int64)(v47 + 1) > v46[1] )
          {
            sub_823810(v46);
            v46 = (_QWORD *)qword_4F18BE0;
            v47 = *(_QWORD *)(qword_4F18BE0 + 16);
          }
          *(_BYTE *)(v46[4] + v47) = 117;
          ++v46[2];
        }
        v35 = (char *)sub_8099D0(v3[10].m128i_i64[0], v34);
        result = sub_80BC40(v35, (_QWORD *)a3);
        goto LABEL_31;
      case 17:
        v18 = 14;
        v19 = "u11__SVCount_t";
        goto LABEL_57;
      case 18:
        v18 = 8;
        v19 = "u6__mfp8";
        goto LABEL_57;
      case 19:
        v18 = 2;
        v19 = "Dn";
        goto LABEL_57;
      case 20:
        v18 = 13;
        v19 = "U10__metainfo";
        goto LABEL_57;
      default:
        goto LABEL_117;
    }
    goto LABEL_30;
  }
  if ( (unsigned __int8)(v7 - 9) <= 2u )
    goto LABEL_30;
  if ( v7 != 2 )
    goto LABEL_18;
LABEL_49:
  v16 = v3[10].m128i_i8[1];
  if ( (v16 & 8) != 0 )
  {
LABEL_30:
    result = sub_810650(v3, 0, a3);
    goto LABEL_31;
  }
  if ( (v16 & 0x40) != 0 )
  {
    v18 = 1;
    v19 = "w";
  }
  else if ( v16 < 0 )
  {
    v18 = 2;
    v19 = "Du";
  }
  else
  {
    v17 = v3[10].m128i_i8[2];
    if ( (v17 & 1) != 0 )
    {
      v18 = 2;
      v19 = "Ds";
    }
    else if ( (v17 & 2) != 0 )
    {
      v18 = 2;
      v19 = "Di";
    }
    else
    {
      if ( (v17 & 4) == 0 )
      {
        v18 = 1;
        switch ( v3[10].m128i_i8[0] )
        {
          case 0:
            v19 = (char *)"c";
            goto LABEL_57;
          case 1:
            v19 = "a";
            goto LABEL_57;
          case 2:
            v19 = (char *)&unk_3F7DA1E;
            goto LABEL_57;
          case 3:
            v19 = "s";
            goto LABEL_57;
          case 4:
            v19 = "t";
            goto LABEL_57;
          case 5:
            v19 = "i";
            goto LABEL_57;
          case 6:
            v19 = "j";
            goto LABEL_57;
          case 7:
            v19 = "l";
            goto LABEL_57;
          case 8:
            v19 = "m";
            goto LABEL_57;
          case 9:
            v19 = "x";
            goto LABEL_57;
          case 0xA:
            v19 = "y";
            goto LABEL_57;
          case 0xB:
            v19 = "n";
            goto LABEL_57;
          case 0xC:
            v19 = "o";
            goto LABEL_57;
          default:
            goto LABEL_117;
        }
      }
      v18 = 1;
      v19 = "b";
    }
  }
LABEL_57:
  *(_QWORD *)a3 += v18;
  sub_8238B0(qword_4F18BE0, v19, v18);
  result = v3[8].m128i_u8[12];
  if ( (_BYTE)result == 13 )
  {
    sub_80F5E0(v3[10].m128i_i64[0], 0, a3);
    result = sub_80F5E0(v3[10].m128i_i64[1], 0, a3);
    goto LABEL_31;
  }
  if ( (unsigned __int8)result > 0xDu )
  {
    if ( (_BYTE)result != 15 )
      goto LABEL_31;
    if ( !HIDWORD(qword_4F077B4) || (_DWORD)qword_4F077B4 || unk_4D04250 > 0xC34Fu )
    {
      v36 = sub_8D4620(v3);
      if ( v36 > 9 )
      {
        v37 = (int)sub_622470(v36, v67);
      }
      else
      {
        v37 = 1;
        LOWORD(v67[0]) = (unsigned __int8)(v36 + 48);
      }
      *(_QWORD *)a3 += v37;
      sub_8238B0(qword_4F18BE0, v67, v37);
      v38 = (_QWORD *)qword_4F18BE0;
      ++*(_QWORD *)a3;
      v39 = v38[2];
      if ( (unsigned __int64)(v39 + 1) > v38[1] )
      {
        sub_823810(v38);
        v38 = (_QWORD *)qword_4F18BE0;
        v39 = *(_QWORD *)(qword_4F18BE0 + 16);
      }
      *(_BYTE *)(v38[4] + v39) = 95;
      ++v38[2];
    }
LABEL_160:
    result = sub_80F5E0(v3[10].m128i_i64[0], 0, a3);
    goto LABEL_31;
  }
  if ( (_BYTE)result != 6 )
  {
    if ( (_BYTE)result == 8 )
    {
      v20 = *(_BYTE *)(a3 + 32);
      v21 = (const __m128i *)v3[11].m128i_i64[0];
      if ( v3[10].m128i_i8[8] < 0 )
      {
        if ( v21 )
        {
          if ( dword_4D0425C && unk_4D04250 <= 0x76BFu && v21[10].m128i_i8[13] == 12 )
            *(_BYTE *)(a3 + 32) = 1;
          sub_80D8A0(v21, 0, 0, (_QWORD *)a3);
        }
      }
      else
      {
        v22 = v3[10].m128i_i8[9];
        if ( (v22 & 0x20) != 0 || v21 )
        {
          if ( v20 )
          {
            sub_80BDC0((unsigned __int64)v21, (_QWORD *)a3);
          }
          else if ( (v22 & 1) != 0 )
          {
            sub_816460(v21, 1, 0, a3);
          }
          else
          {
            if ( (unsigned __int64)v21 > 9 )
            {
              v42 = (int)sub_622470((unsigned __int64)v21, v67);
            }
            else
            {
              v42 = 1;
              LOWORD(v67[0]) = (unsigned __int8)((_BYTE)v21 + 48);
            }
            *(_QWORD *)a3 += v42;
            sub_8238B0(qword_4F18BE0, v67, v42);
          }
        }
      }
      v23 = (_QWORD *)qword_4F18BE0;
      ++*(_QWORD *)a3;
      v24 = v23[2];
      if ( (unsigned __int64)(v24 + 1) > v23[1] )
      {
        sub_823810(v23);
        v23 = (_QWORD *)qword_4F18BE0;
        v24 = *(_QWORD *)(qword_4F18BE0 + 16);
      }
      *(_BYTE *)(v23[4] + v24) = 95;
      ++v23[2];
      result = sub_80F5E0(v3[10].m128i_i64[0], 0, a3);
      *(_BYTE *)(a3 + 32) = v20;
    }
    goto LABEL_31;
  }
  result = sub_7E1E50((__int64)v3);
  if ( !(_DWORD)result )
    goto LABEL_160;
LABEL_31:
  if ( !a2 )
  {
    result = sub_809440((__int64)v3);
    if ( !(_DWORD)result )
      goto LABEL_33;
    if ( !*(_QWORD *)(a3 + 40) )
    {
      result = sub_80A250((__int64)v3, 6, 0, a3);
      goto LABEL_33;
    }
  }
  return result;
}
