// Function: sub_860B80
// Address: 0x860b80
//
__int64 __fastcall sub_860B80(__int64 a1, unsigned __int8 a2, __int64 a3)
{
  __int64 v4; // rdi
  __int64 result; // rax
  __int64 **jj; // rbx
  __int64 v8; // r14
  unsigned __int8 v9; // bl
  int v10; // r9d
  bool v11; // al
  __int64 v12; // rdi
  char v13; // al
  _QWORD *v14; // rax
  __int64 v15; // rax
  char i; // dl
  __int64 v17; // r8
  _QWORD *v18; // rcx
  char v19; // al
  unsigned int v20; // esi
  __int64 v21; // rbx
  char v22; // r13
  __int64 v23; // rbx
  const __m128i *v24; // r13
  char m; // r14
  __int64 v26; // rbx
  char v27; // al
  __int64 n; // rbx
  _QWORD *ii; // r12
  __int64 v30; // rdi
  char v31; // al
  char v32; // dl
  __int64 v33; // rcx
  int v34; // edx
  __int64 v35; // rax
  __int64 v36; // rax
  __int64 v37; // rax
  __int64 j; // rax
  __int64 k; // rax
  __int64 v40; // rax
  __int64 v41; // rcx
  __int64 v42; // r8
  __int64 v43; // r9
  char v44; // al
  unsigned __int8 v45; // di
  char v46; // dl
  unsigned __int8 v47; // al
  int v48; // eax
  bool v49; // zf
  __m128i *v50; // r14
  __int64 v51; // rdi
  __int64 v52; // rax
  int v53; // eax
  char v54; // [rsp+13h] [rbp-3Dh] BYREF
  int v55; // [rsp+14h] [rbp-3Ch] BYREF
  _QWORD v56[7]; // [rsp+18h] [rbp-38h] BYREF

  v4 = *(unsigned __int8 *)(a1 + 80);
  result = (unsigned __int8)(v4 - 4);
  switch ( (char)v4 )
  {
    case 4:
    case 5:
      if ( a2 == 3 || !a2 )
        return sub_860790(v4, *(_QWORD **)(a1 + 88));
      return result;
    case 7:
      v8 = *(_QWORD *)(a1 + 88);
      v9 = *(_BYTE *)(v8 + 136);
      v10 = sub_736990(v8);
      v11 = 0;
      if ( v10 && (*(_BYTE *)(a1 + 81) & 0x10) == 0 )
        v11 = *(_QWORD *)(a1 + 64) != 0;
      if ( !a2 && (*(_QWORD *)(v8 + 168) & 0x2000001000LL) == 0x2000001000LL )
      {
        if ( v9 > 1u )
        {
LABEL_16:
          if ( !v11 )
          {
            v12 = *(_QWORD *)(v8 + 120);
            if ( (*(_BYTE *)(v12 + 140) & 0xFB) == 8 && (sub_8D4C10(v12, dword_4F077C4 != 2) & 1) != 0 )
            {
              if ( (*(_BYTE *)(a1 + 81) & 1) != 0 )
                goto LABEL_114;
              if ( dword_4F077C4 != 2 || dword_4F04C64 )
              {
                if ( *(char *)(v8 + 169) >= 0 )
                {
                  if ( *(char *)(v8 + 171) >= 0 )
                  {
LABEL_23:
                    if ( (*(_BYTE *)(v8 + 91) & 4) == 0 )
                    {
                      if ( (*(_BYTE *)(v8 + 170) & 1) != 0 )
                      {
                        v14 = *(_QWORD **)(*(_QWORD *)(v8 + 128) + 128LL);
                        if ( v8 != v14[2] )
                          goto LABEL_114;
                        while ( 1 )
                        {
                          v14 = (_QWORD *)*v14;
                          if ( !v14 )
                            break;
                          if ( (*(_BYTE *)(*(_QWORD *)v14[2] + 81LL) & 1) != 0 )
                            goto LABEL_114;
                        }
                      }
                      if ( (*(_BYTE *)(v8 + 168) & 0x50) == 0 && !*(_QWORD *)(v8 + 224) )
                      {
                        v15 = sub_8D4130(*(_QWORD *)(v8 + 120));
                        for ( i = *(_BYTE *)(v15 + 140); i == 12; i = *(_BYTE *)(v15 + 140) )
                          v15 = *(_QWORD *)(v15 + 160);
                        if ( i != 14
                          && ((unsigned __int8)(i - 9) > 2u || (*(_BYTE *)(v15 + 177) & 0x20) == 0)
                          && (*(_BYTE *)(v15 + 143) & 1) == 0 )
                        {
                          if ( i )
                          {
                            sub_72F9F0(v8, *(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 184), &v54, v56);
                            if ( !dword_4D047B0
                              || unk_4F04C48 == -1
                              || (*(_BYTE *)(qword_4F04C68[0] + 776LL * unk_4F04C48 + 10) & 1) == 0
                              || (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) & 6) != 0 )
                            {
                              if ( dword_4F04C44 == -1
                                && (v18 = qword_4F04C68,
                                    (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) & 2) == 0)
                                || !(unsigned int)sub_8809D0(a1) )
                              {
                                if ( *(char *)(v8 + 175) < 0 )
                                  goto LABEL_243;
                                goto LABEL_45;
                              }
                              if ( (*(_BYTE *)(a1 + 81) & 1) == 0 )
                              {
                                if ( *(char *)(v8 + 175) < 0 )
                                {
                                  v19 = 4;
                                  v20 = 177;
LABEL_54:
                                  if ( (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 9) & 0x40) == 0 )
                                  {
                                    sub_85B2D0(a1, v20, v19);
                                    goto LABEL_118;
                                  }
                                  goto LABEL_114;
                                }
LABEL_45:
                                if ( v54 != 2 )
                                {
                                  if ( unk_4F072F5
                                    && (!dword_4D047B0 || dword_4D047AC)
                                    && !unk_4D04734
                                    && a2 <= 6u
                                    && ((0x49uLL >> a2) & 1) != 0 )
                                  {
                                    v19 = 3;
LABEL_53:
                                    v20 = (*(_BYTE *)(a1 + 81) & 1) == 0 ? 177 : 550;
                                    goto LABEL_54;
                                  }
LABEL_248:
                                  v19 = 5;
                                  goto LABEL_53;
                                }
                                v51 = *(_QWORD *)v56[0];
                                v52 = *(_QWORD *)(*(_QWORD *)v56[0] + 8LL);
                                if ( v52 && (*(_BYTE *)(v52 + 173) & 4) != 0 )
                                  goto LABEL_248;
                                v19 = 3;
                                if ( *(_BYTE *)(v51 + 48) == 5 || *(_QWORD *)(v51 + 16) )
                                  goto LABEL_53;
                                v53 = sub_731890(v51, 1, &v55, (__int64)v18, v17, (__int64)&dword_4F077C4);
                                if ( !(v55 | v53) )
                                  goto LABEL_248;
LABEL_243:
                                v19 = 4;
                                goto LABEL_53;
                              }
                            }
                          }
                        }
                      }
                    }
LABEL_114:
                    if ( dword_4D04438 && v9 == 1 && (*(_BYTE *)(a1 + 81) & 1) != 0 )
                      sub_649830(a1, a1 + 48, 0);
                    goto LABEL_118;
                  }
LABEL_173:
                  sub_85B2D0(a1, 0xB1u, 4);
                  goto LABEL_118;
                }
                goto LABEL_194;
              }
              if ( sub_729F20(*(_DWORD *)(a1 + 48)) )
                goto LABEL_114;
            }
            v13 = *(_BYTE *)(a1 + 81) & 1;
            if ( *(char *)(v8 + 169) >= 0 )
            {
              if ( *(char *)(v8 + 171) >= 0 )
              {
                if ( v13 && (*(char *)(a1 + 84) >= 0 || (*(_BYTE *)(v8 + 169) & 0x10) != 0) )
                  goto LABEL_114;
                goto LABEL_23;
              }
              if ( !v13 )
                goto LABEL_173;
LABEL_178:
              if ( (*(_DWORD *)(v8 + 168) & 0x10001000) == 0x10000000 )
              {
                sub_85B2D0(a1, 0x226u, 5);
LABEL_118:
                result = *(_QWORD *)v8;
                if ( a1 != *(_QWORD *)v8 )
                {
                  v32 = *(_BYTE *)(a1 + 84);
                  if ( (v32 & 0x20) == 0 )
                  {
                    if ( result )
                    {
                      if ( (*(_BYTE *)(a1 + 81) & 1) != 0 )
                      {
                        *(_BYTE *)(result + 81) |= 1u;
                        v32 = *(_BYTE *)(a1 + 84);
                      }
                      if ( v32 < 0 )
                        *(_BYTE *)(result + 84) |= 0x80u;
                    }
                  }
                }
                if ( dword_4F077C4 != 2 )
                {
                  result = (__int64)&dword_4F04C64;
                  if ( !dword_4F04C64 && (*(_BYTE *)(a1 + 81) & 2) != 0 )
                  {
                    result = (__int64)dword_4D03FE8;
                    if ( dword_4D03FE8[0] )
                    {
                      result = *(_QWORD *)(v8 + 96);
                      if ( result )
                      {
                        if ( *(_BYTE *)(result + 16) == 53 )
                        {
                          v33 = *(_QWORD *)(result + 24);
                          *(_BYTE *)(result + 16) = 7;
                          *(_QWORD *)(result + 24) = v8;
                          *(_QWORD *)(v8 + 256) = *(_QWORD *)(v33 + 32);
                          *(_BYTE *)(v8 + 137) = *(_BYTE *)(v33 + 56);
                          *(_BYTE *)(v8 + 175) = (4 * *(_BYTE *)(v33 + 57)) & 8 | *(_BYTE *)(v8 + 175) & 0xF7;
                          v34 = *(_BYTE *)(v33 + 57) >> 7;
                          result = v34 | *(_BYTE *)(v8 + 90) & 0xFEu;
                          *(_BYTE *)(v8 + 90) = v34 | *(_BYTE *)(v8 + 90) & 0xFE;
                        }
                      }
                    }
                  }
                }
                return result;
              }
              goto LABEL_114;
            }
            if ( v13 )
              goto LABEL_178;
LABEL_194:
            for ( j = *(_QWORD *)(a3 + 152); *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
              ;
            if ( (*(_BYTE *)(*(_QWORD *)(j + 168) + 16LL) & 0x10) == 0
              && unk_4F07290 != a3
              && (*(_BYTE *)(v8 + 91) & 4) == 0 )
            {
              for ( k = *(_QWORD *)(v8 + 120); *(_BYTE *)(k + 140) == 12; k = *(_QWORD *)(k + 160) )
                ;
              if ( (*(_BYTE *)(k + 143) & 1) == 0 )
              {
                v40 = *(_QWORD *)(v8 + 128);
                if ( (!v40 || (*(_BYTE *)(v40 + 33) & 2) == 0 || (*(_BYTE *)(a1 + 83) & 0x40) == 0)
                  && (unk_4F04C48 == -1
                   || (*(_BYTE *)(qword_4F04C68[0] + 776LL * unk_4F04C48 + 10) & 1) == 0
                   || (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) & 6) != 0) )
                {
                  sub_85B2D0(a1, 0x33Au, 4);
                }
              }
            }
            goto LABEL_118;
          }
LABEL_110:
          v31 = *(_BYTE *)(a1 + 81);
          if ( (v31 & 1) != 0 || (*(_BYTE *)(v8 + 91) & 4) != 0 )
          {
            if ( ((*(_BYTE *)(v8 + 169) & 0x10) != 0 || *(char *)(a1 + 84) < 0)
              && (v31 & 2) == 0
              && (*(_BYTE *)(v8 + 156) & 2) == 0 )
            {
              sub_6853B0(HIDWORD(qword_4F077B4) == 0 ? 7 : 5, 0x72u, (FILE *)(a1 + 48), a1);
            }
          }
          else
          {
            sub_72F9F0(v8, *(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 184), &v54, v56);
            if ( v54 != 2
              || (v48 = sub_731890(*(_QWORD *)v56[0], 1, &v55, v41, v42, v43), v49 = (v55 | v48) == 0, v44 = 4, v49) )
            {
              v44 = 5;
            }
            if ( (*(_BYTE *)(v8 + 156) & 2) == 0 )
              sub_85B2D0(a1, 0xB1u, v44);
          }
          goto LABEL_114;
        }
        if ( (*(_BYTE *)(a1 + 81) & 2) == 0 )
        {
          sub_6854C0(0x335u, (FILE *)(a1 + 48), a1);
          goto LABEL_114;
        }
      }
      else if ( v9 > 1u )
      {
        goto LABEL_16;
      }
      if ( !v11 || (*(_BYTE *)(v8 + 88) & 0x70) == 0x30 )
        goto LABEL_114;
      goto LABEL_110;
    case 11:
      v21 = *(_QWORD *)(a1 + 88);
      v22 = *(_BYTE *)(v21 + 172);
      if ( dword_4F077C4 != 2 && unk_4F07778 > 199900 && dword_4D04964 && v22 == 1 )
      {
        if ( *(char *)(v21 + 192) < 0 )
        {
          if ( a2 == 3 || !a2 )
          {
            v45 = 7;
            if ( (*(_BYTE *)(v21 + 88) & 4) == 0 )
              v45 = unk_4F07471;
            sub_6853B0(v45, 0x41Cu, (FILE *)(a1 + 48), a1);
          }
          goto LABEL_65;
        }
        LOBYTE(result) = *(_BYTE *)(v21 + 88);
        goto LABEL_62;
      }
      if ( v22 )
      {
        if ( (*(_BYTE *)(v21 + 88) & 4) != 0 )
        {
          if ( v22 == 2 )
          {
LABEL_106:
            if ( !sub_860410(v21) )
            {
              v36 = *(_QWORD *)(a1 + 96);
              if ( v36 && (*(_BYTE *)(v36 + 80) & 8) == 0 )
                sub_6853B0(HIDWORD(qword_4F077B4) == 0 ? 7 : 5, 0x72u, (FILE *)(a1 + 48), a1);
              goto LABEL_65;
            }
LABEL_64:
            if ( dword_4F077C4 == 2 && *(char *)(v21 + 192) < 0 && !sub_860410(v21) && (a2 == 3 || !a2) )
              sub_6853B0(HIDWORD(qword_4F077B4) == 0 ? 8 : 5, 0x335u, (FILE *)(a1 + 48), a1);
LABEL_65:
            result = (__int64)&dword_4D04438;
            if ( dword_4D04438 && v22 == 1 && (*(_BYTE *)(a1 + 81) & 1) != 0 )
            {
              if ( (*(_BYTE *)(v21 + 195) & 1) == 0 )
                return (__int64)sub_649830(a1, a1 + 48, 0);
              if ( !a2 || a2 == 3 )
              {
                result = sub_899F90(a1);
                if ( !(_DWORD)result )
                  return (__int64)sub_649830(a1, a1 + 48, 0);
              }
            }
            return result;
          }
LABEL_63:
          if ( !(unsigned int)sub_736990(v21) )
            goto LABEL_64;
          goto LABEL_106;
        }
LABEL_132:
        if ( (*(_BYTE *)(a1 + 81) & 1) == 0
          && (v22 != 1 || (unsigned int)sub_736990(v21))
          && (*(char *)(v21 + 192) >= 0 || (*(_BYTE *)(a1 + 81) & 2) == 0 || !sub_729F20(*(_DWORD *)(a1 + 48)))
          && (*(_BYTE *)(v21 + 91) & 4) == 0
          && (*(_WORD *)(v21 + 200) & 0x240) == 0
          && (!unk_4F072F5
           || dword_4D047B0 && !dword_4D047AC && (*(_BYTE *)(*(_QWORD *)a1 + 73LL) & 8) == 0
           || unk_4D04734
           || a2 > 6u
           || ((0x49uLL >> a2) & 1) == 0)
          && ((*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) & 2) == 0 || a2 != 17 && a2 != 2)
          && (*(_BYTE *)(v21 + 206) & 0x10) == 0 )
        {
          sub_85B2D0(a1, 0xB1u, 5);
        }
        goto LABEL_65;
      }
      result = sub_736990(*(_QWORD *)(a1 + 88));
      if ( (_DWORD)result )
      {
        result = *(unsigned __int8 *)(v21 + 88);
        if ( (*(_BYTE *)(v21 + 88) & 0x70) != 0x30 )
        {
LABEL_62:
          if ( (result & 4) != 0 )
            goto LABEL_63;
          goto LABEL_132;
        }
      }
      if ( *(char *)(v21 + 192) >= 0 )
      {
        *(_BYTE *)(v21 + 88) |= 4u;
        *(_BYTE *)(a1 + 81) |= 1u;
      }
      return result;
    case 12:
      result = *(_QWORD *)(a1 + 88);
      if ( !*(_QWORD *)(result + 128) )
        return sub_6854C0(0x72u, (FILE *)(a1 + 48), a1);
      if ( (*(_BYTE *)(a1 + 81) & 1) == 0 && (*(_BYTE *)(result + 91) & 4) == 0 )
        return sub_85B2D0(a1, 0xB1u, 5);
      return result;
    case 14:
      v23 = *(_QWORD *)(*(_QWORD *)(a1 + 88) + 8LL);
      v24 = *(const __m128i **)(v23 + 120);
      for ( m = *(_BYTE *)(v23 + 136); v24[8].m128i_i8[12] == 12; v24 = (const __m128i *)v24[10].m128i_i64[0] )
        ;
      result = sub_8D23B0(v24);
      if ( !(_DWORD)result )
        goto LABEL_80;
      result = sub_8D2600(v24);
      if ( (_DWORD)result )
        goto LABEL_80;
      if ( m )
      {
        result = (__int64)&dword_4F077C4;
        if ( dword_4F077C4 == 2 || m != 2 )
          goto LABEL_156;
      }
      if ( !(unsigned int)sub_8D3410(v24) )
        goto LABEL_157;
      v37 = sub_8D4050(v24);
      result = sub_8D23B0(v37);
      if ( (_DWORD)result )
      {
LABEL_156:
        if ( m != 1 )
        {
LABEL_157:
          if ( !(unsigned int)sub_8D3410(v24)
            || (v35 = sub_8D4050(v24), (unsigned int)sub_8D23B0(v35))
            || (result = sub_8D23B0(v24), !(_DWORD)result)
            || (*(_BYTE *)(v23 + 156) & 2) == 0 )
          {
            result = sub_6851A0(0xEBu, (_DWORD *)(a1 + 48), *(_QWORD *)(*(_QWORD *)a1 + 8LL));
          }
        }
      }
      else if ( (unsigned int)sub_8D23B0(**(_QWORD **)(a1 + 88)) )
      {
        result = (__int64)&dword_4F077C4;
        if ( dword_4F077C4 == 1 && m != 2 )
        {
          *(_BYTE *)(v23 + 136) = 1;
          return result;
        }
        v50 = (__m128i *)sub_7259C0(8);
        sub_73C230(v24, v50);
        v50[11].m128i_i64[0] = 1;
        sub_8D6090(v50);
        *(_QWORD *)(v23 + 120) = v50;
        result = sub_685460(0x692u, (FILE *)(a1 + 48), a1);
      }
      else
      {
        result = **(_QWORD **)(a1 + 88);
        *(_QWORD *)(v23 + 120) = result;
      }
LABEL_80:
      if ( !*(_BYTE *)(v23 + 136) )
        *(_BYTE *)(v23 + 88) |= 4u;
      return result;
    case 15:
      result = *(_QWORD *)(a1 + 88);
      v26 = *(_QWORD *)(result + 8);
      if ( (*(_BYTE *)(v26 + 88) & 4) == 0 )
        return result;
      if ( *(_BYTE *)(v26 + 172) != 2 )
      {
        result = sub_736990(*(_QWORD *)(result + 8));
        if ( !(_DWORD)result )
          return result;
        result = *(_BYTE *)(v26 + 88) & 0x70;
        if ( (_BYTE)result == 48 )
          return result;
      }
      result = sub_860410(v26);
      if ( (_DWORD)result )
        return result;
      if ( (*(_BYTE *)(v26 + 200) & 0x40) != 0 )
        return result;
      result = *(_QWORD *)(v26 + 256);
      if ( result )
      {
        if ( *(_QWORD *)(result + 8) )
          return result;
      }
      result = dword_4F077C0;
      if ( dword_4F077C4 == 1 )
      {
        v46 = *(_BYTE *)(v26 + 88);
        *(_BYTE *)(v26 + 172) = 1;
        *(_BYTE *)(v26 + 88) = v46 & 0x8F | 0x30;
        if ( !(_DWORD)result )
          return result;
        return sub_685490(0x73Bu, (FILE *)(a1 + 48), a1);
      }
      if ( dword_4F077C0 )
      {
        v27 = *(_BYTE *)(v26 + 88);
        *(_BYTE *)(v26 + 172) = 1;
        *(_BYTE *)(v26 + 88) = v27 & 0x8F | 0x30;
        return sub_685490(0x73Bu, (FILE *)(a1 + 48), a1);
      }
      v47 = 7;
      if ( dword_4F077BC && qword_4F077A8 > 0x9DCFu )
        v47 = (*(_BYTE *)(v26 + 198) & 0x10) == 0 ? 5 : 7;
      return sub_6853B0(v47, 0x72u, (FILE *)(a1 + 48), a1);
    case 17:
      for ( n = *(_QWORD *)(a1 + 88); n; n = *(_QWORD *)(n + 8) )
        result = sub_860B80(n, a2, a3);
      return result;
    case 19:
      result = *(_QWORD *)(a1 + 88);
      if ( (*(_BYTE *)(result + 265) & 1) == 0 )
      {
        for ( ii = *(_QWORD **)(result + 168); ii; ii = (_QWORD *)*ii )
        {
          v30 = ii[1];
          result = *(_QWORD *)(v30 + 88);
          if ( (*(_BYTE *)(result + 177) & 0x20) == 0 )
            result = sub_860B80(v30, a2, a3);
        }
      }
      return result;
    case 20:
      result = *(_QWORD *)(a1 + 88);
      for ( jj = *(__int64 ***)(result + 168); jj; jj = (__int64 **)*jj )
      {
        if ( ((_BYTE)jj[10] & 4) == 0 )
          result = sub_860B80(jj[3], a2, a3);
      }
      return result;
    default:
      return (unsigned int)(v4 - 4);
  }
}
