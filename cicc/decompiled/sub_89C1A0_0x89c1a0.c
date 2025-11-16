// Function: sub_89C1A0
// Address: 0x89c1a0
//
__int64 *__fastcall sub_89C1A0(__int64 a1, __int64 a2, const __m128i *a3, __int64 *a4, FILE *a5)
{
  __int64 v5; // r15
  __int64 v7; // rbx
  __int64 **v8; // rax
  unsigned __int64 v9; // rax
  __int64 v10; // rdx
  int v11; // r12d
  __int64 v12; // r12
  __int64 *v14; // rax
  __int64 v15; // r14
  char v16; // al
  unsigned int v17; // r8d
  __int64 v18; // rcx
  __int64 v19; // rax
  __int64 v20; // r14
  char v21; // cl
  int v22; // eax
  __int64 v23; // rax
  char v24; // dl
  char v25; // al
  char v26; // al
  __int64 v27; // rax
  int v28; // eax
  __int64 *v29; // rax
  __int64 v30; // rax
  char v31; // al
  __int64 v32; // rdx
  __int64 v33; // rax
  __int64 v34; // rax
  _BYTE *v35; // rax
  __int64 v36; // rdi
  __int64 v37; // rax
  __int64 v38; // rdx
  __int64 v39; // rcx
  __int64 v40; // r8
  __int64 v41; // r9
  __int64 v42; // rdx
  __int64 v43; // rcx
  __int64 v44; // r8
  __int64 v45; // r9
  int v46; // r13d
  char v47; // r12
  __int64 *v48; // rbx
  char v49; // al
  __int64 v50; // r15
  _QWORD *v51; // r13
  int v52; // eax
  char v53; // al
  __int64 v54; // rdi
  __int64 v55; // rsi
  __int64 v56; // rax
  __int64 *v57; // rax
  __int64 v58; // rdx
  __int64 v59; // rax
  __int64 i; // rax
  int v61; // eax
  bool v62; // dl
  __int64 v63; // rdx
  __int64 flags; // rax
  __int64 v65; // rax
  __int64 *v66; // rbx
  int v67; // r14d
  char v68; // cl
  int v69; // r13d
  __int64 *v70; // rdx
  char v71; // al
  char v72; // r15
  __int64 v73; // rax
  _QWORD *v74; // rax
  char v75; // si
  __int64 v76; // rax
  char v77; // cl
  char v78; // al
  _QWORD *v79; // rsi
  __int64 v80; // rcx
  char v81; // cl
  unsigned int v82; // [rsp+8h] [rbp-98h]
  __int64 v83; // [rsp+8h] [rbp-98h]
  int v84; // [rsp+10h] [rbp-90h]
  char v85; // [rsp+10h] [rbp-90h]
  __int64 v86; // [rsp+10h] [rbp-90h]
  int v87; // [rsp+18h] [rbp-88h]
  __int64 v88; // [rsp+18h] [rbp-88h]
  __int64 v89; // [rsp+18h] [rbp-88h]
  __int64 v90; // [rsp+18h] [rbp-88h]
  int v91; // [rsp+20h] [rbp-80h]
  __int64 v92; // [rsp+20h] [rbp-80h]
  __int64 v93; // [rsp+20h] [rbp-80h]
  __int64 v94; // [rsp+20h] [rbp-80h]
  _BYTE *v95; // [rsp+20h] [rbp-80h]
  __int64 *v96; // [rsp+28h] [rbp-78h]
  int v97; // [rsp+30h] [rbp-70h]
  __int64 v98; // [rsp+30h] [rbp-70h]
  __int64 v99; // [rsp+30h] [rbp-70h]
  __int64 v100; // [rsp+30h] [rbp-70h]
  _BYTE *v101; // [rsp+30h] [rbp-70h]
  _BOOL4 v104; // [rsp+48h] [rbp-58h]
  int v105; // [rsp+4Ch] [rbp-54h]
  unsigned __int8 v106; // [rsp+4Ch] [rbp-54h]
  __m128i v107[5]; // [rsp+50h] [rbp-50h] BYREF

  v5 = a2;
  v7 = a1;
  v8 = *(__int64 ***)(a1 + 192);
  if ( !a2 )
  {
    v14 = *v8;
    v104 = 0;
    v15 = 0;
    v12 = 0;
    *(_DWORD *)(a1 + 52) = 1;
    v96 = v14;
    LOBYTE(v91) = 0;
    goto LABEL_12;
  }
  v96 = *v8;
  if ( (*(_BYTE *)(a2 + 81) & 0x20) != 0 )
  {
    v105 = 1;
    *(_DWORD *)(a1 + 52) = 1;
    LOBYTE(v9) = *(_BYTE *)(a2 + 80);
  }
  else
  {
    v9 = *(unsigned __int8 *)(a2 + 80);
    if ( (unsigned __int8)v9 > 0x14u || (v10 = 1182720, v105 = 0, !_bittest64(&v10, v9)) )
    {
      sub_6854C0(0x313u, a5, a2);
      *(_DWORD *)(a1 + 52) = 1;
      if ( !(_DWORD)qword_4D04464 )
      {
        sub_7ADF70((__int64)v107, 0);
        sub_88F140(a1, (unsigned __int64)v107, 1u, a5);
        sub_7AEA70(v107);
        v11 = *(_DWORD *)(a1 + 36);
        *(_DWORD *)(a1 + 52) = 1;
        if ( !v11 )
        {
          v12 = 0;
          goto LABEL_10;
        }
        v12 = 0;
        **(_WORD **)(a1 + 184) = 74;
        goto LABEL_48;
      }
      v104 = 0;
      v5 = 0;
      v15 = 0;
      v12 = 0;
      LOBYTE(v91) = 0;
      v105 = 1;
      goto LABEL_77;
    }
  }
  switch ( (char)v9 )
  {
    case 4:
    case 5:
      v12 = *(_QWORD *)(*(_QWORD *)(a2 + 96) + 80LL);
      break;
    case 6:
      v12 = *(_QWORD *)(*(_QWORD *)(a2 + 96) + 32LL);
      break;
    case 9:
    case 10:
      v12 = *(_QWORD *)(*(_QWORD *)(a2 + 96) + 56LL);
      break;
    case 19:
    case 20:
    case 21:
    case 22:
      v12 = *(_QWORD *)(a2 + 88);
      break;
    default:
      BUG();
  }
  v15 = *(_QWORD *)(v12 + 176);
  v104 = *(_QWORD *)(v12 + 32) == 0;
  v23 = sub_892240(*(_QWORD *)v15);
  if ( !*(_QWORD *)(v23 + 64) )
    *(_QWORD *)(v23 + 64) = *(_QWORD *)(a1 + 440);
  if ( *(_BYTE *)(a2 + 80) != 20 )
    goto LABEL_65;
  v28 = *(_DWORD *)(a1 + 16);
  if ( (*(_BYTE *)(a2 + 81) & 0x10) == 0 )
  {
    if ( !v28 )
      goto LABEL_65;
    goto LABEL_109;
  }
  if ( v28 )
  {
LABEL_109:
    v35 = *(_BYTE **)(a1 + 240);
    if ( v35 )
    {
      v36 = *(_QWORD *)v35;
      if ( (unsigned __int8)(*(_BYTE *)(*(_QWORD *)v35 + 80LL) - 4) <= 1u
        && *(char *)(*(_QWORD *)(v36 + 88) + 177LL) < 0 )
      {
        if ( (v35[178] & 4) == 0 )
        {
          v99 = *(_QWORD *)(*(_QWORD *)(v36 + 96) + 80LL);
          v73 = sub_878FF0();
          *(_QWORD *)(v73 + 8) = a2;
          *(_DWORD *)(v73 + 16) = dword_4F06650[0];
          *(_QWORD *)v73 = *(_QWORD *)(v99 + 192);
          *(_QWORD *)(v99 + 192) = v73;
        }
      }
      else
      {
        v37 = sub_8788F0(v36);
        if ( v37 )
        {
          switch ( *(_BYTE *)(v37 + 80) )
          {
            case 4:
            case 5:
              v56 = *(_QWORD *)(*(_QWORD *)(v37 + 96) + 80LL);
              break;
            case 6:
              v56 = *(_QWORD *)(*(_QWORD *)(v37 + 96) + 32LL);
              break;
            case 9:
            case 0xA:
              v56 = *(_QWORD *)(*(_QWORD *)(v37 + 96) + 56LL);
              break;
            case 0x13:
            case 0x14:
            case 0x15:
            case 0x16:
              v56 = *(_QWORD *)(v37 + 88);
              break;
            default:
              BUG();
          }
          v57 = *(__int64 **)(v56 + 192);
          if ( v57 )
          {
            while ( *((_DWORD *)v57 + 4) != dword_4F06650[0] )
            {
              v57 = (__int64 *)*v57;
              if ( !v57 )
                goto LABEL_65;
            }
            v58 = v57[1];
            if ( v58 )
            {
              *(_QWORD *)(v12 + 400) = v58;
              switch ( *(_BYTE *)(v58 + 80) )
              {
                case 4:
                case 5:
                  v59 = *(_QWORD *)(*(_QWORD *)(v58 + 96) + 80LL);
                  break;
                case 6:
                  v59 = *(_QWORD *)(*(_QWORD *)(v58 + 96) + 32LL);
                  break;
                case 9:
                case 0xA:
                  v59 = *(_QWORD *)(*(_QWORD *)(v58 + 96) + 56LL);
                  break;
                case 0x13:
                case 0x14:
                case 0x15:
                case 0x16:
                  v59 = *(_QWORD *)(v58 + 88);
                  break;
                default:
                  BUG();
              }
              *(_DWORD *)(*(_QWORD *)(v7 + 192) + 44LL) = *(_DWORD *)(*(_QWORD *)(v59 + 328) + 44LL);
            }
          }
        }
      }
    }
    goto LABEL_65;
  }
  if ( (*(_BYTE *)(v15 + 194) & 0x40) == 0 && *(_BYTE *)(v15 + 174) != 7 )
  {
    if ( *(_DWORD *)(a1 + 44) || (v97 = *(_DWORD *)(a1 + 48)) != 0 )
    {
      *(_DWORD *)(v12 + 64) = *(_DWORD *)(a1 + 156);
      goto LABEL_65;
    }
    v29 = *(__int64 **)(a1 + 240);
    if ( v29 )
    {
      v30 = sub_8788F0(*v29);
      if ( v30 )
      {
        if ( !*(_QWORD *)(*(_QWORD *)(a2 + 88) + 88LL) )
        {
          v88 = *(_QWORD *)(a2 + 88);
          v92 = v30;
          v31 = sub_877F80(a2);
          v32 = v88;
          if ( v31 == 1 )
          {
            v33 = *(_QWORD *)(*(_QWORD *)(v92 + 96) + 8LL);
            if ( !v33 )
              goto LABEL_65;
          }
          else
          {
            v78 = sub_877F80(a2);
            v32 = v88;
            if ( v78 == 3 )
            {
              v79 = *(_QWORD **)(*(_QWORD *)(v92 + 96) + 48LL);
              if ( !v79 )
                goto LABEL_65;
              while ( 1 )
              {
                v33 = v79[1];
                switch ( *(_BYTE *)(v33 + 80) )
                {
                  case 4:
                  case 5:
                    v80 = *(_QWORD *)(*(_QWORD *)(v33 + 96) + 80LL);
                    break;
                  case 6:
                    v80 = *(_QWORD *)(*(_QWORD *)(v33 + 96) + 32LL);
                    break;
                  case 9:
                  case 0xA:
                    v80 = *(_QWORD *)(*(_QWORD *)(v33 + 96) + 56LL);
                    break;
                  case 0x13:
                  case 0x14:
                  case 0x15:
                  case 0x16:
                    v80 = *(_QWORD *)(v33 + 88);
                    break;
                  default:
                    BUG();
                }
                if ( *(_DWORD *)(v80 + 64) == *(_DWORD *)(a1 + 156) )
                  break;
                v79 = (_QWORD *)*v79;
                if ( !v79 )
                  goto LABEL_65;
              }
            }
            else
            {
              v33 = sub_883800(*(_QWORD *)(v92 + 96) + 192LL, *(_QWORD *)a2);
              if ( !v33 )
                goto LABEL_65;
              v32 = v88;
              while ( 1 )
              {
                v81 = *(_BYTE *)(v33 + 80);
                if ( v81 == 17 || v81 == 20 )
                  break;
                v33 = *(_QWORD *)(v33 + 32);
                if ( !v33 )
                  goto LABEL_65;
              }
            }
          }
          if ( *(_BYTE *)(v33 + 80) == 17 )
          {
            v33 = *(_QWORD *)(v33 + 88);
            if ( v33 )
            {
              v97 = 1;
              goto LABEL_98;
            }
          }
          else
          {
LABEL_98:
            while ( *(_BYTE *)(v33 + 80) != 20 || *(_DWORD *)(*(_QWORD *)(v33 + 88) + 64LL) != *(_DWORD *)(a1 + 156) )
            {
              if ( v97 )
              {
                v33 = *(_QWORD *)(v33 + 8);
                if ( v33 )
                  continue;
              }
              goto LABEL_65;
            }
            *(_QWORD *)(v32 + 88) = v33;
            v94 = v32;
            v100 = *(_QWORD *)(v33 + 88);
            v74 = sub_878440();
            v74[1] = v5;
            v86 = v100;
            v90 = v94;
            *v74 = *(_QWORD *)(v100 + 96);
            *(_QWORD *)(v100 + 96) = v74;
            *(_QWORD *)(v94 + 288) = *(_QWORD *)(v100 + 288);
            *(_QWORD *)(v94 + 72) = *(_QWORD *)(v100 + 72);
            *(_DWORD *)(*(_QWORD *)(a1 + 192) + 44LL) = *(_DWORD *)(*(_QWORD *)(v100 + 328) + 44LL);
            v75 = *(_BYTE *)(v100 + 248) & 2;
            v76 = *(_QWORD *)(v94 + 176);
            v77 = *(_BYTE *)(v94 + 248) & 0xFD;
            v95 = *(_BYTE **)(v100 + 176);
            v101 = (_BYTE *)v76;
            *(_BYTE *)(v90 + 248) = v75 | v77;
            sub_736C90(v76, v95[192] >> 7);
            v101[172] = v95[172];
            v101[88] = v95[88] & 0x70 | v101[88] & 0x8F;
            *(_BYTE *)(v90 + 248) = *(_BYTE *)(v86 + 248) & 0x10 | *(_BYTE *)(v90 + 248) & 0xEF;
            *(_QWORD *)(v90 + 56) = *(_QWORD *)(v86 + 56);
            v101[206] = v95[206] & 0x10 | v101[206] & 0xEF;
          }
        }
      }
    }
  }
LABEL_65:
  sub_897580(v7, (__int64 *)v5, v12);
  *(_QWORD *)(*(_QWORD *)(v12 + 176) + 248LL) = *(_QWORD *)(v12 + 104);
  v24 = (8 * (*(_BYTE *)(v7 + 84) & 1)) | *(_BYTE *)(v12 + 160) & 0xF7;
  *(_BYTE *)(v12 + 160) = v24;
  v25 = v24 & 0xEF | (16 * (*(_BYTE *)(v7 + 88) & 1));
  *(_BYTE *)(v12 + 160) = v25;
  v26 = (32 * (*(_BYTE *)(v7 + 128) & 1)) | v25 & 0xDF;
  *(_BYTE *)(v12 + 160) = v26;
  if ( *(_DWORD *)(v7 + 24) && !*(_DWORD *)(v7 + 16) && (v26 & 1) == 0 )
    sub_899910(v5, v12, a5);
  sub_890F90(v7, v15, v5, v12);
  v27 = *(_QWORD *)(v15 + 152);
  if ( *(_BYTE *)(v27 + 140) == 7
    && !*(_QWORD *)(v12 + 368)
    && (v34 = *(_QWORD *)(*(_QWORD *)(v27 + 168) + 56LL)) != 0
    && (*(_BYTE *)v34 & 0x20) != 0
    && (sub_879080((__m128i *)(v12 + 336), *(const __m128i **)(v34 + 8), *(_QWORD *)(v7 + 192)),
        (v91 = *(_DWORD *)(v7 + 52)) == 0) )
  {
    if ( *(_QWORD *)(v7 + 240) )
      LOBYTE(v91) = 1;
    else
      sub_894C00(*(_QWORD *)v15);
  }
  else
  {
    LOBYTE(v91) = 0;
  }
  if ( !v105 )
  {
    if ( (*(_BYTE *)(v5 + 81) & 0x10) != 0
      && !*(_DWORD *)(v7 + 44)
      && !*(_DWORD *)(v7 + 48)
      && (!*(_QWORD *)(v7 + 240) || (v105 = *(_DWORD *)(v7 + 16)) != 0) )
    {
      v105 = sub_89BFC0(v7, v5, 1, a5) == 0;
    }
    if ( !(_DWORD)qword_4D04464 )
      goto LABEL_15;
    goto LABEL_77;
  }
LABEL_12:
  v105 = 1;
  if ( !(_DWORD)qword_4D04464 )
  {
    if ( !v5 )
      goto LABEL_45;
LABEL_14:
    if ( !v15 )
    {
LABEL_16:
      sub_8975E0((const __m128i *)v7, dword_4F0664C, 0);
      goto LABEL_17;
    }
LABEL_15:
    if ( (*(_BYTE *)(v15 + 193) & 0x10) != 0 )
      goto LABEL_17;
    goto LABEL_16;
  }
LABEL_77:
  if ( word_4F06418[0] == 56 && (unsigned __int16)sub_7BE840(0, 0) == 152 )
  {
    *(_QWORD *)(v7 + 36) = 0x100000001LL;
    *(_QWORD *)(v7 + 472) = *(_QWORD *)&dword_4F063F8;
    sub_7B8B50(0, 0, v38, v39, v40, v41);
    *(_QWORD *)(v7 + 480) = qword_4F063F0;
    sub_7B8B50(0, 0, v42, v43, v44, v45);
  }
  if ( v5 )
    goto LABEL_14;
LABEL_17:
  if ( !v105 )
  {
    sub_7ADF70((__int64)v107, 1);
    v87 = dword_4F06650[0];
    v16 = sub_877F80(v5);
    sub_88F140(v7, (unsigned __int64)v107, v16 == 1, a5);
    v84 = dword_4F06650[0];
    v17 = dword_4F04C3C;
    dword_4F04C3C = 1;
    if ( !v15 || (*(_BYTE *)(v15 + 193) & 0x10) == 0 )
    {
      v82 = v17;
      if ( *(_DWORD *)(v7 + 36) )
        sub_8756F0(3, v5, a5, 0);
      else
        sub_8756F0(1, v5, a5, 0);
      v17 = v82;
    }
    dword_4F04C3C = v17;
    if ( (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) & 0x42) == 2
      && (*(_BYTE *)(v5 + 81) & 0x10) != 0
      && *(_QWORD *)(v7 + 240) )
    {
      if ( !*(_DWORD *)(v7 + 36) )
      {
        if ( *(_QWORD *)(v12 + 304) )
        {
          if ( *(_QWORD *)(v12 + 32) )
          {
            v18 = *(_QWORD *)(v7 + 240);
            goto LABEL_25;
          }
          goto LABEL_58;
        }
        goto LABEL_118;
      }
      if ( (*(_BYTE *)(v15 + 206) & 2) != 0 || *(_DWORD *)(v7 + 40) || *(_BYTE *)(v5 + 80) != 20 )
      {
        if ( *(_QWORD *)(v12 + 304) )
        {
LABEL_58:
          if ( a3 )
          {
            *(__m128i *)(v12 + 184) = _mm_loadu_si128(a3);
            *(__m128i *)(v12 + 200) = _mm_loadu_si128(a3 + 1);
            *(__m128i *)(v12 + 216) = _mm_loadu_si128(a3 + 2);
            *(__m128i *)(v12 + 232) = _mm_loadu_si128(a3 + 3);
            *(__m128i *)(v12 + 248) = _mm_loadu_si128(a3 + 4);
            *(__m128i *)(v12 + 264) = _mm_loadu_si128(a3 + 5);
            *(_QWORD *)(v12 + 280) = a3[6].m128i_i64[0];
            a3->m128i_i64[1] = 0;
          }
          *(_QWORD *)(*(_QWORD *)(v12 + 176) + 64LL) = *(_QWORD *)&a5->_flags;
          sub_879080((__m128i *)v12, v107, *(_QWORD *)(v7 + 192));
LABEL_24:
          v18 = *(_QWORD *)(v7 + 240);
          if ( !v18 )
          {
LABEL_26:
            sub_890F90(v7, v15, v5, v12);
            sub_893800(v7, v5, v12);
            if ( *(_DWORD *)(v7 + 16)
              && (unk_4F04C48 == -1 || (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) & 6) == 0) )
            {
              sub_899080((_QWORD *)v12, *(_QWORD *)(v7 + 240));
            }
            if ( !*(_DWORD *)(v7 + 36) )
              goto LABEL_37;
            goto LABEL_31;
          }
LABEL_25:
          sub_5EA7A0(
            v5,
            *(_QWORD *)(v7 + 440),
            *(_QWORD *)(v7 + 448),
            v18,
            *(_DWORD *)(v7 + 36),
            v91,
            (__int64)qword_4D03B88);
          goto LABEL_26;
        }
        goto LABEL_118;
      }
      *(_QWORD *)(v12 + 80) = sub_888280(v5, v12, v87, v84);
    }
    if ( *(_QWORD *)(v12 + 304) )
    {
LABEL_22:
      if ( !*(_DWORD *)(v7 + 36) && *(_QWORD *)(v12 + 32) )
        goto LABEL_24;
      goto LABEL_58;
    }
LABEL_118:
    sub_879080((__m128i *)(v12 + 296), (const __m128i *)(v7 + 288), *(_QWORD *)(v7 + 192));
    *(_DWORD *)(v7 + 320) = 1;
    goto LABEL_22;
  }
LABEL_45:
  sub_7ADF70((__int64)v107, 0);
  sub_88F140(v7, (unsigned __int64)v107, 1u, a5);
  sub_7AEA70(v107);
  v22 = *(_DWORD *)(v7 + 36);
  *(_DWORD *)(v7 + 52) = 1;
  if ( !v22 )
    goto LABEL_10;
  if ( !v12 )
  {
    **(_WORD **)(v7 + 184) = 74;
    goto LABEL_48;
  }
  v105 = 1;
LABEL_31:
  if ( (*(_BYTE *)(v12 + 248) & 0x18) != 0 )
    **(_WORD **)(v7 + 184) = 75;
  else
    **(_WORD **)(v7 + 184) = 74;
  if ( v105 )
  {
LABEL_48:
    if ( a3 && a3[4].m128i_i8[0] < 0 )
      sub_6851C0(0x5Du, a5);
    goto LABEL_10;
  }
  *(_DWORD *)(v12 + 280) = sub_7A7D00();
  if ( a3 && a3[4].m128i_i8[0] < 0 )
    sub_6851C0(0x5Du, a5);
LABEL_37:
  if ( *(_BYTE *)(v5 + 80) != 20 )
    goto LABEL_10;
  v19 = *(_QWORD *)(*(_QWORD *)(v12 + 176) + 152LL);
  v20 = v19;
  if ( *(_BYTE *)(v19 + 140) == 12 )
  {
    do
      v19 = *(_QWORD *)(v19 + 160);
    while ( *(_BYTE *)(v19 + 140) == 12 );
    v20 = v19;
  }
  v85 = sub_877F80(v5);
  v21 = sub_877F80(v5);
  switch ( *(_BYTE *)(v5 + 80) )
  {
    case 4:
    case 5:
      v98 = *(_QWORD *)(*(_QWORD *)(v5 + 96) + 80LL);
      break;
    case 6:
      v98 = *(_QWORD *)(*(_QWORD *)(v5 + 96) + 32LL);
      break;
    case 9:
    case 0xA:
      v98 = *(_QWORD *)(*(_QWORD *)(v5 + 96) + 56LL);
      break;
    case 0x13:
    case 0x14:
    case 0x15:
    case 0x16:
      v98 = *(_QWORD *)(v5 + 88);
      break;
    default:
      v98 = 0;
      break;
  }
  v106 = 8;
  if ( unk_4F0697C )
  {
    if ( v85 == 3 || (v106 = 3, v21 == 1) )
      v106 = 5;
  }
  if ( !v96 )
    goto LABEL_186;
  v83 = v12;
  v46 = 0;
  v47 = 0;
  v89 = v7;
  v48 = v96;
  v93 = v5;
  do
  {
    v49 = *((_BYTE *)v48 + 56);
    v50 = v48[1];
    if ( (v49 & 9) != 1 )
    {
      if ( *((_DWORD *)v48 + 15) == v46 )
        goto LABEL_133;
      goto LABEL_130;
    }
    if ( !unk_4D04310 )
    {
      sub_684AA0(8u, 0x2C1u, (_DWORD *)(v50 + 48));
LABEL_206:
      if ( *((_DWORD *)v48 + 15) == v46 )
      {
        v49 = *((_BYTE *)v48 + 56);
        goto LABEL_133;
      }
LABEL_130:
      if ( unk_4D04310 )
      {
        v49 = *((_BYTE *)v48 + 56);
        if ( (v49 & 1) != 0 )
          goto LABEL_132;
      }
      goto LABEL_143;
    }
    if ( !*(_DWORD *)(v89 + 16) )
      goto LABEL_141;
    if ( !*(_DWORD *)(v89 + 36) )
    {
      sub_684AA0(7u, 0xA3Fu, (_DWORD *)(v50 + 48));
      goto LABEL_206;
    }
    *(_BYTE *)(v98 + 424) |= 8u;
    v49 = *((_BYTE *)v48 + 56);
LABEL_141:
    if ( *((_DWORD *)v48 + 15) == v46 )
      goto LABEL_133;
    if ( (v49 & 1) != 0 )
    {
LABEL_132:
      v46 = *((_DWORD *)v48 + 15);
LABEL_133:
      if ( (v49 & 0x70) == 0x10 )
        v47 = 1;
      goto LABEL_135;
    }
LABEL_143:
    v51 = **(_QWORD ***)(v20 + 168);
    if ( v51 )
    {
      while ( 1 )
      {
        v53 = *(_BYTE *)(v50 + 80);
        v54 = v51[1];
        v55 = *(_QWORD *)(v50 + 88);
        if ( v53 == 3 )
          break;
        if ( v53 != 2 )
        {
          v52 = sub_8DCC70(v54, *(_QWORD *)(v55 + 104), 1, 0);
          goto LABEL_146;
        }
        if ( (unsigned int)sub_8DCCD0(v54, v55, 1, 0) )
        {
LABEL_151:
          v46 = *((_DWORD *)v48 + 15);
          v49 = *((_BYTE *)v48 + 56);
          goto LABEL_133;
        }
LABEL_147:
        v51 = (_QWORD *)*v51;
        if ( !v51 )
          goto LABEL_172;
      }
      v52 = sub_8DCBF0(v54, v55, 1, 0);
LABEL_146:
      if ( v52 )
        goto LABEL_151;
      goto LABEL_147;
    }
LABEL_172:
    v46 = *((_DWORD *)v48 + 15);
    if ( v85 == 3 && (unsigned int)sub_88FAD0(*(_BYTE *)(v50 + 80), *(_QWORD *)(v50 + 88), *(_QWORD *)(v20 + 160), 1u) )
    {
      if ( (v48[7] & 0x70) == 0x10 )
        v47 = 1;
    }
    else
    {
      for ( i = v20; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
        ;
      if ( *(_QWORD *)(*(_QWORD *)(i + 168) + 40LL) )
      {
        v61 = sub_88FAD0(*(_BYTE *)(v50 + 80), *(_QWORD *)(v50 + 88), *(_QWORD *)(*(_QWORD *)(v20 + 168) + 40LL), 1u);
        v62 = v61 == 0;
      }
      else
      {
        v62 = 1;
        v61 = 0;
      }
      if ( (v47 & 1) != 0 && v62 )
      {
        v47 = 1;
        sub_686B60(unk_4F07471, 0x831u, (FILE *)(v50 + 48), v50, v93);
LABEL_181:
        *(_BYTE *)(v98 + 424) |= 1u;
        if ( v106 != 3 && (v48[7] & 0x10) == 0 )
          sub_686B60(v106, 0x1BDu, (FILE *)(v50 + 48), v50, v93);
        goto LABEL_135;
      }
      if ( (v48[7] & 0x70) == 0x10 )
        v47 = 1;
      if ( !v61 )
        goto LABEL_181;
    }
LABEL_135:
    v48 = (__int64 *)*v48;
  }
  while ( v48 );
  v12 = v83;
  v7 = v89;
LABEL_186:
  if ( !v104 && (*(_BYTE *)(v98 + 424) & 8) != 0 )
  {
    if ( !*(_QWORD *)(v7 + 240) )
      goto LABEL_192;
    v63 = *(unsigned int *)(v7 + 148);
    flags = (unsigned int)a5->_flags;
    if ( (_DWORD)flags == (_DWORD)v63 )
    {
      v63 = *(unsigned __int16 *)(v7 + 152);
      flags = *((unsigned __int16 *)&a5->_flags + 2);
    }
    if ( flags != v63 )
    {
LABEL_192:
      sub_6854F0(7u, 0xA40u, a5, (_QWORD *)(v7 + 148));
      *(_BYTE *)(v98 + 424) &= ~8u;
    }
  }
  if ( (*(_BYTE *)(v12 + 160) & 8) != 0 )
  {
    v65 = *(_QWORD *)(v12 + 176);
    if ( v65 )
    {
      if ( (*(_BYTE *)(v65 + 198) & 0x20) != 0 )
      {
        v66 = v96;
        if ( v96 )
        {
          v67 = 0;
          v68 = 0;
          v69 = 0;
          do
          {
            if ( (v66[7] & 0x10) != 0 )
            {
              while ( 1 )
              {
                v70 = (__int64 *)*v66;
                v71 = v68 & (v67 == 0);
                v72 = v71;
                if ( !*v66 )
                {
                  if ( !v71 )
                    goto LABEL_10;
                  goto LABEL_212;
                }
                if ( !v69 )
                  break;
                if ( v71 )
                  goto LABEL_214;
                v66 = (__int64 *)*v66;
                v68 = v69;
                if ( (v70[7] & 0x10) == 0 )
                  goto LABEL_203;
              }
              v69 = 1;
              sub_6851C0(0xDF5u, (_DWORD *)(v66[1] + 48));
              v68 = 1;
              if ( v72 )
              {
LABEL_214:
                v69 = 1;
LABEL_212:
                v67 = 1;
                sub_6851C0(0xDF6u, (_DWORD *)(v66[1] + 48));
                v68 = 1;
              }
            }
LABEL_203:
            v66 = (__int64 *)*v66;
          }
          while ( v66 );
        }
      }
    }
  }
LABEL_10:
  *a4 = v12;
  return a4;
}
