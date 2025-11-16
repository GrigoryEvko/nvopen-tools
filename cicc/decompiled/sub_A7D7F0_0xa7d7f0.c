// Function: sub_A7D7F0
// Address: 0xa7d7f0
//
__int64 __fastcall sub_A7D7F0(int a1, __int64 a2, __int64 a3, size_t a4, _QWORD *a5)
{
  unsigned __int64 v6; // r14
  char v11; // cl
  char *v12; // rdi
  size_t v13; // r14
  char *v14; // rcx
  __int64 v15; // rdi
  unsigned __int64 v16; // rax
  _BYTE *v17; // rdx
  void *v18; // rsi
  __int64 v19; // rax
  __int64 v20; // rdi
  __int64 v21; // rdx
  __int64 v22; // rax
  void *v23; // r13
  int v24; // eax
  int v25; // eax
  __int64 v26; // rdi
  _QWORD *v27; // rax
  char *v28; // rdi
  __int64 v29; // rdx
  unsigned int v30; // r13d
  __int64 v31; // rdi
  unsigned int v32; // r13d
  __int64 v33; // rdi
  unsigned __int64 v34; // rax
  char *v35; // rdx
  char *v36; // rax
  char *v37; // rdx
  char v38; // cl
  size_t v39; // rbx
  char *v40; // r13
  int v41; // eax
  int v42; // eax
  __int64 v43; // rsi
  unsigned __int8 v44; // cl
  __int64 v45; // rdi
  size_t v46; // rax
  unsigned int v47; // r14d
  __int64 v48; // rax
  __int64 v49; // rdx
  unsigned __int64 v50; // rbx
  __int64 v51; // rdi
  __int64 v52; // rax
  __int64 v53; // rax
  __int64 v54; // rdi
  __int64 v55; // rsi
  __int64 v56; // r13
  __int64 v57; // rax
  bool v58; // zf
  __int64 v59; // rax
  __int64 v60; // rdi
  char *v61; // r13
  size_t v62; // r14
  __int64 v63; // rsi
  __int64 v64; // rcx
  size_t v65; // r14
  int v66; // eax
  int v67; // eax
  void *v68; // r12
  size_t v69; // r13
  bool v70; // al
  const char *v71; // rdx
  int v72; // eax
  __int64 v73; // rax
  _QWORD *v74; // rsi
  int v75; // eax
  __int64 v76; // rdi
  __int64 v77; // rdi
  __int64 v78; // rdi
  _QWORD *v79; // rax
  size_t v80; // [rsp+8h] [rbp-88h]
  char v81; // [rsp+8h] [rbp-88h]
  char v82; // [rsp+8h] [rbp-88h]
  char v83; // [rsp+8h] [rbp-88h]
  void *s1; // [rsp+10h] [rbp-80h] BYREF
  size_t v85; // [rsp+18h] [rbp-78h]
  _QWORD v86[2]; // [rsp+20h] [rbp-70h] BYREF
  _QWORD *v87; // [rsp+30h] [rbp-60h] BYREF
  __int64 v88; // [rsp+38h] [rbp-58h]
  _QWORD v89[2]; // [rsp+40h] [rbp-50h] BYREF
  __int16 v90; // [rsp+50h] [rbp-40h]

  v85 = a4;
  v6 = a4;
  s1 = (void *)a3;
  if ( a4 <= 3 )
    goto LABEL_2;
  if ( *(_DWORD *)a3 == 1953063538 )
  {
    if ( (*(_BYTE *)(a2 + 2) & 1) != 0 )
      sub_B2C6D0(a2);
    v15 = *(_QWORD *)(a2 + 40);
    LODWORD(v6) = 1;
    v87 = *(_QWORD **)(*(_QWORD *)(a2 + 96) + 8LL);
    *a5 = sub_B6E160(v15, 14, &v87, 1);
    return (unsigned int)v6;
  }
  v11 = a1;
  if ( v6 == 14 )
  {
    if ( *(_QWORD *)a3 == 0x702E646165726874LL && *(_DWORD *)(a3 + 8) == 1953393007 && *(_WORD *)(a3 + 12) == 29285 )
    {
      LODWORD(v6) = 1;
      *a5 = sub_B6E160(*(_QWORD *)(a2 + 40), 352, 0, 0);
      return (unsigned int)v6;
    }
    if ( *(_DWORD *)a3 == 1852794222 && *(_BYTE *)(a3 + 4) == 46 )
    {
      v17 = (_BYTE *)(a3 + 5);
      v85 = 9;
      v16 = 9;
      s1 = (void *)(a3 + 5);
      goto LABEL_64;
    }
  }
  else if ( v6 != 4 && *(_DWORD *)a3 == 1852794222 && *(_BYTE *)(a3 + 4) == 46 )
  {
    v16 = v6 - 5;
    v17 = (_BYTE *)(a3 + 5);
    s1 = v17;
    v85 = v6 - 5;
    if ( v6 - 5 <= 5 )
    {
      if ( v16 <= 2 )
      {
        if ( (_BYTE)a1 )
          goto LABEL_30;
        goto LABEL_56;
      }
LABEL_65:
      if ( *(_WORD *)v17 == 26210 && v17[2] == 109 )
      {
        v64 = a3 + 8;
        s1 = (void *)(a3 + 8);
        v85 = v6 - 8;
        if ( v6 - 8 <= 0xB )
          goto LABEL_2;
        v65 = v6 - 20;
        if ( *(_QWORD *)(v64 + v65) != 0x762E32336634762ELL || *(_DWORD *)(v64 + v65 + 8) != 946419249 )
          goto LABEL_2;
        v85 = v65;
        if ( (_BYTE)a1 )
        {
          v63 = 3598;
          if ( v65 != 3 )
          {
            v63 = 3596;
            goto LABEL_150;
          }
        }
        else
        {
          v63 = 574;
          if ( v65 != 3 )
          {
LABEL_149:
            v63 = 572;
LABEL_150:
            if ( v65 != 4 )
              goto LABEL_2;
            if ( *(_DWORD *)(a3 + 8) != 1651269996 )
            {
              v63 = (_BYTE)a1 == 0 ? 573 : 3597;
              if ( *(_DWORD *)(a3 + 8) != 1953259884 )
                goto LABEL_2;
            }
LABEL_137:
            LODWORD(v6) = 1;
            *a5 = sub_B6E160(*(_QWORD *)(a2 + 40), v63, 0, 0);
            return (unsigned int)v6;
          }
        }
        if ( *(_WORD *)(a3 + 8) == 27757 && *(_BYTE *)(a3 + 10) == 97 )
          goto LABEL_137;
        if ( (_BYTE)a1 )
          goto LABEL_2;
        goto LABEL_149;
      }
      if ( (_BYTE)a1 )
      {
        if ( v16 > 4 )
        {
          if ( *(_DWORD *)v17 == 2053923702 && v17[4] == 46 )
          {
            v32 = 65;
            goto LABEL_78;
          }
          if ( *(_DWORD *)v17 == 1953391478 && v17[4] == 46 )
          {
            v32 = 66;
            goto LABEL_78;
          }
          if ( v16 > 6 )
          {
            if ( *(_DWORD *)v17 == 1684107638 && *((_WORD *)v17 + 2) == 29540 && v17[6] == 46 )
            {
              v32 = 311;
              goto LABEL_78;
            }
            if ( *(_DWORD *)v17 == 1684107638 && *((_WORD *)v17 + 2) == 30052 && v17[6] == 46 )
            {
              v32 = 359;
              goto LABEL_78;
            }
            if ( *(_DWORD *)v17 == 1970499958 && *((_WORD *)v17 + 2) == 29538 && v17[6] == 46 )
            {
              v32 = 338;
              goto LABEL_78;
            }
            if ( *(_DWORD *)v17 == 1970499958 && *((_WORD *)v17 + 2) == 30050 && v17[6] == 46 )
            {
              v32 = 371;
LABEL_78:
              if ( (*(_BYTE *)(a2 + 2) & 1) != 0 )
                sub_B2C6D0(a2);
              v33 = *(_QWORD *)(a2 + 40);
              LODWORD(v6) = a1;
              v87 = *(_QWORD **)(*(_QWORD *)(a2 + 96) + 8LL);
              *a5 = sub_B6E160(v33, v32, &v87, 1);
              return (unsigned int)v6;
            }
            goto LABEL_31;
          }
        }
LABEL_30:
        if ( v16 <= 2 )
          goto LABEL_2;
LABEL_31:
        if ( *(_WORD *)v17 != 29558 || v17[2] != 116 )
          goto LABEL_2;
        s1 = (void *)(a3 + 8);
        v85 = v6 - 8;
        if ( !byte_4F80C40 && (unsigned int)sub_2207590(&byte_4F80C40) )
        {
          sub_C88F40(&unk_4F80C50, "^([1234]|[234]lane)\\.v[a-z0-9]*$", 32, 0);
          __cxa_atexit(sub_C88FF0, &unk_4F80C50, &qword_4A427C0);
          sub_2207640(&byte_4F80C40);
        }
        v18 = s1;
        v87 = v89;
        v88 = 0x200000000LL;
        LODWORD(v6) = sub_C89090(&unk_4F80C50, s1, v85, &v87, 0);
        if ( (_BYTE)v6 )
        {
          v19 = *(_QWORD *)(a2 + 24);
          v20 = *(_QWORD *)(a2 + 40);
          v21 = *(_QWORD *)(v19 + 16);
          v22 = (8LL * *(unsigned int *)(v19 + 12) - 8) >> 3;
          v86[0] = *(_QWORD *)(v21 + 8);
          v86[1] = *(_QWORD *)(v21 + 16);
          if ( v87[3] == 1 )
            v18 = (void *)dword_3F27E30[v22 - 3];
          else
            v18 = (void *)dword_3F27E20[v22 - 5];
          *a5 = sub_B6E160(v20, v18, v86, 2);
        }
        if ( v87 != v89 )
          _libc_free(v87, v18);
        return (unsigned int)v6;
      }
      if ( v16 > 5 )
      {
        if ( *(_DWORD *)v17 == 1852404326 )
        {
          v30 = 310;
          if ( *((_WORD *)v17 + 2) == 28276 )
            goto LABEL_59;
        }
        goto LABEL_57;
      }
LABEL_56:
      if ( v16 <= 3 )
        goto LABEL_2;
LABEL_57:
      if ( *(_DWORD *)v17 == 1953063538 )
      {
        v30 = 14;
LABEL_59:
        if ( (*(_BYTE *)(a2 + 2) & 1) != 0 )
          sub_B2C6D0(a2);
        v31 = *(_QWORD *)(a2 + 40);
        LODWORD(v6) = 1;
        v87 = *(_QWORD **)(*(_QWORD *)(a2 + 96) + 8LL);
        *a5 = sub_B6E160(v31, v30, &v87, 1);
        return (unsigned int)v6;
      }
      if ( *(_DWORD *)v17 == 1885627489 )
      {
        if ( *(_QWORD *)(a2 + 104) != 2 )
          goto LABEL_2;
        v43 = **(_QWORD **)(*(_QWORD *)(a2 + 24) + 16LL);
        if ( (unsigned int)*(unsigned __int8 *)(v43 + 8) - 17 <= 1 )
        {
          v44 = *(_BYTE *)(*(_QWORD *)(v43 + 24) + 8LL);
          if ( v44 <= 3u || v44 == 5 || (v44 & 0xFD) == 4 )
          {
            v45 = *(_QWORD *)(a2 + 40);
            v87 = **(_QWORD ***)(*(_QWORD *)(a2 + 24) + 16LL);
            LODWORD(v6) = 1;
            *a5 = sub_B6E160(v45, 579, &v87, 1);
            return (unsigned int)v6;
          }
        }
      }
      if ( v16 == 4 || *(_DWORD *)v17 != 1986225762 || v17[4] != 116 )
        goto LABEL_2;
LABEL_102:
      *a5 = 0;
      LODWORD(v6) = 1;
      return (unsigned int)v6;
    }
LABEL_64:
    if ( *(_DWORD *)v17 == 1868850786 && *((_WORD *)v17 + 2) == 11892 )
    {
      v46 = v6 - 11;
      s1 = (void *)(a3 + 11);
      v85 = v46;
      v47 = (_BYTE)a1 == 0 ? 571 : 3595;
      if ( v46 == 10 )
      {
        if ( *(_QWORD *)(a3 + 11) == 0x38762E3233663276LL && *(_WORD *)(a3 + 19) == 14441 )
          goto LABEL_113;
      }
      else if ( v46 == 11
             && *(_QWORD *)(a3 + 11) == 0x31762E3233663476LL
             && *(_WORD *)(a3 + 19) == 26934
             && *(_BYTE *)(a3 + 21) == 56 )
      {
LABEL_113:
        v48 = sub_BCAE30(**(_QWORD **)(*(_QWORD *)(a2 + 24) + 16LL));
        v88 = v49;
        v87 = (_QWORD *)v48;
        v50 = (unsigned __int64)sub_CA1930(&v87) >> 4;
        v51 = **(_QWORD **)(a2 + 40);
        v87 = **(_QWORD ***)(*(_QWORD *)(a2 + 24) + 16LL);
        v52 = sub_BCB150(v51);
        v53 = sub_BCDA70(v52, v50);
        v54 = *(_QWORD *)(a2 + 40);
        v55 = v47;
        v88 = v53;
        LODWORD(v6) = 1;
        *a5 = sub_B6E160(v54, v55, &v87, 2);
        return (unsigned int)v6;
      }
      goto LABEL_2;
    }
    goto LABEL_65;
  }
  if ( (_BYTE)a1 )
  {
    if ( *(_DWORD *)a3 == 778401389 )
    {
      v28 = (char *)(a3 + 4);
      s1 = (void *)(a3 + 4);
      v85 = v6 - 4;
      if ( v6 == 10 )
      {
        if ( *(_DWORD *)(a3 + 4) == 1886675830
          && *(_WORD *)(a3 + 8) == 13366
          && *(_DWORD *)(**(_QWORD **)(*(_QWORD *)(a2 + 24) + 16LL) + 32LL) == 4 )
        {
          LODWORD(v6) = a1;
          v87 = (_QWORD *)sub_BD5D20(a2);
          v90 = 773;
          v88 = v29;
          v89[0] = ".old";
          sub_BD6B50(a2, &v87);
          return (unsigned int)v6;
        }
        goto LABEL_2;
      }
      if ( v6 - 4 > 4 )
      {
        v34 = v6 - 9;
        v35 = &v28[v6 - 9];
        if ( *(_DWORD *)v35 == 1765045806 && v35[4] == 49 )
        {
          v85 = v6 - 9;
          if ( v34 <= 0x16 )
          {
            if ( v34 <= 5 )
              goto LABEL_2;
          }
          else
          {
            v36 = &v28[v6 - 32];
            if ( !(*(_QWORD *)v36 ^ 0x616369646572702ELL | *((_QWORD *)v36 + 1) ^ 0x366932762E646574LL)
              && *((_DWORD *)v36 + 4) == 880160308
              && *((_WORD *)v36 + 10) == 13161
              && v36[22] == 50 )
            {
              v85 = v6 - 32;
              if ( v6 == 40 )
              {
                LODWORD(v6) = a1;
                if ( *(_QWORD *)(a3 + 4) == 0x746E692E6C6C756DLL )
                  return (unsigned int)v6;
              }
              else if ( v6 == 39 )
              {
                LOBYTE(v6) = memcmp(v28, "vqdmull", 7u) == 0;
                return (unsigned int)v6;
              }
              goto LABEL_2;
            }
          }
          v37 = &v28[v6 - 15];
          if ( *(_DWORD *)v37 == 1764914734 && *((_WORD *)v37 + 2) == 13366 )
          {
            v85 = v6 - 15;
            if ( v6 - 15 > 0xB && (v81 = v11, !memcmp(v28, "vldr.gather.", 0xCu)) )
            {
              v38 = v81;
              s1 = (void *)(a3 + 16);
              v85 = v6 - 27;
            }
            else
            {
              v67 = sub_95CB50((const void **)&s1, "vstr.scatter.", 0xDu);
              v38 = 0;
              LODWORD(v6) = v67;
              if ( !(_BYTE)v67 )
                return (unsigned int)v6;
            }
            v39 = v85;
            if ( v85 > 4 )
            {
              v40 = (char *)s1;
              v82 = v38;
              v41 = memcmp(s1, "base.", 5u);
              v38 = v82;
              if ( !v41 )
              {
                s1 = v40 + 5;
                v85 = v39 - 5;
                sub_95CB50((const void **)&s1, "wb.", 3u);
                LOBYTE(v42) = sub_9691B0(s1, v85, "predicated.v2i64", 16);
                LODWORD(v6) = v42;
                return (unsigned int)v6;
              }
            }
            v83 = v38;
            LODWORD(v6) = sub_95CB50((const void **)&s1, "offset.predicated.", 0x12u);
            if ( (_BYTE)v6 )
            {
              if ( v83 )
              {
                v68 = s1;
                v69 = v85;
                v70 = sub_9691B0(s1, v85, "v2i64.p0i64", 11);
                v71 = "v2i64.p0";
                if ( v70 )
                  return (unsigned int)v6;
              }
              else
              {
                v68 = s1;
                v69 = v85;
                if ( sub_9691B0(s1, v85, "p0i64.v2i64", 11) )
                  return (unsigned int)v6;
                v71 = "p0.v2i64";
              }
              LOBYTE(v72) = sub_9691B0(v68, v69, v71, 8);
              LODWORD(v6) = v72;
              return (unsigned int)v6;
            }
          }
        }
      }
LABEL_2:
      LODWORD(v6) = 0;
      return (unsigned int)v6;
    }
    if ( v6 <= 6 )
      goto LABEL_2;
    if ( *(_DWORD *)a3 != 778396771 )
      goto LABEL_2;
    if ( *(_WORD *)(a3 + 4) != 25462 )
      goto LABEL_2;
    if ( *(_BYTE *)(a3 + 6) != 120 )
      goto LABEL_2;
    v12 = (char *)(a3 + 7);
    s1 = (void *)(a3 + 7);
    v85 = v6 - 7;
    if ( v6 - 7 <= 0x15 )
      goto LABEL_2;
    v13 = v6 - 29;
    v14 = &v12[v13];
    if ( *(_QWORD *)&v12[v13] ^ 0x616369646572702ELL | *(_QWORD *)&v12[v13 + 8] ^ 0x366932762E646574LL
      || *((_DWORD *)v14 + 4) != 880160308
      || *((_WORD *)v14 + 10) != 12649 )
    {
      goto LABEL_2;
    }
    v85 = v13;
    if ( v13 == 2 )
    {
      if ( *(_WORD *)(a3 + 7) != 28977 && *(_WORD *)(a3 + 7) != 28978 )
        goto LABEL_173;
    }
    else if ( v13 != 3 || memcmp(v12, "1qa", 3u) )
    {
LABEL_173:
      if ( !sub_9691B0(s1, v85, "2qa", 3) && !sub_9691B0(s1, v85, "3q", 2) )
      {
        LOBYTE(v66) = sub_9691B0(s1, v85, "3qa", 3);
        LODWORD(v6) = v66;
        return (unsigned int)v6;
      }
    }
LABEL_20:
    LODWORD(v6) = a1;
    return (unsigned int)v6;
  }
  if ( *(_DWORD *)a3 != 778401395 )
    goto LABEL_2;
  s1 = (void *)(a3 + 4);
  v85 = v6 - 4;
  if ( v6 - 4 > 1 )
  {
    if ( *(_WORD *)(a3 + 4) == 26210 )
    {
      v61 = (char *)(a3 + 6);
      s1 = (void *)(a3 + 6);
      v85 = v6 - 6;
      if ( v6 - 6 > 4 )
      {
        v62 = v6 - 11;
        if ( *(_DWORD *)&v61[v62] == 1851878446 && v61[v62 + 4] == 101 )
        {
          v85 = v62;
          if ( v62 == 3 )
          {
            v63 = 1185;
            if ( !memcmp(v61, "dot", 3u) )
              goto LABEL_137;
          }
          else if ( v62 == 5 )
          {
            v63 = 1187;
            if ( !memcmp(v61, "mlalb", 5u) )
              goto LABEL_137;
            if ( !memcmp(v61, "mlalt", 5u) )
            {
              v63 = 1189;
              goto LABEL_137;
            }
          }
        }
      }
      goto LABEL_20;
    }
    if ( v6 == 16 )
    {
      if ( *(_QWORD *)(a3 + 4) == 0x3166622E74766366LL && *(_DWORD *)(a3 + 12) == 842229302 )
        goto LABEL_102;
    }
    else if ( v6 == 18 )
    {
      if ( *(_QWORD *)(a3 + 4) == 0x622E746E74766366LL
        && *(_DWORD *)(a3 + 12) == 1714827622
        && *(_WORD *)(a3 + 16) == 12851 )
      {
        goto LABEL_102;
      }
    }
    else if ( v6 - 4 <= 4 )
    {
      goto LABEL_126;
    }
    if ( *(_DWORD *)(a3 + 4) == 1902404705 && *(_BYTE *)(a3 + 8) == 118 )
    {
      v73 = *(_QWORD *)(a2 + 24);
      v85 = v6 - 9;
      v74 = *(_QWORD **)(v73 + 16);
      s1 = (void *)(a3 + 9);
      v75 = *(unsigned __int8 *)(*v74 + 8LL);
      if ( (unsigned int)(v75 - 17) <= 1 )
        LOBYTE(v75) = *(_BYTE *)(**(_QWORD **)(*v74 + 16LL) + 8LL);
      if ( (unsigned __int8)v75 <= 3u || (_BYTE)v75 == 5 || (LODWORD(v6) = a1, (v75 & 0xFD) == 4) )
      {
        v87 = (_QWORD *)*v74;
        v76 = *(_QWORD *)(a2 + 40);
        LODWORD(v6) = 1;
        v88 = v74[2];
        *a5 = sub_B6E160(v76, 1273, &v87, 2);
      }
      return (unsigned int)v6;
    }
LABEL_126:
    if ( *(_WORD *)(a3 + 4) == 25708 )
    {
      s1 = (void *)(a3 + 6);
      v85 = v6 - 6;
      if ( !byte_4F80C20 && (unsigned int)sub_2207590(&byte_4F80C20) )
      {
        sub_C88F40(&unk_4F80C30, "^[234](.nxv[a-z0-9]+|$)", 23, 0);
        __cxa_atexit(sub_C88FF0, &unk_4F80C30, &qword_4A427C0);
        sub_2207640(&byte_4F80C20);
      }
      LODWORD(v6) = sub_C89090(&unk_4F80C30, s1, v85, 0, 0);
      if ( !(_BYTE)v6 )
        goto LABEL_2;
      v56 = *(_QWORD *)(**(_QWORD **)(*(_QWORD *)(a2 + 24) + 16LL) + 24LL);
      if ( (*(_BYTE *)(a2 + 2) & 1) != 0 )
        sub_B2C6D0(a2);
      v57 = *(_QWORD *)(*(_QWORD *)(a2 + 96) + 8LL);
      v58 = *(_BYTE *)(v57 + 8) == 18;
      LODWORD(v57) = *(_DWORD *)(v57 + 32);
      BYTE4(v87) = v58;
      LODWORD(v87) = v57;
      v59 = sub_BCE1B0(v56, v87);
      v60 = *(_QWORD *)(a2 + 40);
      v86[0] = v59;
      *a5 = sub_B6E160(v60, dword_3F27E10[*(char *)s1 - 50], v86, 1);
      return (unsigned int)v6;
    }
  }
  if ( !(unsigned __int8)sub_95CB50((const void **)&s1, "tuple.", 6u) )
    goto LABEL_2;
  v23 = s1;
  v80 = v85;
  LOBYTE(v24) = sub_A7BBF0(s1, v85, "get", 3u);
  LODWORD(v6) = v24;
  if ( !(_BYTE)v24 )
  {
    LOBYTE(v25) = sub_A7BBF0(v23, v80, "set", 3u);
    LODWORD(v6) = v25;
    if ( (_BYTE)v25 )
    {
      v78 = *(_QWORD *)(a2 + 40);
      v79 = *(_QWORD **)(*(_QWORD *)(a2 + 24) + 16LL);
      v87 = (_QWORD *)v79[1];
      v88 = v79[3];
      v89[0] = v79[2];
      *a5 = sub_B6E160(v78, 382, &v87, 3);
      return (unsigned int)v6;
    }
    if ( !byte_4F80C00 && (unsigned int)sub_2207590(&byte_4F80C00) )
    {
      sub_C88F40(&unk_4F80C10, "^create[234](.nxv[a-z0-9]+|$)", 29, 0);
      __cxa_atexit(sub_C88FF0, &unk_4F80C10, &qword_4A427C0);
      sub_2207640(&byte_4F80C00);
    }
    LODWORD(v6) = sub_C89090(&unk_4F80C10, s1, v85, 0, 0);
    if ( (_BYTE)v6 )
    {
      v26 = *(_QWORD *)(a2 + 40);
      v27 = *(_QWORD **)(*(_QWORD *)(a2 + 24) + 16LL);
      v87 = (_QWORD *)*v27;
      v88 = v27[2];
      *a5 = sub_B6E160(v26, 382, &v87, 2);
      return (unsigned int)v6;
    }
    goto LABEL_2;
  }
  v87 = **(_QWORD ***)(*(_QWORD *)(a2 + 24) + 16LL);
  if ( (*(_BYTE *)(a2 + 2) & 1) != 0 )
    sub_B2C6D0(a2);
  v77 = *(_QWORD *)(a2 + 40);
  v88 = *(_QWORD *)(*(_QWORD *)(a2 + 96) + 8LL);
  *a5 = sub_B6E160(v77, 381, &v87, 2);
  return (unsigned int)v6;
}
