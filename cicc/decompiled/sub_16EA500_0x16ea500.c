// Function: sub_16EA500
// Address: 0x16ea500
//
char *__fastcall sub_16EA500(__int64 a1, int a2, int a3, __int64 a4, int a5, int a6)
{
  char *result; // rax
  char *v9; // rsi
  __int64 v10; // rcx
  __int64 v11; // r15
  __int64 v12; // r8
  int v13; // edi
  int v14; // r14d
  char *v15; // rdi
  char *v16; // rdx
  int v17; // esi
  __int64 v18; // rax
  __int64 v19; // rdx
  signed __int64 v20; // rdi
  __int64 v21; // rdx
  __int64 v22; // rdi
  signed __int64 v23; // rax
  signed __int64 v24; // rdx
  int v25; // r8d
  __int64 v26; // rdx
  int v27; // edx
  __int64 v28; // rax
  signed __int64 v29; // rcx
  unsigned __int64 v30; // rdx
  signed __int64 v31; // rsi
  __int64 v32; // rdx
  __int64 v33; // rdx
  signed __int64 v34; // rdx
  __int64 v35; // rax
  signed __int64 v36; // rsi
  __int64 v37; // rdx
  __int64 v38; // rax
  signed __int64 v39; // rdx
  signed __int64 v40; // rdi
  signed __int64 v41; // rsi
  __int64 v42; // rsi
  char *v43; // rdx
  unsigned __int64 v44; // rsi
  signed __int64 v45; // rdx
  signed __int64 v46; // rdi
  signed __int64 v47; // rsi
  __int64 v48; // rsi
  char *v49; // rax
  int v50; // r8d
  __int64 v51; // r14
  signed __int64 v52; // rcx
  signed __int64 v53; // rsi
  __int64 v54; // rcx
  signed __int64 v55; // rdx
  signed __int64 v56; // r14
  signed __int64 v57; // rcx
  signed __int64 v58; // rsi
  int v59; // eax
  int v60; // r8d
  unsigned __int64 v61; // rsi
  int v62; // edx
  char *v63; // rax
  int v64; // ecx
  char *v65; // rax
  char *v66; // rdx
  __int64 v67; // r15
  char *v68; // rax
  signed __int64 v69; // rsi
  signed __int64 v70; // rsi
  signed __int64 v71; // rcx
  __int64 v72; // rdx
  signed __int64 v73; // rsi
  __int64 v74; // rax
  int v75; // eax
  char *v76; // [rsp+0h] [rbp-60h]
  __int64 v77; // [rsp+0h] [rbp-60h]
  __int64 v78; // [rsp+8h] [rbp-58h]
  int v79; // [rsp+10h] [rbp-50h]
  __int64 v80; // [rsp+10h] [rbp-50h]
  __int64 v81; // [rsp+10h] [rbp-50h]
  __int64 v82; // [rsp+10h] [rbp-50h]
  int v83; // [rsp+10h] [rbp-50h]
  int v84; // [rsp+10h] [rbp-50h]
  int v85; // [rsp+10h] [rbp-50h]
  _DWORD v87[13]; // [rsp+2Ch] [rbp-34h] BYREF

  result = *(char **)a1;
  v9 = *(char **)(a1 + 8);
  if ( *(_QWORD *)a1 >= (unsigned __int64)v9 )
    goto LABEL_113;
  v10 = *(_QWORD *)(a1 + 40);
  v78 = v10;
  v11 = v10;
  if ( *result == 94 )
  {
    a6 = *(_DWORD *)(a1 + 16);
    *(_QWORD *)a1 = result + 1;
    if ( !a6 )
    {
      v71 = *(_QWORD *)(a1 + 32);
      v72 = v11;
      if ( v11 >= v71 )
      {
        v73 = ((v71 + 1 + ((unsigned __int64)(v71 + 1) >> 63)) & 0xFFFFFFFFFFFFFFFELL) + (v71 + 1) / 2;
        if ( v71 < v73 )
        {
          sub_16E90A0(a1, v73, v11, v71, a5, 0);
          v72 = *(_QWORD *)(a1 + 40);
        }
      }
      v74 = *(_QWORD *)(a1 + 24);
      v10 = v72 + 1;
      *(_QWORD *)(a1 + 40) = v72 + 1;
      *(_QWORD *)(v74 + 8 * v72) = 402653184;
    }
    *(_DWORD *)(*(_QWORD *)(a1 + 56) + 72LL) |= 1u;
    ++*(_DWORD *)(*(_QWORD *)(a1 + 56) + 76LL);
    result = *(char **)a1;
    v9 = *(char **)(a1 + 8);
    v11 = *(_QWORD *)(a1 + 40);
    if ( (unsigned __int64)v9 <= *(_QWORD *)a1 )
    {
LABEL_55:
      if ( v78 == v11 )
        goto LABEL_113;
      return result;
    }
  }
  LODWORD(v12) = 0;
  v13 = 1;
  while ( 1 )
  {
    v16 = result + 1;
    if ( result + 1 < v9 )
    {
      v10 = (unsigned int)*result;
      if ( (_DWORD)v10 == a2 )
      {
        v10 = (unsigned int)result[1];
        if ( (_DWORD)v10 == a3 )
        {
LABEL_48:
          if ( (_DWORD)v12 )
          {
            v27 = *(_DWORD *)(a1 + 16);
            v28 = v11 - 1;
            *(_QWORD *)(a1 + 40) = v11 - 1;
            if ( !v27 )
            {
              v29 = *(_QWORD *)(a1 + 32);
              if ( v28 >= v29 )
              {
                v30 = (v29 + 1 + ((unsigned __int64)(v29 + 1) >> 63)) & 0xFFFFFFFFFFFFFFFELL;
                v31 = v30 + (v29 + 1) / 2;
                if ( v29 < v31 )
                {
                  sub_16E90A0(a1, v31, v30, v29, v12, a6);
                  v28 = *(_QWORD *)(a1 + 40);
                  v11 = v28 + 1;
                }
              }
              v32 = *(_QWORD *)(a1 + 24);
              *(_QWORD *)(a1 + 40) = v11;
              *(_QWORD *)(v32 + 8 * v28) = 0x20000000;
            }
            *(_DWORD *)(*(_QWORD *)(a1 + 56) + 72LL) |= 2u;
            result = *(char **)(a1 + 56);
            ++*((_DWORD *)result + 20);
            v11 = *(_QWORD *)(a1 + 40);
          }
          goto LABEL_55;
        }
      }
      *(_QWORD *)a1 = v16;
      v14 = *result;
      if ( v14 != 92 )
        goto LABEL_6;
    }
    else
    {
      *(_QWORD *)a1 = v16;
      v14 = *result;
      if ( v14 != 92 )
        goto LABEL_6;
      if ( !*(_DWORD *)(a1 + 16) )
        *(_DWORD *)(a1 + 16) = 5;
      *(_QWORD *)(a1 + 8) = &unk_4FA17D0;
      v16 = (char *)&unk_4FA17D0;
    }
    *(_QWORD *)a1 = v16 + 1;
    LODWORD(v12) = *v16;
    v14 = v12 | 0x100;
    if ( ((unsigned int)v12 | 0x100) == 0x128 )
    {
      v38 = ++*(_QWORD *)(*(_QWORD *)(a1 + 56) + 112LL);
      if ( v38 <= 9 )
        *(_QWORD *)(a1 + 8 * v38 + 64) = *(_QWORD *)(a1 + 40);
      v10 = *(unsigned int *)(a1 + 16);
      if ( !(_DWORD)v10 )
      {
        v39 = *(_QWORD *)(a1 + 40);
        v40 = *(_QWORD *)(a1 + 32);
        if ( v39 >= v40 )
        {
          v12 = (v40 + 1) / 2;
          v41 = v12 + ((v40 + 1 + ((unsigned __int64)(v40 + 1) >> 63)) & 0xFFFFFFFFFFFFFFFELL);
          if ( v40 < v41 )
          {
            v80 = v38;
            sub_16E90A0(a1, v41, v39, 0, v12, a6);
            v39 = *(_QWORD *)(a1 + 40);
            v38 = v80;
          }
        }
        v42 = *(_QWORD *)(a1 + 24);
        *(_QWORD *)(a1 + 40) = v39 + 1;
        *(_QWORD *)(v42 + 8 * v39) = v38 | 0x68000000;
      }
      v43 = *(char **)a1;
      v44 = *(_QWORD *)(a1 + 8);
      if ( *(_QWORD *)a1 < v44 && (v44 <= (unsigned __int64)(v43 + 1) || *v43 != 92 || v43[1] != 41) )
      {
        v81 = v38;
        sub_16EA500(a1, 92, 41);
        v38 = v81;
      }
      if ( v38 <= 9 )
        *(_QWORD *)(a1 + 8 * v38 + 144) = *(_QWORD *)(a1 + 40);
      if ( *(_DWORD *)(a1 + 16) )
      {
        v49 = *(char **)a1;
        v9 = *(char **)(a1 + 8);
        if ( (unsigned __int64)v9 <= *(_QWORD *)a1 || v9 <= v49 + 1 )
          goto LABEL_90;
        if ( *v49 != 92 )
          goto LABEL_88;
      }
      else
      {
        v45 = *(_QWORD *)(a1 + 40);
        v46 = *(_QWORD *)(a1 + 32);
        if ( v45 >= v46 )
        {
          v12 = (v46 + 1) / 2;
          v47 = v12 + ((v46 + 1 + ((unsigned __int64)(v46 + 1) >> 63)) & 0xFFFFFFFFFFFFFFFELL);
          if ( v46 < v47 )
          {
            v82 = v38;
            sub_16E90A0(a1, v47, v45, v10, v12, a6);
            v45 = *(_QWORD *)(a1 + 40);
            v38 = v82;
          }
        }
        v48 = *(_QWORD *)(a1 + 24);
        *(_QWORD *)(a1 + 40) = v45 + 1;
        *(_QWORD *)(v48 + 8 * v45) = v38 | 0x70000000;
        v49 = *(char **)a1;
        v9 = *(char **)(a1 + 8);
        if ( *(_QWORD *)a1 >= (unsigned __int64)v9 || v9 <= v49 + 1 || *v49 != 92 )
        {
LABEL_88:
          if ( !*(_DWORD *)(a1 + 16) )
            *(_DWORD *)(a1 + 16) = 8;
LABEL_90:
          *(_QWORD *)a1 = &unk_4FA17D0;
          v11 = *(_QWORD *)(a1 + 40);
          *(_QWORD *)(a1 + 8) = &unk_4FA17D0;
LABEL_29:
          v9 = (char *)&unk_4FA17D0;
          result = (char *)&unk_4FA17D0;
          goto LABEL_14;
        }
      }
      if ( v49[1] != 41 )
        goto LABEL_88;
      result = v49 + 2;
      *(_QWORD *)a1 = result;
      goto LABEL_9;
    }
    if ( v14 > 296 )
    {
      if ( v14 == 379 )
      {
        if ( !*(_DWORD *)(a1 + 16) )
          *(_DWORD *)(a1 + 16) = 13;
        goto LABEL_28;
      }
      if ( v14 > 379 )
      {
        if ( v14 == 381 )
          goto LABEL_26;
LABEL_32:
        sub_16EA3B0(a1, (unsigned int)(char)v14, (__int64)v16, v10, v12, a6);
LABEL_33:
        result = *(char **)a1;
        v9 = *(char **)(a1 + 8);
        goto LABEL_9;
      }
      if ( v14 == 297 )
      {
LABEL_26:
        if ( !*(_DWORD *)(a1 + 16) )
          *(_DWORD *)(a1 + 16) = 8;
LABEL_28:
        *(_QWORD *)a1 = &unk_4FA17D0;
        *(_QWORD *)(a1 + 8) = &unk_4FA17D0;
        goto LABEL_29;
      }
      if ( (unsigned int)(v14 - 305) > 8 )
        goto LABEL_32;
      v17 = *(_DWORD *)(a1 + 16);
      LODWORD(v12) = (v12 & 0xFFFFFEFF) - 48;
      v18 = (int)v12;
      v19 = *(_QWORD *)(a1 + 8LL * (int)v12 + 144);
      if ( v19 )
      {
        if ( !v17 )
        {
          v20 = *(_QWORD *)(a1 + 32);
          v21 = v11;
          if ( v20 <= v11 )
          {
            v70 = (v20 + 1) / 2 + ((v20 + 1 + ((unsigned __int64)(v20 + 1) >> 63)) & 0xFFFFFFFFFFFFFFFELL);
            if ( v20 < v70 )
            {
              v77 = (int)v12;
              v84 = v12;
              sub_16E90A0(a1, v70, v11, v10, v12, a6);
              v21 = *(_QWORD *)(a1 + 40);
              v18 = v77;
              LODWORD(v12) = v84;
            }
          }
          v22 = *(_QWORD *)(a1 + 24);
          *(_QWORD *)(a1 + 40) = v21 + 1;
          *(_QWORD *)(v22 + 8 * v21) = (int)(v12 | 0x38000000);
          v19 = *(_QWORD *)(a1 + 8 * v18 + 144);
        }
        v79 = v12;
        sub_16E9110((_QWORD *)a1, *(_QWORD *)(a1 + 8 * v18 + 64) + 1LL, v19, v10, v12, a6);
        if ( !*(_DWORD *)(a1 + 16) )
        {
          v23 = *(_QWORD *)(a1 + 40);
          v24 = *(_QWORD *)(a1 + 32);
          v25 = v79;
          if ( v23 >= v24 )
          {
            v69 = (v24 + 1) / 2 + ((v24 + 1 + ((unsigned __int64)(v24 + 1) >> 63)) & 0xFFFFFFFFFFFFFFFELL);
            if ( v24 < v69 )
            {
              sub_16E90A0(a1, v69, v24, v10, v79, a6);
              v23 = *(_QWORD *)(a1 + 40);
              v25 = v79;
            }
          }
          v26 = *(_QWORD *)(a1 + 24);
          LODWORD(v12) = v25 | 0x40000000;
          *(_QWORD *)(a1 + 40) = v23 + 1;
          *(_QWORD *)(v26 + 8 * v23) = (int)v12;
        }
      }
      else
      {
        if ( !v17 )
          *(_DWORD *)(a1 + 16) = 6;
        *(_QWORD *)a1 = &unk_4FA17D0;
        *(_QWORD *)(a1 + 8) = &unk_4FA17D0;
      }
      *(_DWORD *)(*(_QWORD *)(a1 + 56) + 120LL) = 1;
      result = *(char **)a1;
      v9 = *(char **)(a1 + 8);
      goto LABEL_9;
    }
LABEL_6:
    if ( v14 != 46 )
    {
      if ( v14 == 91 )
      {
        sub_16E97A0(a1, (__int64)v9, (__int64)v16, v10, v12, a6);
        result = *(char **)a1;
        v9 = *(char **)(a1 + 8);
        goto LABEL_9;
      }
      if ( v14 == 42 && !v13 )
      {
        if ( !*(_DWORD *)(a1 + 16) )
          *(_DWORD *)(a1 + 16) = 13;
        *(_QWORD *)a1 = &unk_4FA17D0;
        *(_QWORD *)(a1 + 8) = &unk_4FA17D0;
      }
      goto LABEL_32;
    }
    if ( (*(_BYTE *)(*(_QWORD *)(a1 + 56) + 40LL) & 8) != 0 )
    {
      v68 = *(char **)a1;
      v9 = *(char **)(a1 + 8);
      *(_QWORD *)a1 = v87;
      *(_QWORD *)(a1 + 8) = (char *)v87 + 3;
      v76 = v68;
      v87[0] = 6097502;
      sub_16E97A0(a1, (__int64)v9, (__int64)v87 + 3, v10, v12, a6);
      result = v76;
      *(_QWORD *)a1 = v76;
      *(_QWORD *)(a1 + 8) = v9;
    }
    else
    {
      if ( *(_DWORD *)(a1 + 16) )
        goto LABEL_33;
      v34 = *(_QWORD *)(a1 + 32);
      v35 = v11;
      if ( v34 <= v11 )
      {
        v36 = (v34 + 1) / 2 + ((v34 + 1 + ((unsigned __int64)(v34 + 1) >> 63)) & 0xFFFFFFFFFFFFFFFELL);
        if ( v34 < v36 )
        {
          sub_16E90A0(a1, v36, v34, v10, v12, a6);
          v35 = *(_QWORD *)(a1 + 40);
        }
      }
      v37 = *(_QWORD *)(a1 + 24);
      *(_QWORD *)(a1 + 40) = v35 + 1;
      *(_QWORD *)(v37 + 8 * v35) = 671088640;
      result = *(char **)a1;
      v9 = *(char **)(a1 + 8);
    }
LABEL_9:
    if ( v9 > result )
      break;
LABEL_13:
    v11 = *(_QWORD *)(a1 + 40);
LABEL_14:
    LODWORD(v12) = v14 == 36;
LABEL_15:
    v13 = 0;
    if ( result >= v9 )
      goto LABEL_48;
  }
  v15 = result + 1;
  if ( *result == 42 )
  {
    a6 = *(_DWORD *)(a1 + 16);
    v33 = *(_QWORD *)(a1 + 40);
    *(_QWORD *)a1 = v15;
    if ( !a6 )
    {
      sub_16E9180((_QWORD *)a1, 1207959552, v33 - v11 + 1, v11, v12, 0);
      v33 = *(_QWORD *)(a1 + 40);
      v50 = *(_DWORD *)(a1 + 16);
      v51 = v33 - v11;
      if ( !v50 )
      {
        v52 = *(_QWORD *)(a1 + 32);
        if ( v33 >= v52 )
        {
          v53 = (v52 + 1) / 2 + ((v52 + 1 + ((unsigned __int64)(v52 + 1) >> 63)) & 0xFFFFFFFFFFFFFFFELL);
          if ( v52 < v53 )
          {
            sub_16E90A0(a1, v53, v33, v52, 0, a6);
            v33 = *(_QWORD *)(a1 + 40);
          }
        }
        v54 = *(_QWORD *)(a1 + 24);
        *(_QWORD *)(a1 + 40) = v33 + 1;
        *(_QWORD *)(v54 + 8 * v33) = v51 | 0x50000000;
        v10 = *(_QWORD *)(a1 + 40);
        if ( *(_DWORD *)(a1 + 16) )
        {
          result = *(char **)a1;
          v9 = *(char **)(a1 + 8);
          v11 = *(_QWORD *)(a1 + 40);
        }
        else
        {
          sub_16E9180((_QWORD *)a1, 1476395008, *(_QWORD *)(a1 + 40) - v11 + 1, v11, v50, a6);
          v55 = *(_QWORD *)(a1 + 40);
          LODWORD(v12) = *(_DWORD *)(a1 + 16);
          v56 = v55 - v11;
          if ( !(_DWORD)v12 )
          {
            v57 = *(_QWORD *)(a1 + 32);
            if ( v55 >= v57 )
            {
              v58 = (v57 + 1) / 2 + ((v57 + 1 + ((unsigned __int64)(v57 + 1) >> 63)) & 0xFFFFFFFFFFFFFFFELL);
              if ( v57 < v58 )
              {
                v83 = *(_DWORD *)(a1 + 16);
                sub_16E90A0(a1, v58, v55, v57, 0, a6);
                v55 = *(_QWORD *)(a1 + 40);
                LODWORD(v12) = v83;
              }
            }
            v10 = *(_QWORD *)(a1 + 24);
            *(_QWORD *)(a1 + 40) = v55 + 1;
            *(_QWORD *)(v10 + 8 * v55) = v56 | 0x60000000;
            result = *(char **)a1;
            v9 = *(char **)(a1 + 8);
            v11 = *(_QWORD *)(a1 + 40);
            goto LABEL_15;
          }
          result = *(char **)a1;
          v9 = *(char **)(a1 + 8);
          v11 = *(_QWORD *)(a1 + 40);
        }
        goto LABEL_59;
      }
      v15 = *(char **)a1;
      v9 = *(char **)(a1 + 8);
    }
    v11 = v33;
    result = v15;
LABEL_59:
    LODWORD(v12) = 0;
    goto LABEL_15;
  }
  if ( v9 <= v15 || *result != 92 || result[1] != 123 )
    goto LABEL_13;
  *(_QWORD *)a1 = result + 2;
  v59 = sub_16E8DE0((unsigned __int8 **)a1);
  v61 = *(_QWORD *)(a1 + 8);
  v62 = v59;
  v63 = *(char **)a1;
  v64 = v62;
  if ( *(_QWORD *)a1 < v61 && *v63 == 44 )
  {
    v64 = 256;
    *(_QWORD *)a1 = v63 + 1;
    if ( v61 > (unsigned __int64)(v63 + 1) && (unsigned int)(unsigned __int8)v63[1] - 48 <= 9 )
    {
      v85 = v62;
      v75 = sub_16E8DE0((unsigned __int8 **)a1);
      v62 = v85;
      v64 = v75;
      if ( v85 > v75 )
      {
        if ( !*(_DWORD *)(a1 + 16) )
          *(_DWORD *)(a1 + 16) = 10;
        *(_QWORD *)a1 = &unk_4FA17D0;
        *(_QWORD *)(a1 + 8) = &unk_4FA17D0;
      }
    }
  }
  sub_16E9280(a1, v11, v62, v64, v60);
  v65 = *(char **)a1;
  v9 = *(char **)(a1 + 8);
  if ( *(_QWORD *)a1 >= (unsigned __int64)v9 )
    goto LABEL_110;
  v66 = v65 + 1;
  if ( v9 <= v65 + 1 )
    goto LABEL_110;
  if ( *v65 == 92 && v65[1] == 125 )
  {
    result = v65 + 2;
    v11 = *(_QWORD *)(a1 + 40);
    LODWORD(v12) = 0;
    *(_QWORD *)a1 = result;
    goto LABEL_15;
  }
  if ( v9 > v66 )
  {
    while ( *(v66 - 1) != 92 || *v66 != 125 )
    {
      *(_QWORD *)a1 = v66;
      v65 = v66++;
      if ( v9 <= v66 )
        goto LABEL_110;
    }
    if ( v9 > v65 )
    {
      if ( !*(_DWORD *)(a1 + 16) )
        *(_DWORD *)(a1 + 16) = 10;
      goto LABEL_112;
    }
  }
LABEL_110:
  if ( !*(_DWORD *)(a1 + 16) )
    *(_DWORD *)(a1 + 16) = 9;
LABEL_112:
  result = (char *)&unk_4FA17D0;
  v67 = *(_QWORD *)(a1 + 40);
  *(_QWORD *)a1 = &unk_4FA17D0;
  *(_QWORD *)(a1 + 8) = &unk_4FA17D0;
  if ( v78 != v67 )
    return result;
LABEL_113:
  if ( !*(_DWORD *)(a1 + 16) )
    *(_DWORD *)(a1 + 16) = 14;
  *(_QWORD *)a1 = &unk_4FA17D0;
  *(_QWORD *)(a1 + 8) = &unk_4FA17D0;
  return (char *)&unk_4FA17D0;
}
