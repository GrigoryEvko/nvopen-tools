// Function: sub_1024090
// Address: 0x1024090
//
__int64 __fastcall sub_1024090(__int64 a1, __int64 a2, int a3, __int64 a4)
{
  bool v7; // r13
  bool v8; // r15
  __int64 v9; // rdx
  unsigned __int8 v10; // al
  __int64 v11; // rdx
  _BYTE *v12; // rdi
  __int64 v13; // rax
  char *v14; // rax
  char v15; // dl
  __int64 v16; // rcx
  __int64 v17; // rdi
  __int64 v18; // rdx
  __int64 v19; // rsi
  __int16 v20; // si
  _BYTE *v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rdx
  __int64 v25; // rax
  int v26; // edx
  __int64 v27; // rcx
  __int64 v28; // rsi
  __int64 v29; // rdx
  __int64 v30; // rdi
  __int16 v31; // ax
  int v32; // eax
  __int64 v33; // rsi
  __int64 v34; // rdi
  __int16 v35; // ax
  int v36; // eax
  __int64 v37; // rdi
  __int64 v38; // rsi
  __int16 v39; // si
  __int64 v40; // rsi
  __int64 v41; // rdi
  __int16 v42; // dx
  __int64 v43; // rdi
  __int64 v44; // rsi
  __int16 v45; // si
  __int64 v46; // rsi
  __int64 v47; // rdi
  __int16 v48; // ax
  int v49; // eax
  __int64 v50; // rsi
  __int64 v51; // rcx
  __int64 v52; // rdx
  __int64 v53; // rdi
  __int16 v54; // dx
  _QWORD v56[10]; // [rsp+10h] [rbp-50h] BYREF

  v7 = a3 == 6;
  v8 = a3 == 7;
  if ( a3 != 6 && (unsigned int)(a3 - 8) > 1 && a3 != 7 && (unsigned int)(a3 - 12) > 3 )
    goto LABEL_24;
  v9 = *(_QWORD *)(a2 + 16);
  v10 = *(_BYTE *)a2;
  if ( v9 && !*(_QWORD *)(v9 + 8) && (unsigned __int8)(v10 - 82) <= 1u )
  {
    sub_B53900(a2);
    v22 = *(_BYTE **)(*(_QWORD *)(a2 + 16) + 24LL);
    if ( *v22 == 86 )
    {
      v26 = *(_DWORD *)(a4 + 16);
      *(_BYTE *)a1 = 1;
      *(_QWORD *)(a1 + 8) = v22;
      *(_DWORD *)(a1 + 16) = v26;
      *(_QWORD *)(a1 + 24) = 0;
      return a1;
    }
    v10 = *(_BYTE *)a2;
  }
  if ( v10 <= 0x1Cu )
    goto LABEL_24;
  if ( v10 == 85 )
  {
    v23 = *(_QWORD *)(a2 - 32);
    if ( v23 && !*(_BYTE *)v23 && *(_QWORD *)(v23 + 24) == *(_QWORD *)(a2 + 80) && (*(_BYTE *)(v23 + 33) & 0x20) != 0 )
      goto LABEL_34;
LABEL_24:
    *(_BYTE *)a1 = 0;
    *(_QWORD *)(a1 + 8) = a2;
    *(_DWORD *)(a1 + 16) = 0;
    *(_QWORD *)(a1 + 24) = 0;
    return a1;
  }
  if ( v10 != 86 )
    goto LABEL_24;
  v11 = (*(_BYTE *)(a2 + 7) & 0x40) != 0 ? *(_QWORD *)(a2 - 8) : a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  v12 = *(_BYTE **)v11;
  v13 = *(_QWORD *)(*(_QWORD *)v11 + 16LL);
  if ( !v13 || *(_QWORD *)(v13 + 8) || (unsigned __int8)(*v12 - 82) > 1u )
    goto LABEL_24;
  sub_B53900((__int64)v12);
  v10 = *(_BYTE *)a2;
  if ( *(_BYTE *)a2 <= 0x1Cu )
    goto LABEL_47;
  if ( v10 == 85 )
  {
    v23 = *(_QWORD *)(a2 - 32);
LABEL_34:
    if ( !v23 )
      goto LABEL_40;
    if ( !*(_BYTE *)v23
      && *(_QWORD *)(v23 + 24) == *(_QWORD *)(a2 + 80)
      && (*(_BYTE *)(v23 + 33) & 0x20) != 0
      && *(_DWORD *)(v23 + 36) == 366 )
    {
      goto LABEL_59;
    }
LABEL_37:
    if ( v23
      && !*(_BYTE *)v23
      && *(_QWORD *)(v23 + 24) == *(_QWORD *)(a2 + 80)
      && (*(_BYTE *)(v23 + 33) & 0x20) != 0
      && *(_DWORD *)(v23 + 36) == 365 )
    {
      goto LABEL_71;
    }
LABEL_40:
    v24 = *(_QWORD *)(a2 - 32);
    if ( v24
      && !*(_BYTE *)v24
      && *(_QWORD *)(v24 + 24) == *(_QWORD *)(a2 + 80)
      && (*(_BYTE *)(v24 + 33) & 0x20) != 0
      && *(_DWORD *)(v24 + 36) == 329 )
    {
      goto LABEL_88;
    }
    goto LABEL_43;
  }
  if ( v10 == 86 )
  {
    v14 = *(char **)(a2 - 96);
    v15 = *v14;
    if ( *v14 != 82 )
      goto LABEL_16;
    v27 = *(_QWORD *)(a2 - 64);
    v28 = *((_QWORD *)v14 - 8);
    v29 = *(_QWORD *)(a2 - 32);
    v30 = *((_QWORD *)v14 - 4);
    if ( v27 == v28 && v29 == v30 )
    {
      v31 = *((_WORD *)v14 + 1);
    }
    else
    {
      if ( v27 != v30 || v29 != v28 )
        goto LABEL_66;
      v31 = *((_WORD *)v14 + 1);
      if ( v27 != v28 )
      {
        v32 = sub_B52870(v31 & 0x3F);
        goto LABEL_58;
      }
    }
    v32 = v31 & 0x3F;
LABEL_58:
    if ( (unsigned int)(v32 - 36) <= 1 )
    {
LABEL_59:
      *(_QWORD *)(a1 + 8) = a2;
      *(_DWORD *)(a1 + 16) = 0;
      *(_QWORD *)(a1 + 24) = 0;
      *(_BYTE *)a1 = a3 == 8;
      return a1;
    }
    v10 = *(_BYTE *)a2;
    if ( *(_BYTE *)a2 <= 0x1Cu )
      goto LABEL_47;
    if ( v10 == 85 )
    {
      v23 = *(_QWORD *)(a2 - 32);
      goto LABEL_37;
    }
    if ( v10 != 86 )
      goto LABEL_80;
    v14 = *(char **)(a2 - 96);
    v15 = *v14;
    if ( *v14 != 82 )
      goto LABEL_16;
    v29 = *(_QWORD *)(a2 - 32);
    v27 = *(_QWORD *)(a2 - 64);
LABEL_66:
    v33 = *((_QWORD *)v14 - 8);
    v34 = *((_QWORD *)v14 - 4);
    if ( v27 == v33 && v29 == v34 )
    {
      v35 = *((_WORD *)v14 + 1);
    }
    else
    {
      if ( v27 != v34 || v29 != v33 )
        goto LABEL_84;
      v35 = *((_WORD *)v14 + 1);
      if ( v27 != v33 )
      {
        v36 = sub_B52870(v35 & 0x3F);
        goto LABEL_70;
      }
    }
    v36 = v35 & 0x3F;
LABEL_70:
    if ( (unsigned int)(v36 - 34) <= 1 )
    {
LABEL_71:
      *(_QWORD *)(a1 + 8) = a2;
      *(_DWORD *)(a1 + 16) = 0;
      *(_QWORD *)(a1 + 24) = 0;
      *(_BYTE *)a1 = a3 == 9;
      return a1;
    }
    v10 = *(_BYTE *)a2;
    if ( *(_BYTE *)a2 <= 0x1Cu )
      goto LABEL_47;
  }
LABEL_80:
  if ( v10 == 85 )
    goto LABEL_40;
  if ( v10 != 86 )
    goto LABEL_43;
  v14 = *(char **)(a2 - 96);
  v15 = *v14;
  if ( *v14 != 82 )
    goto LABEL_16;
  v29 = *(_QWORD *)(a2 - 32);
  v27 = *(_QWORD *)(a2 - 64);
LABEL_84:
  v37 = *((_QWORD *)v14 - 8);
  v38 = *((_QWORD *)v14 - 4);
  if ( v27 == v37 && v29 == v38 )
  {
    v39 = *((_WORD *)v14 + 1);
    goto LABEL_87;
  }
  if ( v27 != v38 || v29 != v37 )
    goto LABEL_99;
  v39 = *((_WORD *)v14 + 1);
  if ( v27 == v37 )
  {
LABEL_87:
    if ( (v39 & 0x3Fu) - 38 > 1 )
      goto LABEL_99;
LABEL_88:
    *(_BYTE *)a1 = v8;
    *(_QWORD *)(a1 + 8) = a2;
    *(_DWORD *)(a1 + 16) = 0;
    *(_QWORD *)(a1 + 24) = 0;
    return a1;
  }
  if ( (unsigned int)sub_B52870(*((_WORD *)v14 + 1) & 0x3F) - 38 <= 1 )
    goto LABEL_88;
  v10 = *(_BYTE *)a2;
  if ( *(_BYTE *)a2 <= 0x1Cu )
    goto LABEL_47;
LABEL_43:
  if ( v10 == 85 )
  {
    v25 = *(_QWORD *)(a2 - 32);
    if ( !v25
      || *(_BYTE *)v25
      || *(_QWORD *)(v25 + 24) != *(_QWORD *)(a2 + 80)
      || (*(_BYTE *)(v25 + 33) & 0x20) == 0
      || *(_DWORD *)(v25 + 36) != 330 )
    {
      goto LABEL_47;
    }
    goto LABEL_103;
  }
  if ( v10 != 86 )
    goto LABEL_47;
  v14 = *(char **)(a2 - 96);
  v15 = *v14;
  if ( *v14 != 82 )
    goto LABEL_16;
  v29 = *(_QWORD *)(a2 - 32);
  v27 = *(_QWORD *)(a2 - 64);
LABEL_99:
  v40 = *((_QWORD *)v14 - 8);
  v41 = *((_QWORD *)v14 - 4);
  if ( v27 == v40 && v29 == v41 )
  {
    v42 = *((_WORD *)v14 + 1);
LABEL_102:
    if ( (v42 & 0x3Fu) - 40 > 1 )
      goto LABEL_17;
LABEL_103:
    *(_BYTE *)a1 = v7;
    *(_QWORD *)(a1 + 8) = a2;
    *(_DWORD *)(a1 + 16) = 0;
    *(_QWORD *)(a1 + 24) = 0;
    return a1;
  }
  if ( v27 != v41 || v29 != v40 )
    goto LABEL_17;
  v42 = *((_WORD *)v14 + 1);
  if ( v27 == v40 )
    goto LABEL_102;
  if ( (unsigned int)sub_B52870(*((_WORD *)v14 + 1) & 0x3F) - 40 <= 1 )
    goto LABEL_103;
  if ( *(_BYTE *)a2 != 86 )
    goto LABEL_47;
  v14 = *(char **)(a2 - 96);
  v15 = *v14;
LABEL_16:
  if ( v15 != 83 )
    goto LABEL_17;
  v50 = *(_QWORD *)(a2 - 64);
  v51 = *((_QWORD *)v14 - 8);
  v52 = *(_QWORD *)(a2 - 32);
  v53 = *((_QWORD *)v14 - 4);
  if ( v50 == v51 && v52 == v53 )
  {
    v54 = *((_WORD *)v14 + 1);
LABEL_142:
    if ( (v54 & 0x3Fu) - 4 <= 1 )
      goto LABEL_22;
    goto LABEL_17;
  }
  if ( v50 != v53 || v52 != v51 )
    goto LABEL_17;
  v54 = *((_WORD *)v14 + 1);
  if ( v50 == v51 )
    goto LABEL_142;
  if ( (unsigned int)sub_B52870(*((_WORD *)v14 + 1) & 0x3F) - 4 <= 1 )
    goto LABEL_22;
  if ( *(_BYTE *)a2 != 86 )
    goto LABEL_47;
  v14 = *(char **)(a2 - 96);
LABEL_17:
  if ( *v14 != 83 )
    goto LABEL_47;
  v16 = *(_QWORD *)(a2 - 64);
  v17 = *((_QWORD *)v14 - 8);
  v18 = *(_QWORD *)(a2 - 32);
  v19 = *((_QWORD *)v14 - 4);
  if ( v16 == v17 && v18 == v19 )
  {
    v20 = *((_WORD *)v14 + 1);
    goto LABEL_21;
  }
  if ( v16 != v19 || v18 != v17 )
  {
LABEL_120:
    v43 = *((_QWORD *)v14 - 8);
    v44 = *((_QWORD *)v14 - 4);
    if ( v16 == v43 && v18 == v44 )
    {
      v45 = *((_WORD *)v14 + 1);
    }
    else
    {
      if ( v16 != v44 || v18 != v43 )
        goto LABEL_124;
      v45 = *((_WORD *)v14 + 1);
      if ( v16 != v43 )
      {
        if ( (unsigned int)sub_B52870(*((_WORD *)v14 + 1) & 0x3F) - 2 <= 1 )
          goto LABEL_129;
        if ( *(_BYTE *)a2 != 86 )
          goto LABEL_47;
        v14 = *(char **)(a2 - 96);
        if ( *v14 != 83 )
          goto LABEL_47;
        v18 = *(_QWORD *)(a2 - 32);
        v16 = *(_QWORD *)(a2 - 64);
LABEL_124:
        v46 = *((_QWORD *)v14 - 8);
        v47 = *((_QWORD *)v14 - 4);
        if ( v16 == v46 && v18 == v47 )
        {
          v48 = *((_WORD *)v14 + 1);
        }
        else
        {
          if ( v16 != v47 || v18 != v46 )
            goto LABEL_47;
          v48 = *((_WORD *)v14 + 1);
          if ( v16 != v46 )
          {
            v49 = sub_B52870(v48 & 0x3F);
LABEL_128:
            if ( (unsigned int)(v49 - 10) <= 1 )
              goto LABEL_129;
LABEL_47:
            v56[0] = 248;
            v56[1] = 0x100000000LL;
            if ( sub_1024060(v56, a2) )
              goto LABEL_22;
            v56[0] = 237;
            if ( !sub_1024060(v56, a2) )
            {
              v56[0] = 246;
              if ( sub_1024060(v56, a2) )
              {
                *(_QWORD *)(a1 + 8) = a2;
                *(_DWORD *)(a1 + 16) = 0;
                *(_QWORD *)(a1 + 24) = 0;
                *(_BYTE *)a1 = a3 == 14;
                return a1;
              }
              v56[0] = 235;
              if ( sub_1024060(v56, a2) )
              {
                *(_QWORD *)(a1 + 8) = a2;
                *(_DWORD *)(a1 + 16) = 0;
                *(_QWORD *)(a1 + 24) = 0;
                *(_BYTE *)a1 = a3 == 15;
                return a1;
              }
              goto LABEL_24;
            }
LABEL_129:
            *(_QWORD *)(a1 + 8) = a2;
            *(_DWORD *)(a1 + 16) = 0;
            *(_QWORD *)(a1 + 24) = 0;
            *(_BYTE *)a1 = a3 == 13;
            return a1;
          }
        }
        v49 = v48 & 0x3F;
        goto LABEL_128;
      }
    }
    if ( (v45 & 0x3Fu) - 2 <= 1 )
      goto LABEL_129;
    goto LABEL_124;
  }
  v20 = *((_WORD *)v14 + 1);
  if ( v16 != v17 )
  {
    if ( (unsigned int)sub_B52870(*((_WORD *)v14 + 1) & 0x3F) - 12 <= 1 )
      goto LABEL_22;
    if ( *(_BYTE *)a2 != 86 )
      goto LABEL_47;
    v14 = *(char **)(a2 - 96);
    if ( *v14 != 83 )
      goto LABEL_47;
    v18 = *(_QWORD *)(a2 - 32);
    v16 = *(_QWORD *)(a2 - 64);
    goto LABEL_120;
  }
LABEL_21:
  if ( (v20 & 0x3Fu) - 12 > 1 )
    goto LABEL_120;
LABEL_22:
  *(_QWORD *)(a1 + 8) = a2;
  *(_DWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 24) = 0;
  *(_BYTE *)a1 = a3 == 12;
  return a1;
}
