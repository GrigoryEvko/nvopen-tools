// Function: sub_1ACF0B0
// Address: 0x1acf0b0
//
__int64 __fastcall sub_1ACF0B0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v6; // r15
  __int64 v7; // r12
  unsigned __int8 v8; // al
  __int64 v9; // rsi
  __int64 v10; // rdx
  unsigned int v11; // eax
  unsigned int v12; // edx
  unsigned int v13; // eax
  int v14; // esi
  __int64 v16; // rax
  unsigned int v17; // r12d
  bool v18; // al
  __int64 v19; // rax
  unsigned int v20; // eax
  unsigned int v21; // edx
  unsigned int v22; // eax
  int v23; // esi
  bool v24; // zf
  __int64 v25; // rdx
  __int64 v26; // r12
  char v27; // al
  int v28; // eax
  __int64 *v29; // rax
  char v30; // dl
  unsigned __int64 v31; // r12
  unsigned __int64 v32; // rdx
  unsigned __int64 v33; // rax
  unsigned __int64 v34; // rdx
  __int64 *v35; // rdi
  unsigned int v36; // r8d
  __int64 *v37; // rsi
  __int64 v38; // rsi
  __int64 v39; // rdx
  char v40; // si
  __int64 v41; // rdx
  __int64 v42; // rax
  int v43; // eax
  bool v44; // al
  __int64 v45; // rax
  __int64 v46; // rax
  int v47; // [rsp+4h] [rbp-3Ch]
  __int64 v48; // [rsp+8h] [rbp-38h]
  __int64 v49; // [rsp+8h] [rbp-38h]
  __int64 v50; // [rsp+8h] [rbp-38h]

  if ( *(_BYTE *)(a1 + 16) == 3 && (*(_BYTE *)(a1 + 80) & 2) != 0 )
    *(_DWORD *)(a2 + 4) = 2;
  v6 = *(_QWORD *)(a1 + 8);
  if ( !v6 )
    return 0;
  while ( 1 )
  {
    while ( 1 )
    {
      v7 = (__int64)sub_1648700(v6);
      v8 = *(_BYTE *)(v7 + 16);
      if ( v8 == 5 )
      {
        *(_BYTE *)(a2 + 25) = 1;
        if ( *(_BYTE *)(*(_QWORD *)v7 + 8LL) != 15 )
          return 1;
LABEL_7:
        if ( (unsigned __int8)sub_1ACF0B0(v7, a2, a3) )
          return 1;
        goto LABEL_8;
      }
      if ( v8 <= 0x17u )
      {
        *(_BYTE *)(a2 + 25) = 1;
        if ( v8 > 0x10u || !(unsigned __int8)sub_1ACF050(v7) )
          return 1;
        goto LABEL_8;
      }
      if ( !*(_BYTE *)(a2 + 24) )
      {
        v9 = *(_QWORD *)(*(_QWORD *)(v7 + 40) + 56LL);
        v10 = *(_QWORD *)(a2 + 16);
        if ( v10 )
        {
          if ( v10 != v9 )
          {
            *(_BYTE *)(a2 + 24) = 1;
            v8 = *(_BYTE *)(v7 + 16);
          }
        }
        else
        {
          *(_QWORD *)(a2 + 16) = v9;
          v8 = *(_BYTE *)(v7 + 16);
        }
      }
      if ( v8 != 54 )
        break;
      *(_BYTE *)(a2 + 1) = 1;
      v11 = *(unsigned __int16 *)(v7 + 18);
      if ( (v11 & 1) != 0 )
        return 1;
      v12 = *(_DWORD *)(a2 + 28);
      v13 = (v11 >> 7) & 7;
      if ( v12 != 4 || (v14 = 6, v13 != 5) )
      {
        if ( v13 != 4 || (v14 = 6, v12 != 5) )
        {
          if ( v12 >= v13 )
            v13 = *(_DWORD *)(a2 + 28);
          v14 = v13;
        }
      }
      *(_DWORD *)(a2 + 28) = v14;
      v6 = *(_QWORD *)(v6 + 8);
      if ( !v6 )
        return 0;
    }
    if ( v8 == 55 )
    {
      v19 = *(_QWORD *)(v7 - 48);
      if ( a1 == v19 && v19 )
        return 1;
      v20 = *(unsigned __int16 *)(v7 + 18);
      if ( (v20 & 1) != 0 )
        return 1;
      v21 = *(_DWORD *)(a2 + 28);
      v22 = (v20 >> 7) & 7;
      if ( v21 != 4 || (v23 = 6, v22 != 5) )
      {
        if ( v22 != 4 || (v23 = 6, v21 != 5) )
        {
          if ( v21 >= v22 )
            v22 = *(_DWORD *)(a2 + 28);
          v23 = v22;
        }
      }
      v24 = *(_DWORD *)(a2 + 4) == 3;
      *(_DWORD *)(a2 + 28) = v23;
      if ( !v24 )
      {
        v25 = *(_QWORD *)(v7 - 24);
        if ( *(_BYTE *)(v25 + 16) != 3 )
          goto LABEL_51;
        v26 = *(_QWORD *)(v7 - 48);
        if ( *(_BYTE *)(v26 + 16) <= 0x10u )
        {
          v48 = v25;
          v27 = sub_1593E50(v26);
          v25 = v48;
          if ( v27 )
            return 1;
        }
        v49 = v25;
        if ( sub_15E4F60(v25) || v26 != *(_QWORD *)(v49 - 24) )
        {
          v28 = *(_DWORD *)(a2 + 4);
          if ( *(_BYTE *)(v26 + 16) == 54 )
          {
            v38 = *(_QWORD *)(v26 - 24);
            if ( v38 )
            {
              if ( v49 == v38 )
              {
                if ( v28 > 0 )
                  goto LABEL_8;
LABEL_66:
                *(_DWORD *)(a2 + 4) = 1;
                goto LABEL_8;
              }
            }
          }
          if ( v28 <= 1 )
          {
            *(_DWORD *)(a2 + 4) = 2;
            *(_QWORD *)(a2 + 8) = v26;
            goto LABEL_8;
          }
          if ( v28 == 2 && *(_QWORD *)(a2 + 8) == v26 )
            goto LABEL_8;
          goto LABEL_51;
        }
        if ( *(int *)(a2 + 4) <= 0 )
          goto LABEL_66;
      }
      goto LABEL_8;
    }
    if ( v8 == 71 || v8 == 56 )
      goto LABEL_7;
    if ( (v8 & 0xFD) == 0x4D )
    {
      v29 = *(__int64 **)(a3 + 8);
      if ( *(__int64 **)(a3 + 16) != v29 )
        goto LABEL_62;
      v35 = &v29[*(unsigned int *)(a3 + 28)];
      v36 = *(_DWORD *)(a3 + 28);
      if ( v29 != v35 )
      {
        v37 = 0;
        while ( v7 != *v29 )
        {
          if ( *v29 == -2 )
            v37 = v29;
          if ( v35 == ++v29 )
          {
            if ( !v37 )
              goto LABEL_104;
            *v37 = v7;
            --*(_DWORD *)(a3 + 32);
            ++*(_QWORD *)a3;
            goto LABEL_7;
          }
        }
        goto LABEL_8;
      }
LABEL_104:
      if ( v36 < *(_DWORD *)(a3 + 24) )
      {
        *(_DWORD *)(a3 + 28) = v36 + 1;
        *v35 = v7;
        ++*(_QWORD *)a3;
      }
      else
      {
LABEL_62:
        sub_16CCBA0(a3, v7);
        if ( !v30 )
          goto LABEL_8;
      }
      goto LABEL_7;
    }
    if ( (unsigned __int8)(v8 - 75) <= 1u )
    {
      *(_BYTE *)a2 = 1;
      goto LABEL_8;
    }
    if ( v8 != 78 )
      break;
    v39 = *(_QWORD *)(v7 - 24);
    if ( *(_BYTE *)(v39 + 16) || (v40 = *(_BYTE *)(v39 + 33), (v40 & 0x20) == 0) )
    {
LABEL_92:
      v31 = v7 | 4;
      goto LABEL_71;
    }
    if ( (*(_DWORD *)(v39 + 36) & 0xFFFFFFFD) != 0x85 )
    {
      if ( (v40 & 0x20) == 0 || *(_DWORD *)(v39 + 36) != 137 )
        goto LABEL_92;
      v16 = *(_QWORD *)(v7 + 24 * (3LL - (*(_DWORD *)(v7 + 20) & 0xFFFFFFF)));
      v17 = *(_DWORD *)(v16 + 32);
      if ( v17 <= 0x40 )
        v18 = *(_QWORD *)(v16 + 24) == 0;
      else
        v18 = v17 == (unsigned int)sub_16A57B0(v16 + 24);
      if ( !v18 )
        return 1;
LABEL_51:
      *(_DWORD *)(a2 + 4) = 3;
      goto LABEL_8;
    }
    v41 = *(_DWORD *)(v7 + 20) & 0xFFFFFFF;
    v42 = *(_QWORD *)(v7 + 24 * (3 - v41));
    if ( *(_DWORD *)(v42 + 32) <= 0x40u )
    {
      v44 = *(_QWORD *)(v42 + 24) == 0;
    }
    else
    {
      v47 = *(_DWORD *)(v42 + 32);
      v50 = *(_DWORD *)(v7 + 20) & 0xFFFFFFF;
      v43 = sub_16A57B0(v42 + 24);
      v41 = v50;
      v44 = v47 == v43;
    }
    if ( !v44 )
      return 1;
    v45 = *(_QWORD *)(v7 - 24 * v41);
    if ( v45 && a1 == v45 )
    {
      *(_DWORD *)(a2 + 4) = 3;
      v41 = *(_DWORD *)(v7 + 20) & 0xFFFFFFF;
    }
    v46 = *(_QWORD *)(v7 + 24 * (1 - v41));
    if ( a1 == v46 && v46 )
      goto LABEL_75;
LABEL_8:
    v6 = *(_QWORD *)(v6 + 8);
    if ( !v6 )
      return 0;
  }
  if ( v8 != 29 )
    return 1;
  v31 = v7 & 0xFFFFFFFFFFFFFFFBLL;
LABEL_71:
  v32 = v31 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (v31 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
  {
    v33 = v32 - 24;
    v34 = v32 - 72;
    if ( (v31 & 4) == 0 )
      v33 = v34;
    if ( v33 == v6 )
    {
LABEL_75:
      *(_BYTE *)(a2 + 1) = 1;
      goto LABEL_8;
    }
  }
  return 1;
}
