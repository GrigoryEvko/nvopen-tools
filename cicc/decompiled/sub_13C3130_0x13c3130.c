// Function: sub_13C3130
// Address: 0x13c3130
//
bool __fastcall sub_13C3130(__int64 a1, _QWORD *a2, _QWORD *a3)
{
  __int64 v5; // r13
  __int64 v6; // r12
  unsigned __int8 v7; // al
  __int64 v8; // r14
  unsigned __int8 v9; // al
  __int64 v10; // r14
  int v11; // edx
  __int64 v12; // rsi
  unsigned int v14; // ecx
  __int64 *v15; // rax
  __int64 v16; // r8
  __int64 v17; // r15
  bool v18; // al
  bool v19; // dl
  __int64 v20; // rdi
  int v21; // edx
  unsigned int v22; // edi
  __int64 *v23; // rcx
  __int64 v24; // r8
  __int64 v25; // rsi
  __int64 v26; // rdx
  int v27; // eax
  int v28; // ecx
  int v29; // r9d
  int v30; // r9d

  v5 = sub_14AD280(*a2, *(_QWORD *)(a1 + 8), 6);
  v6 = sub_14AD280(*a3, *(_QWORD *)(a1 + 8), 6);
  v7 = *(_BYTE *)(v5 + 16);
  if ( v7 <= 3u )
  {
    if ( *(_BYTE *)(v6 + 16) > 3u )
    {
      if ( !sub_13C3050(a1 + 24, v5) )
        goto LABEL_5;
    }
    else
    {
      v8 = a1 + 24;
      if ( !sub_13C3050(a1 + 24, v5) )
        goto LABEL_4;
      if ( sub_13C3050(a1 + 24, v6) )
      {
        if ( v5 != v6 )
          return 0;
        goto LABEL_5;
      }
    }
    if ( byte_4F98F20 )
      return 0;
    v25 = v5;
    v26 = v6;
    if ( !v5 )
      goto LABEL_5;
    goto LABEL_44;
  }
  v8 = a1 + 24;
  if ( *(_BYTE *)(v6 + 16) > 3u )
    goto LABEL_6;
LABEL_4:
  if ( sub_13C3050(v8, v6) )
  {
    if ( byte_4F98F20 )
      return 0;
    v25 = v6;
    v26 = v5;
LABEL_44:
    if ( (unsigned __int8)sub_13C1940(a1, v25, v26) )
      return 0;
  }
LABEL_5:
  v7 = *(_BYTE *)(v5 + 16);
LABEL_6:
  if ( v7 != 54
    || (v17 = *(_QWORD *)(v5 - 24), *(_BYTE *)(v17 + 16) != 3)
    || (v20 = a1 + 128, !sub_13C3050(a1 + 128, *(_QWORD *)(v5 - 24))) )
  {
    v9 = *(_BYTE *)(v6 + 16);
    if ( v9 > 0x17u )
    {
      if ( v9 != 54 )
      {
        v11 = *(_DWORD *)(a1 + 256);
        v12 = *(_QWORD *)(a1 + 240);
        if ( v11 )
          goto LABEL_18;
        goto LABEL_47;
      }
      v10 = *(_QWORD *)(v6 - 24);
      if ( *(_BYTE *)(v10 + 16) == 3 && sub_13C3050(a1 + 128, *(_QWORD *)(v6 - 24)) )
      {
        v11 = *(_DWORD *)(a1 + 256);
        if ( !v11 )
          goto LABEL_52;
        v12 = *(_QWORD *)(a1 + 240);
LABEL_19:
        v14 = (v11 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
        v15 = (__int64 *)(v12 + 16LL * v14);
        v16 = *v15;
        if ( v5 == *v15 )
        {
LABEL_20:
          v17 = v15[1];
          if ( v10 )
          {
LABEL_21:
            v18 = v17 != 0;
            v19 = v10 != 0;
            if ( v17 && v10 )
              goto LABEL_29;
            goto LABEL_23;
          }
          goto LABEL_38;
        }
        v27 = 1;
        while ( v16 != -8 )
        {
          v30 = v27 + 1;
          v14 = (v11 - 1) & (v27 + v14);
          v15 = (__int64 *)(v12 + 16LL * v14);
          v16 = *v15;
          if ( v5 == *v15 )
            goto LABEL_20;
          v27 = v30;
        }
        if ( !v10 )
        {
          v17 = 0;
LABEL_38:
          v18 = v17 != 0;
          if ( v11 )
          {
            v21 = v11 - 1;
            v22 = v21 & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
            v23 = (__int64 *)(v12 + 16LL * v22);
            v24 = *v23;
            if ( v6 == *v23 )
            {
LABEL_40:
              v10 = v23[1];
              goto LABEL_21;
            }
            v28 = 1;
            while ( v24 != -8 )
            {
              v29 = v28 + 1;
              v22 = v21 & (v28 + v22);
              v23 = (__int64 *)(v12 + 16LL * v22);
              v24 = *v23;
              if ( v6 == *v23 )
                goto LABEL_40;
              v28 = v29;
            }
          }
LABEL_48:
          v10 = 0;
          v19 = 0;
LABEL_23:
          if ( byte_4F98F20 && (v18 || v19) )
            return v17 == v10;
          return 1;
        }
LABEL_52:
        if ( byte_4F98F20 )
        {
          v17 = 0;
          return v17 == v10;
        }
        return 1;
      }
    }
    v11 = *(_DWORD *)(a1 + 256);
    if ( v11 )
    {
      v12 = *(_QWORD *)(a1 + 240);
LABEL_18:
      v10 = 0;
      goto LABEL_19;
    }
LABEL_47:
    v18 = 0;
    v17 = 0;
    goto LABEL_48;
  }
  if ( *(_BYTE *)(v6 + 16) != 54
    || (v10 = *(_QWORD *)(v6 - 24), *(_BYTE *)(v10 + 16) != 3)
    || !sub_13C3050(v20, *(_QWORD *)(v6 - 24)) )
  {
    v12 = *(_QWORD *)(a1 + 240);
    v11 = *(_DWORD *)(a1 + 256);
    goto LABEL_38;
  }
LABEL_29:
  if ( v10 != v17 )
    return 0;
  if ( !byte_4F98F20 )
    return 1;
  return v17 == v10;
}
