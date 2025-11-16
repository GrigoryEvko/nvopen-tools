// Function: sub_17373D0
// Address: 0x17373d0
//
__int64 __fastcall sub_17373D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v4; // eax
  __int64 v6; // rax
  __int64 v7; // r12
  unsigned int v8; // r14d
  bool v9; // al
  __int64 v10; // rax
  char v11; // r12
  __int64 v12; // rax
  _BYTE *v13; // r14
  unsigned __int8 v14; // al
  unsigned int v15; // r15d
  bool v16; // al
  __int64 v17; // rbx
  __int64 v18; // rax
  unsigned int v19; // r12d
  bool v20; // al
  __int64 v21; // rax
  unsigned int v22; // r12d
  int v23; // r12d
  unsigned int v24; // r15d
  __int64 v25; // rax
  char v26; // dl
  unsigned int v27; // r15d
  int v28; // r14d
  __int64 v29; // rax
  char v30; // dl
  bool v31; // al
  int v32; // [rsp+Ch] [rbp-34h]
  int v33; // [rsp+Ch] [rbp-34h]

  v4 = *(unsigned __int8 *)(a2 + 16);
  if ( (unsigned __int8)v4 <= 0x17u )
  {
    if ( (_BYTE)v4 != 5 || (unsigned int)*(unsigned __int16 *)(a2 + 18) - 23 > 1 )
      return 0;
    v6 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
    v7 = *(_QWORD *)(a2 - 24 * v6);
    if ( *(_BYTE *)(v7 + 16) == 13 )
    {
      v8 = *(_DWORD *)(v7 + 32);
      if ( v8 <= 0x40 )
      {
        if ( *(_QWORD *)(v7 + 24) != 1 )
          return 0;
LABEL_11:
        v10 = *(_QWORD *)(a2 + 24 * (1 - v6));
        if ( !v10 )
          return 0;
        goto LABEL_22;
      }
      v9 = v8 - 1 == (unsigned int)sub_16A57B0(v7 + 24);
    }
    else
    {
      if ( *(_BYTE *)(*(_QWORD *)v7 + 8LL) != 16 )
        return 0;
      v21 = sub_15A1020(
              *(_BYTE **)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)),
              a2,
              -3LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF),
              4 * v6);
      if ( !v21 || *(_BYTE *)(v21 + 16) != 13 )
      {
        v27 = 0;
        v28 = *(_QWORD *)(*(_QWORD *)v7 + 32LL);
        if ( v28 )
        {
          do
          {
            v29 = sub_15A0A60(v7, v27);
            if ( !v29 )
              return 0;
            v30 = *(_BYTE *)(v29 + 16);
            if ( v30 != 9 )
            {
              if ( v30 != 13 )
                return 0;
              if ( *(_DWORD *)(v29 + 32) <= 0x40u )
              {
                v31 = *(_QWORD *)(v29 + 24) == 1;
              }
              else
              {
                v33 = *(_DWORD *)(v29 + 32);
                v31 = v33 - 1 == (unsigned int)sub_16A57B0(v29 + 24);
              }
              if ( !v31 )
                return 0;
            }
          }
          while ( v28 != ++v27 );
        }
LABEL_10:
        v6 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
        goto LABEL_11;
      }
      v22 = *(_DWORD *)(v21 + 32);
      if ( v22 <= 0x40 )
        v9 = *(_QWORD *)(v21 + 24) == 1;
      else
        v9 = v22 - 1 == (unsigned int)sub_16A57B0(v21 + 24);
    }
    if ( !v9 )
      return 0;
    goto LABEL_10;
  }
  if ( (unsigned int)(v4 - 47) > 1 )
    return 0;
  v11 = *(_BYTE *)(a2 + 23) & 0x40;
  if ( v11 )
    v12 = *(_QWORD *)(a2 - 8);
  else
    v12 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
  v13 = *(_BYTE **)v12;
  v14 = *(_BYTE *)(*(_QWORD *)v12 + 16LL);
  if ( v14 == 13 )
  {
    v15 = *((_DWORD *)v13 + 8);
    if ( v15 <= 0x40 )
      v16 = *((_QWORD *)v13 + 3) == 1;
    else
      v16 = v15 - 1 == (unsigned int)sub_16A57B0((__int64)(v13 + 24));
    if ( !v16 )
      return 0;
  }
  else
  {
    if ( *(_BYTE *)(*(_QWORD *)v13 + 8LL) != 16 || v14 > 0x10u )
      return 0;
    v18 = sub_15A1020(v13, a2, *(_QWORD *)v13, a4);
    if ( v18 && *(_BYTE *)(v18 + 16) == 13 )
    {
      v19 = *(_DWORD *)(v18 + 32);
      if ( v19 <= 0x40 )
        v20 = *(_QWORD *)(v18 + 24) == 1;
      else
        v20 = v19 - 1 == (unsigned int)sub_16A57B0(v18 + 24);
      if ( !v20 )
        return 0;
    }
    else
    {
      v23 = *(_QWORD *)(*(_QWORD *)v13 + 32LL);
      if ( v23 )
      {
        v24 = 0;
        do
        {
          v25 = sub_15A0A60((__int64)v13, v24);
          if ( !v25 )
            return 0;
          v26 = *(_BYTE *)(v25 + 16);
          if ( v26 != 9 )
          {
            if ( v26 != 13 )
              return 0;
            if ( *(_DWORD *)(v25 + 32) <= 0x40u )
            {
              if ( *(_QWORD *)(v25 + 24) != 1 )
                return 0;
            }
            else
            {
              v32 = *(_DWORD *)(v25 + 32);
              if ( (unsigned int)sub_16A57B0(v25 + 24) != v32 - 1 )
                return 0;
            }
          }
        }
        while ( v23 != ++v24 );
      }
    }
    v11 = *(_BYTE *)(a2 + 23) & 0x40;
  }
  if ( v11 )
    v17 = *(_QWORD *)(a2 - 8);
  else
    v17 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
  v10 = *(_QWORD *)(v17 + 24);
  if ( !v10 )
    return 0;
LABEL_22:
  **(_QWORD **)(a1 + 8) = v10;
  return 1;
}
