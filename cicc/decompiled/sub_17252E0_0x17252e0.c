// Function: sub_17252E0
// Address: 0x17252e0
//
__int64 __fastcall sub_17252E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  char v4; // al
  __int64 *v6; // r13
  unsigned __int8 v7; // al
  unsigned int v8; // r14d
  bool v9; // al
  __int64 v10; // rax
  __int64 v11; // rax
  unsigned int v12; // r13d
  __int64 v13; // rax
  __int64 v14; // r13
  unsigned int v15; // r14d
  bool v16; // al
  __int64 v17; // rax
  unsigned int v18; // r13d
  unsigned int v19; // r15d
  int v20; // r14d
  __int64 v21; // rax
  char v22; // dl
  unsigned int v23; // r15d
  int v24; // r14d
  __int64 v25; // rax
  char v26; // dl
  bool v27; // al
  int v28; // [rsp+Ch] [rbp-34h]
  int v29; // [rsp+Ch] [rbp-34h]

  v4 = *(_BYTE *)(a2 + 16);
  if ( v4 == 37 )
  {
    v6 = *(__int64 **)(a2 - 48);
    v7 = *((_BYTE *)v6 + 16);
    if ( v7 == 13 )
    {
      v8 = *((_DWORD *)v6 + 8);
      if ( v8 <= 0x40 )
        v9 = v6[3] == 0;
      else
        v9 = v8 == (unsigned int)sub_16A57B0((__int64)(v6 + 3));
    }
    else
    {
      if ( *(_BYTE *)(*v6 + 8) != 16 || v7 > 0x10u )
        return 0;
      v11 = sub_15A1020(*(_BYTE **)(a2 - 48), a2, *v6, a4);
      if ( !v11 || *(_BYTE *)(v11 + 16) != 13 )
      {
        v19 = 0;
        v20 = *(_QWORD *)(*v6 + 32);
        if ( v20 )
        {
          do
          {
            v21 = sub_15A0A60((__int64)v6, v19);
            if ( !v21 )
              return 0;
            v22 = *(_BYTE *)(v21 + 16);
            if ( v22 != 9 )
            {
              if ( v22 != 13 )
                return 0;
              if ( *(_DWORD *)(v21 + 32) <= 0x40u )
              {
                if ( *(_QWORD *)(v21 + 24) )
                  return 0;
              }
              else
              {
                v28 = *(_DWORD *)(v21 + 32);
                if ( v28 != (unsigned int)sub_16A57B0(v21 + 24) )
                  return 0;
              }
            }
          }
          while ( v20 != ++v19 );
        }
LABEL_9:
        v10 = *(_QWORD *)(a2 - 24);
        if ( v10 )
          goto LABEL_10;
        return 0;
      }
      v12 = *(_DWORD *)(v11 + 32);
      if ( v12 <= 0x40 )
        v9 = *(_QWORD *)(v11 + 24) == 0;
      else
        v9 = v12 == (unsigned int)sub_16A57B0(v11 + 24);
    }
    if ( !v9 )
      return 0;
    goto LABEL_9;
  }
  if ( v4 != 5 || *(_WORD *)(a2 + 18) != 13 )
    return 0;
  v13 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  v14 = *(_QWORD *)(a2 - 24 * v13);
  if ( *(_BYTE *)(v14 + 16) != 13 )
  {
    if ( *(_BYTE *)(*(_QWORD *)v14 + 8LL) != 16 )
      return 0;
    v17 = sub_15A1020(
            *(_BYTE **)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)),
            a2,
            -3LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF),
            4 * v13);
    if ( !v17 || *(_BYTE *)(v17 + 16) != 13 )
    {
      v23 = 0;
      v24 = *(_QWORD *)(*(_QWORD *)v14 + 32LL);
      if ( v24 )
      {
        while ( 1 )
        {
          v25 = sub_15A0A60(v14, v23);
          if ( !v25 )
            return 0;
          v26 = *(_BYTE *)(v25 + 16);
          if ( v26 != 9 )
          {
            if ( v26 != 13 )
              return 0;
            if ( *(_DWORD *)(v25 + 32) <= 0x40u )
            {
              v27 = *(_QWORD *)(v25 + 24) == 0;
            }
            else
            {
              v29 = *(_DWORD *)(v25 + 32);
              v27 = v29 == (unsigned int)sub_16A57B0(v25 + 24);
            }
            if ( !v27 )
              return 0;
          }
          if ( v24 == ++v23 )
            goto LABEL_21;
        }
      }
      goto LABEL_21;
    }
    v18 = *(_DWORD *)(v17 + 32);
    if ( v18 <= 0x40 )
      v16 = *(_QWORD *)(v17 + 24) == 0;
    else
      v16 = v18 == (unsigned int)sub_16A57B0(v17 + 24);
    goto LABEL_20;
  }
  v15 = *(_DWORD *)(v14 + 32);
  if ( v15 > 0x40 )
  {
    v16 = v15 == (unsigned int)sub_16A57B0(v14 + 24);
LABEL_20:
    if ( v16 )
    {
LABEL_21:
      v13 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
      goto LABEL_22;
    }
    return 0;
  }
  if ( *(_QWORD *)(v14 + 24) )
    return 0;
LABEL_22:
  v10 = *(_QWORD *)(a2 + 24 * (1 - v13));
  if ( !v10 )
    return 0;
LABEL_10:
  **(_QWORD **)(a1 + 8) = v10;
  return 1;
}
