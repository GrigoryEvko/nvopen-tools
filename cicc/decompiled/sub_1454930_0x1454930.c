// Function: sub_1454930
// Address: 0x1454930
//
bool __fastcall sub_1454930(__int64 a1, _QWORD *a2, _DWORD *a3)
{
  char v3; // al
  __int16 v5; // ax
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 v9; // rax
  unsigned int v10; // ebx
  __int64 v11; // rsi
  __int64 v12; // rax
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rax
  char v17; // al

  v3 = *(_BYTE *)(a1 + 16);
  switch ( v3 )
  {
    case 48:
      v9 = *(_QWORD *)(a1 - 48);
      if ( !v9 )
        return 0;
      *a2 = v9;
      v7 = *(_QWORD *)(a1 - 24);
      if ( *(_BYTE *)(v7 + 16) != 13 )
        goto LABEL_24;
      break;
    case 5:
      v5 = *(_WORD *)(a1 + 18);
      if ( v5 != 24 )
        goto LABEL_4;
      v14 = *(_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
      if ( !v14 )
        return 0;
      *a2 = v14;
      v7 = *(_QWORD *)(a1 + 24 * (1LL - (*(_DWORD *)(a1 + 20) & 0xFFFFFFF)));
      if ( *(_BYTE *)(v7 + 16) != 13 )
      {
LABEL_24:
        v3 = *(_BYTE *)(a1 + 16);
        if ( v3 != 49 )
        {
          if ( v3 == 5 )
          {
            v5 = *(_WORD *)(a1 + 18);
LABEL_4:
            if ( v5 != 25 )
            {
LABEL_5:
              if ( v5 == 23 )
              {
                v6 = *(_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
                if ( v6 )
                {
                  *a2 = v6;
                  v7 = *(_QWORD *)(a1 + 24 * (1LL - (*(_DWORD *)(a1 + 20) & 0xFFFFFFF)));
                  if ( *(_BYTE *)(v7 + 16) == 13 )
                  {
LABEL_8:
                    *a3 = 23;
                    goto LABEL_16;
                  }
                }
              }
              return 0;
            }
            v16 = *(_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
            if ( !v16 )
              return 0;
            *a2 = v16;
            v7 = *(_QWORD *)(a1 + 24 * (1LL - (*(_DWORD *)(a1 + 20) & 0xFFFFFFF)));
            if ( *(_BYTE *)(v7 + 16) != 13 )
              goto LABEL_32;
LABEL_12:
            *a3 = 25;
            goto LABEL_16;
          }
LABEL_26:
          if ( v3 != 47 )
            return 0;
          goto LABEL_27;
        }
LABEL_10:
        v8 = *(_QWORD *)(a1 - 48);
        if ( !v8 )
          return 0;
        *a2 = v8;
        v7 = *(_QWORD *)(a1 - 24);
        if ( *(_BYTE *)(v7 + 16) != 13 )
        {
LABEL_32:
          v17 = *(_BYTE *)(a1 + 16);
          if ( v17 != 47 )
          {
            if ( v17 != 5 )
              return 0;
            v5 = *(_WORD *)(a1 + 18);
            goto LABEL_5;
          }
LABEL_27:
          v15 = *(_QWORD *)(a1 - 48);
          if ( v15 )
          {
            *a2 = v15;
            v7 = *(_QWORD *)(a1 - 24);
            if ( *(_BYTE *)(v7 + 16) == 13 )
              goto LABEL_8;
          }
          return 0;
        }
        goto LABEL_12;
      }
      break;
    case 49:
      goto LABEL_10;
    default:
      goto LABEL_26;
  }
  *a3 = 24;
LABEL_16:
  v10 = *(_DWORD *)(v7 + 32);
  v11 = *(_QWORD *)(v7 + 24);
  v12 = 1LL << ((unsigned __int8)v10 - 1);
  if ( v10 > 0x40 )
  {
    if ( (*(_QWORD *)(v11 + 8LL * ((v10 - 1) >> 6)) & v12) != 0 )
      return 0;
    return v10 != (unsigned int)sub_16A57B0(v7 + 24);
  }
  else
  {
    if ( (v12 & v11) != 0 )
      return 0;
    return v11 != 0;
  }
}
