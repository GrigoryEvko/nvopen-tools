// Function: sub_10E3680
// Address: 0x10e3680
//
__int64 __fastcall sub_10E3680(__int64 a1, char *a2)
{
  unsigned int v2; // r12d
  unsigned __int8 v3; // al
  __int64 v5; // rax
  __int64 v7; // rax
  __int64 v8; // rcx
  __int64 v9; // rdi
  __int64 v10; // rdx
  __int64 v11; // r14
  __int64 v12; // rcx
  __int64 v13; // r13
  __int16 v14; // ax
  int v15; // eax
  __int64 v16; // rdx
  _BYTE *v17; // rax
  __int64 v18; // rdx
  _BYTE *v19; // rax

  v3 = *a2;
  if ( (unsigned __int8)*a2 <= 0x1Cu )
    return 0;
  if ( v3 == 85 )
  {
    v7 = *((_QWORD *)a2 - 4);
    if ( v7 )
    {
      if ( !*(_BYTE *)v7
        && *(_QWORD *)(v7 + 24) == *((_QWORD *)a2 + 10)
        && (*(_BYTE *)(v7 + 33) & 0x20) != 0
        && *(_DWORD *)(v7 + 36) == 330 )
      {
        v8 = *(_QWORD *)&a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
        if ( v8 )
        {
          v9 = *(_QWORD *)&a2[32 * (1LL - (*((_DWORD *)a2 + 1) & 0x7FFFFFF))];
          **(_QWORD **)a1 = v8;
          if ( *(_BYTE *)v9 == 17 )
          {
            v2 = 1;
            **(_QWORD **)(a1 + 8) = v9 + 24;
            return v2;
          }
          v16 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v9 + 8) + 8LL) - 17;
          if ( (unsigned int)v16 <= 1 && *(_BYTE *)v9 <= 0x15u )
          {
            v17 = sub_AD7630(v9, *(unsigned __int8 *)(a1 + 16), v16);
            if ( v17 )
            {
              if ( *v17 == 17 )
              {
                v2 = 1;
                **(_QWORD **)(a1 + 8) = v17 + 24;
                return v2;
              }
            }
          }
        }
      }
    }
  }
  else
  {
    if ( v3 != 86 )
      return 0;
    v5 = *((_QWORD *)a2 - 12);
    if ( *(_BYTE *)v5 != 82 )
      return 0;
    v10 = *((_QWORD *)a2 - 8);
    v11 = *(_QWORD *)(v5 - 64);
    v12 = *((_QWORD *)a2 - 4);
    v13 = *(_QWORD *)(v5 - 32);
    if ( v11 == v10 && v13 == v12 )
    {
      v14 = *(_WORD *)(v5 + 2);
      goto LABEL_18;
    }
    if ( v13 == v10 && v11 == v12 )
    {
      v14 = *(_WORD *)(v5 + 2);
      if ( v11 != v10 )
      {
        v15 = sub_B52870(v14 & 0x3F);
LABEL_19:
        LOBYTE(v2) = v11 != 0 && (unsigned int)(v15 - 40) <= 1;
        if ( (_BYTE)v2 )
        {
          **(_QWORD **)a1 = v11;
          if ( *(_BYTE *)v13 == 17 )
          {
            **(_QWORD **)(a1 + 8) = v13 + 24;
            return v2;
          }
          v18 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v13 + 8) + 8LL) - 17;
          if ( (unsigned int)v18 <= 1 && *(_BYTE *)v13 <= 0x15u )
          {
            v19 = sub_AD7630(v13, *(unsigned __int8 *)(a1 + 16), v18);
            if ( v19 )
            {
              if ( *v19 == 17 )
              {
                **(_QWORD **)(a1 + 8) = v19 + 24;
                return v2;
              }
            }
          }
        }
        return 0;
      }
LABEL_18:
      v15 = v14 & 0x3F;
      goto LABEL_19;
    }
  }
  return 0;
}
