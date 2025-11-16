// Function: sub_10ABFB0
// Address: 0x10abfb0
//
__int64 __fastcall sub_10ABFB0(__int64 a1, char *a2)
{
  unsigned __int8 v2; // al
  __int64 v4; // rax
  __int64 v6; // rax
  __int64 v7; // rcx
  __int64 v8; // rdi
  __int64 v9; // rdx
  __int64 v10; // r12
  __int64 v11; // rcx
  __int64 v12; // r13
  __int16 v13; // ax
  int v14; // eax
  __int64 v15; // rdx
  _BYTE *v16; // rax

  v2 = *a2;
  if ( (unsigned __int8)*a2 <= 0x1Cu )
    return 0;
  if ( v2 == 85 )
  {
    v6 = *((_QWORD *)a2 - 4);
    if ( v6 )
    {
      if ( !*(_BYTE *)v6
        && *(_QWORD *)(v6 + 24) == *((_QWORD *)a2 + 10)
        && (*(_BYTE *)(v6 + 33) & 0x20) != 0
        && *(_DWORD *)(v6 + 36) == 366 )
      {
        v7 = *(_QWORD *)&a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
        if ( v7 )
        {
          v8 = *(_QWORD *)&a2[32 * (1LL - (*((_DWORD *)a2 + 1) & 0x7FFFFFF))];
          **(_QWORD **)a1 = v7;
          if ( *(_BYTE *)v8 == 17 )
          {
            **(_QWORD **)(a1 + 8) = v8 + 24;
            return 1;
          }
          v15 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v8 + 8) + 8LL) - 17;
          if ( (unsigned int)v15 <= 1 && *(_BYTE *)v8 <= 0x15u )
          {
            v16 = sub_AD7630(v8, *(unsigned __int8 *)(a1 + 16), v15);
            if ( v16 )
            {
              if ( *v16 == 17 )
              {
                **(_QWORD **)(a1 + 8) = v16 + 24;
                return 1;
              }
            }
          }
        }
      }
    }
  }
  else
  {
    if ( v2 != 86 )
      return 0;
    v4 = *((_QWORD *)a2 - 12);
    if ( *(_BYTE *)v4 != 82 )
      return 0;
    v9 = *((_QWORD *)a2 - 8);
    v10 = *(_QWORD *)(v4 - 64);
    v11 = *((_QWORD *)a2 - 4);
    v12 = *(_QWORD *)(v4 - 32);
    if ( v10 == v9 && v12 == v11 )
    {
      v13 = *(_WORD *)(v4 + 2);
      goto LABEL_19;
    }
    if ( v12 == v9 && v10 == v11 )
    {
      v13 = *(_WORD *)(v4 + 2);
      if ( v10 != v9 )
      {
        v14 = sub_B52870(v13 & 0x3F);
        goto LABEL_20;
      }
LABEL_19:
      v14 = v13 & 0x3F;
LABEL_20:
      if ( (unsigned int)(v14 - 36) <= 1 && v10 )
      {
        **(_QWORD **)a1 = v10;
        return sub_991580(a1 + 8, v12);
      }
    }
  }
  return 0;
}
