// Function: sub_10AA0A0
// Address: 0x10aa0a0
//
__int64 __fastcall sub_10AA0A0(_QWORD **a1, char *a2)
{
  unsigned __int8 v2; // al
  __int64 v3; // rax
  __int64 v5; // rax
  __int64 v6; // rcx
  __int64 v7; // rax
  _QWORD *v8; // rdx
  __int64 v9; // rdx
  __int64 v10; // r12
  __int64 v11; // rcx
  __int64 v12; // r13
  __int16 v13; // ax
  int v14; // eax

  v2 = *a2;
  if ( (unsigned __int8)*a2 <= 0x1Cu )
    return 0;
  if ( v2 == 85 )
  {
    v5 = *((_QWORD *)a2 - 4);
    if ( v5
      && !*(_BYTE *)v5
      && *(_QWORD *)(v5 + 24) == *((_QWORD *)a2 + 10)
      && (*(_BYTE *)(v5 + 33) & 0x20) != 0
      && *(_DWORD *)(v5 + 36) == 329 )
    {
      v6 = *(_QWORD *)&a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
      v7 = *(_QWORD *)&a2[32 * (1LL - (*((_DWORD *)a2 + 1) & 0x7FFFFFF))];
      if ( v6 )
      {
        v8 = *a1;
        if ( v7 )
        {
          *v8 = v6;
          *a1[1] = v7;
          return 1;
        }
        *v8 = v6;
      }
      else if ( v7 )
      {
        **a1 = v7;
        return 0;
      }
    }
    return 0;
  }
  if ( v2 == 86 )
  {
    v3 = *((_QWORD *)a2 - 12);
    if ( *(_BYTE *)v3 == 82 )
    {
      v9 = *((_QWORD *)a2 - 8);
      v10 = *(_QWORD *)(v3 - 64);
      v11 = *((_QWORD *)a2 - 4);
      v12 = *(_QWORD *)(v3 - 32);
      if ( v10 == v9 && v12 == v11 )
      {
        v13 = *(_WORD *)(v3 + 2);
      }
      else
      {
        if ( v12 != v9 || v10 != v11 )
          return 0;
        v13 = *(_WORD *)(v3 + 2);
        if ( v10 != v9 )
        {
          v14 = sub_B52870(v13 & 0x3F);
LABEL_20:
          if ( (unsigned int)(v14 - 38) <= 1 )
          {
            if ( v10 )
            {
              **a1 = v10;
              if ( v12 )
              {
                *a1[1] = v12;
                return 1;
              }
            }
            else if ( v12 )
            {
              **a1 = v12;
              return 0;
            }
          }
          return 0;
        }
      }
      v14 = v13 & 0x3F;
      goto LABEL_20;
    }
  }
  return 0;
}
