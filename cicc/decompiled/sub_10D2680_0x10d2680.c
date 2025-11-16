// Function: sub_10D2680
// Address: 0x10d2680
//
char __fastcall sub_10D2680(_QWORD **a1, char *a2)
{
  unsigned __int8 v2; // al
  __int64 v3; // rax
  char result; // al
  __int64 v5; // rax
  _BYTE *v6; // r13
  _BYTE *v7; // r12
  char v8; // al
  __int64 v9; // rsi
  unsigned __int8 *v10; // rdx
  unsigned __int8 *v11; // r12
  unsigned __int8 *v12; // rcx
  unsigned __int8 *v13; // r13
  __int16 v14; // ax
  int v15; // eax
  char v16; // al
  __int64 v17; // rsi
  __int64 v18; // rax
  __int64 v19; // rax

  v2 = *a2;
  if ( (unsigned __int8)*a2 <= 0x1Cu )
    return 0;
  if ( v2 != 85 )
  {
    if ( v2 != 86 )
      return 0;
    v3 = *((_QWORD *)a2 - 12);
    if ( *(_BYTE *)v3 != 82 )
      return 0;
    v10 = (unsigned __int8 *)*((_QWORD *)a2 - 8);
    v11 = *(unsigned __int8 **)(v3 - 64);
    v12 = (unsigned __int8 *)*((_QWORD *)a2 - 4);
    v13 = *(unsigned __int8 **)(v3 - 32);
    if ( v11 == v10 && v13 == v12 )
    {
      v14 = *(_WORD *)(v3 + 2);
    }
    else
    {
      if ( v13 != v10 || v11 != v12 )
        return 0;
      v14 = *(_WORD *)(v3 + 2);
      if ( v11 != v10 )
      {
        v15 = sub_B52870(v14 & 0x3F);
LABEL_23:
        if ( (unsigned int)(v15 - 38) <= 1 )
        {
          result = sub_996420(a1, 30, v11) & (v13 != 0);
          if ( result )
          {
            *a1[2] = v13;
            return result;
          }
          result = sub_996420(a1, 30, v13) & (v11 != 0);
          if ( result )
          {
            *a1[2] = v11;
            return result;
          }
        }
        return 0;
      }
    }
    v15 = v14 & 0x3F;
    goto LABEL_23;
  }
  v5 = *((_QWORD *)a2 - 4);
  if ( !v5
    || *(_BYTE *)v5
    || *(_QWORD *)(v5 + 24) != *((_QWORD *)a2 + 10)
    || (*(_BYTE *)(v5 + 33) & 0x20) == 0
    || *(_DWORD *)(v5 + 36) != 329 )
  {
    return 0;
  }
  v6 = *(_BYTE **)&a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
  v7 = *(_BYTE **)&a2[32 * (1LL - (*((_DWORD *)a2 + 1) & 0x7FFFFFF))];
  if ( *v6 == 59 )
  {
    v16 = sub_995B10(a1, *((_QWORD *)v6 - 8));
    v17 = *((_QWORD *)v6 - 4);
    if ( v16 && v17 )
    {
      *a1[1] = v17;
    }
    else
    {
      if ( !(unsigned __int8)sub_995B10(a1, v17) )
        goto LABEL_14;
      v18 = *((_QWORD *)v6 - 8);
      if ( !v18 )
        goto LABEL_14;
      *a1[1] = v18;
    }
    if ( v7 )
    {
      *a1[2] = v7;
      return 1;
    }
  }
LABEL_14:
  if ( *v7 == 59 )
  {
    v8 = sub_995B10(a1, *((_QWORD *)v7 - 8));
    v9 = *((_QWORD *)v7 - 4);
    if ( v8 && v9 )
    {
      *a1[1] = v9;
    }
    else
    {
      if ( !(unsigned __int8)sub_995B10(a1, v9) )
        return 0;
      v19 = *((_QWORD *)v7 - 8);
      if ( !v19 )
        return 0;
      *a1[1] = v19;
    }
    *a1[2] = v6;
    return 1;
  }
  return 0;
}
