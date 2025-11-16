// Function: sub_F15980
// Address: 0xf15980
//
_BOOL8 __fastcall sub_F15980(_QWORD **a1, __int64 a2)
{
  __int64 v2; // rax
  _BOOL8 result; // rax
  char v5; // al
  char *v6; // rdx
  __int64 v7; // rcx
  unsigned __int8 v8; // cl
  unsigned __int64 v9; // rdi
  int v10; // ecx
  __int64 v11; // rcx
  _BYTE *v12; // rcx
  char *v13; // rax
  __int64 v14; // rdx
  unsigned __int8 v15; // cl
  int v16; // edx
  __int64 v17; // rdx
  _BYTE *v18; // rdx

  v2 = *(_QWORD *)(a2 + 16);
  if ( !v2 || *(_QWORD *)(v2 + 8) )
    return 0;
  v5 = *(_BYTE *)a2;
  if ( *(_BYTE *)a2 != 69 )
    goto LABEL_5;
  v13 = *(char **)(a2 - 32);
  v14 = *((_QWORD *)v13 + 2);
  if ( !v14 || *(_QWORD *)(v14 + 8) )
    return 0;
  v15 = *v13;
  if ( (unsigned __int8)*v13 <= 0x1Cu )
  {
    if ( v15 != 5 )
      return 0;
    v16 = *((unsigned __int16 *)v13 + 1);
    if ( (*((_WORD *)v13 + 1) & 0xFFF7) != 0x11 && (v16 & 0xFFFD) != 0xD )
      return 0;
  }
  else
  {
    if ( v15 > 0x36u || ((0x40540000000000uLL >> v15) & 1) == 0 )
      return 0;
    v16 = v15 - 29;
  }
  if ( v16 != 13 )
    return 0;
  if ( (v13[1] & 4) == 0 )
    return 0;
  v17 = *((_QWORD *)v13 - 8);
  if ( !v17 )
    return 0;
  **a1 = v17;
  v18 = (_BYTE *)*((_QWORD *)v13 - 4);
  if ( *v18 != 17 )
  {
    v5 = *(_BYTE *)a2;
LABEL_5:
    if ( v5 != 68 )
      return 0;
    result = sub_B44910(a2);
    if ( !result )
      return 0;
    v6 = *(char **)(a2 - 32);
    v7 = *((_QWORD *)v6 + 2);
    if ( !v7 || *(_QWORD *)(v7 + 8) )
      return 0;
    v8 = *v6;
    if ( (unsigned __int8)*v6 <= 0x1Cu )
    {
      if ( v8 != 5 )
        return 0;
      v10 = *((unsigned __int16 *)v6 + 1);
      if ( (*((_WORD *)v6 + 1) & 0xFFFD) != 0xD && (v10 & 0xFFF7) != 0x11 )
        return 0;
    }
    else
    {
      if ( v8 > 0x36u )
        return 0;
      v9 = 0x40540000000000uLL >> v8;
      v10 = v8 - 29;
      if ( (v9 & 1) == 0 )
        return 0;
    }
    if ( v10 == 13 && (v6[1] & 4) != 0 )
    {
      v11 = *((_QWORD *)v6 - 8);
      if ( v11 )
      {
        *a1[2] = v11;
        v12 = (_BYTE *)*((_QWORD *)v6 - 4);
        if ( *v12 == 17 )
        {
          *a1[3] = v12;
          return result;
        }
      }
    }
    return 0;
  }
  *a1[1] = v18;
  return 1;
}
