// Function: sub_2B08CD0
// Address: 0x2b08cd0
//
char *__fastcall sub_2B08CD0(char *a1, char *a2, __int64 a3)
{
  char *result; // rax
  __int64 v5; // rdx
  __int64 v6; // rsi
  __int64 v7; // rdx
  char v8; // dl
  char *v9; // rsi
  bool v11; // cl
  bool v12; // cl
  char v13; // cl
  bool v14; // dl
  bool v15; // dl
  bool v16; // dl

  result = a2;
  v5 = a2 - a1;
  v6 = (a2 - a1) >> 5;
  v7 = v5 >> 3;
  if ( v6 > 0 )
  {
    v8 = *(_BYTE *)(a3 + 1);
    v9 = &a1[32 * v6];
    while ( !(v8 ? *(_DWORD *)a1 != 0 : *((_DWORD *)a1 + 1) != 0) )
    {
      if ( v8 )
        v11 = *((_DWORD *)a1 + 2) != 0;
      else
        v11 = *((_DWORD *)a1 + 3) != 0;
      if ( v11 )
        return a1 + 8;
      if ( v8 )
        v12 = *((_DWORD *)a1 + 4) != 0;
      else
        v12 = *((_DWORD *)a1 + 5) != 0;
      if ( v12 )
        return a1 + 16;
      if ( v8 )
      {
        if ( *((_DWORD *)a1 + 6) )
          return a1 + 24;
      }
      else if ( *((_DWORD *)a1 + 7) )
      {
        return a1 + 24;
      }
      a1 += 32;
      if ( a1 == v9 )
      {
        v7 = (result - a1) >> 3;
        goto LABEL_24;
      }
    }
    return a1;
  }
LABEL_24:
  if ( v7 == 2 )
  {
    v13 = *(_BYTE *)(a3 + 1);
    goto LABEL_38;
  }
  if ( v7 == 3 )
  {
    v13 = *(_BYTE *)(a3 + 1);
    if ( v13 )
      v15 = *(_DWORD *)a1 != 0;
    else
      v15 = *((_DWORD *)a1 + 1) != 0;
    if ( v15 )
      return a1;
    a1 += 8;
LABEL_38:
    if ( v13 )
      v16 = *(_DWORD *)a1 != 0;
    else
      v16 = *((_DWORD *)a1 + 1) != 0;
    if ( !v16 )
    {
      a1 += 8;
      goto LABEL_28;
    }
    return a1;
  }
  if ( v7 != 1 )
    return result;
  v13 = *(_BYTE *)(a3 + 1);
LABEL_28:
  if ( v13 )
    v14 = *(_DWORD *)a1 != 0;
  else
    v14 = *((_DWORD *)a1 + 1) != 0;
  if ( v14 )
    return a1;
  return result;
}
