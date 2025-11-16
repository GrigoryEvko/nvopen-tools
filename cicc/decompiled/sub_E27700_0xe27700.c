// Function: sub_E27700
// Address: 0xe27700
//
unsigned __int64 __fastcall sub_E27700(__int64 a1, __int64 *a2, int a3)
{
  char v5; // bl
  unsigned __int64 v6; // rdx
  __int64 v7; // rsi
  char v8; // cl
  unsigned __int64 result; // rax
  unsigned __int64 v10; // rdi
  int v11; // eax
  __int64 v12; // rcx

  if ( a3 == 1 )
    goto LABEL_26;
  v5 = 0;
  if ( a3 != 2 )
    goto LABEL_3;
  v6 = *a2;
  if ( !*a2 )
    goto LABEL_27;
  v7 = a2[1];
  v8 = *(_BYTE *)v7;
  if ( *(_BYTE *)v7 == 63 )
  {
    a2[1] = v7 + 1;
    *a2 = v6 - 1;
LABEL_26:
    v5 = sub_E22E40(a1, a2);
LABEL_3:
    v6 = *a2;
    if ( *a2 )
    {
      v7 = a2[1];
      v8 = *(_BYTE *)v7;
      goto LABEL_5;
    }
LABEL_27:
    *(_BYTE *)(a1 + 8) = 1;
    return 0;
  }
LABEL_5:
  if ( (unsigned int)(v8 - 84) <= 3 )
  {
    result = sub_E272C0(a1, a2);
    goto LABEL_7;
  }
  if ( v6 > 2 && *(_WORD *)v7 == 9252 && *(_BYTE *)(v7 + 2) == 81 || v8 == 65 || (unsigned __int8)(v8 - 80) <= 3u )
  {
    *(_BYTE *)(a1 + 8) = 0;
    if ( *(char *)v7 <= 79 )
    {
LABEL_16:
      result = sub_E287D0(a1, a2);
      goto LABEL_7;
    }
    v10 = v6 - 1;
    if ( v6 != 1 )
    {
      v11 = *(char *)(v7 + 1);
      if ( (unsigned int)(v11 - 48) > 9 )
      {
        v12 = v7 + 1;
        if ( (_BYTE)v11 == 69 )
        {
          v12 = v7 + 2;
          v10 = v6 - 2;
          if ( v6 == 2 )
            goto LABEL_43;
          LOBYTE(v11) = *(_BYTE *)(v7 + 2);
        }
        if ( (_BYTE)v11 == 73 )
        {
          if ( !--v10 )
            goto LABEL_43;
          LOBYTE(v11) = *(_BYTE *)++v12;
        }
        if ( (_BYTE)v11 == 70 )
        {
          if ( v10 == 1 )
            goto LABEL_43;
          LOBYTE(v11) = *(_BYTE *)(v12 + 1);
        }
        if ( (char)v11 > 68 )
        {
          if ( (unsigned __int8)(v11 - 81) <= 3u )
          {
LABEL_37:
            result = sub_E28F00(a1, a2);
            goto LABEL_7;
          }
        }
        else if ( (char)v11 > 64 )
        {
          goto LABEL_16;
        }
      }
      else
      {
        if ( (_BYTE)v11 == 54 )
          goto LABEL_16;
        if ( (_BYTE)v11 == 56 )
          goto LABEL_37;
      }
    }
LABEL_43:
    *(_BYTE *)(a1 + 8) = 1;
    return 0;
  }
  if ( v8 == 89 )
  {
    result = sub_E27C40(a1, a2);
    goto LABEL_7;
  }
  if ( v6 <= 5 )
  {
    if ( v6 <= 3 )
      goto LABEL_31;
  }
  else if ( *(_DWORD *)v7 == 943793188 && *(_WORD *)(v7 + 4) == 16448 )
  {
    goto LABEL_21;
  }
  if ( *(_DWORD *)v7 == 910238756 )
  {
LABEL_21:
    if ( (unsigned __int8)sub_E20730((size_t *)a2, 6u, "$$A8@@") )
    {
      result = sub_E28570(a1, a2, 1);
    }
    else
    {
      sub_E20730((size_t *)a2, 4u, "$$A6");
      result = sub_E28570(a1, a2, 0);
    }
    goto LABEL_7;
  }
LABEL_31:
  if ( v8 == 63 )
    result = sub_E25410(a1, a2);
  else
    result = sub_E22F60(a1, (size_t *)a2);
LABEL_7:
  if ( !result )
    return 0;
  if ( !*(_BYTE *)(a1 + 8) )
    *(_BYTE *)(result + 12) |= v5;
  return result;
}
