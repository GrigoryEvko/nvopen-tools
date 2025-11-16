// Function: sub_27CDC30
// Address: 0x27cdc30
//
char *__fastcall sub_27CDC30(char *a1, int a2)
{
  char *result; // rax
  __int64 v3; // rdx
  char v4; // dl
  __int16 v5; // dx
  char *v6; // rax

  result = a1;
  while ( 1 )
  {
    v3 = *((_QWORD *)result + 1);
    if ( *(_BYTE *)(v3 + 8) == 14 && a2 != *(_DWORD *)(v3 + 8) >> 8 )
      return result;
    v4 = *result;
    if ( (unsigned __int8)*result <= 0x1Cu )
    {
      if ( v4 != 5 )
        return result;
      v5 = *((_WORD *)result + 1);
      if ( (unsigned __int16)(v5 - 49) <= 1u )
        goto LABEL_12;
      if ( v5 != 34 )
        return result;
LABEL_11:
      result = *(char **)&result[-32 * (*((_DWORD *)result + 1) & 0x7FFFFFF)];
    }
    else
    {
      if ( (unsigned __int8)(v4 - 78) > 1u )
      {
        if ( v4 != 63 )
          return result;
        goto LABEL_11;
      }
LABEL_12:
      if ( (result[7] & 0x40) != 0 )
        v6 = (char *)*((_QWORD *)result - 1);
      else
        v6 = &result[-32 * (*((_DWORD *)result + 1) & 0x7FFFFFF)];
      result = *(char **)v6;
    }
  }
}
