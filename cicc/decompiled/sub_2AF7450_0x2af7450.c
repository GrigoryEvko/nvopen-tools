// Function: sub_2AF7450
// Address: 0x2af7450
//
char *__fastcall sub_2AF7450(char *a1)
{
  unsigned __int8 v1; // al
  __int64 v2; // rax
  int v3; // eax
  char *result; // rax
  char v5; // dl

  v1 = *a1;
  if ( (unsigned __int8)*a1 <= 0x1Cu )
    goto LABEL_24;
  if ( (unsigned __int8)(v1 - 61) <= 1u )
  {
    result = (char *)*((_QWORD *)a1 - 4);
  }
  else
  {
    if ( v1 != 85 )
      goto LABEL_24;
    v2 = *((_QWORD *)a1 - 4);
    if ( !v2 )
      goto LABEL_24;
    if ( *(_BYTE *)v2 )
      goto LABEL_24;
    if ( *(_QWORD *)(v2 + 24) != *((_QWORD *)a1 + 10) )
      goto LABEL_24;
    v3 = *(_DWORD *)(v2 + 36);
    if ( !v3 )
      goto LABEL_24;
    if ( v3 != 8975 && v3 != 8937 )
    {
      if ( v3 == 9567 || v3 == 9549 )
      {
        result = *(char **)&a1[32 * (2LL - (*((_DWORD *)a1 + 1) & 0x7FFFFFF))];
        goto LABEL_14;
      }
LABEL_24:
      BUG();
    }
    result = *(char **)&a1[32 * (1LL - (*((_DWORD *)a1 + 1) & 0x7FFFFFF))];
  }
LABEL_14:
  v5 = *result;
  if ( (unsigned __int8)*result > 0x1Cu )
  {
    if ( v5 != 63 )
      return 0;
  }
  else if ( v5 == 5 )
  {
    if ( *((_WORD *)result + 1) != 34 )
      return 0;
  }
  else
  {
    return 0;
  }
  return result;
}
