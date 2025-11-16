// Function: sub_815620
// Address: 0x815620
//
char *__fastcall sub_815620(char *src, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int8 *v6; // r14
  size_t v7; // r12
  size_t v8; // rbx
  char *v9; // r12

  if ( (*(_BYTE *)(a2 + 89) & 8) == 0 && dword_4F077C4 == 2 )
  {
    if ( (_BYTE)a3 == 7 )
    {
      v6 = sub_813030(a2, a2);
    }
    else
    {
      if ( (_BYTE)a3 != 11 )
        sub_721090();
      v6 = sub_815600(a2, a2, a3, a4, a5, a6);
    }
  }
  else
  {
    v6 = *(unsigned __int8 **)(a2 + 8);
  }
  if ( *v6 == 95 && v6[1] == 90 )
    v6 += 2;
  v7 = strlen((const char *)v6);
  v8 = strlen(src);
  v9 = (char *)sub_7E1510(v7 + v8 + 1);
  strcpy(v9, src);
  strcpy(&v9[v8], (const char *)v6);
  return v9;
}
