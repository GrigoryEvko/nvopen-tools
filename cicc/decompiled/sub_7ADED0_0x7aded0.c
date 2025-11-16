// Function: sub_7ADED0
// Address: 0x7aded0
//
__int64 __fastcall sub_7ADED0(char *s1, __int64 a2)
{
  unsigned int v2; // r13d
  size_t v4; // r14

  v2 = 1;
  if ( strcmp(s1, (const char *)a2) )
  {
    v2 = 0;
    if ( *(_BYTE *)a2 == 95 && *(_BYTE *)(a2 + 1) == 95 )
    {
      v4 = strlen(s1);
      if ( v4 + 2 == strlen((const char *)(a2 + 2)) && *(_BYTE *)(a2 + v4 + 2) == 95 && *(_BYTE *)(a2 + v4 + 3) == 95 )
        return strncmp(s1, (const char *)(a2 + 2), v4) == 0;
    }
  }
  return v2;
}
