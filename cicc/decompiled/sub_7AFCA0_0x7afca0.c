// Function: sub_7AFCA0
// Address: 0x7afca0
//
_BOOL8 __fastcall sub_7AFCA0(char **a1, __int64 a2)
{
  char *v3; // rdi

  v3 = *(char **)a2;
  if ( (*(_BYTE *)(a2 + 8) & 0x10) != 0 )
    return !sub_722E50(v3, *a1, 0, 0, 0);
  else
    return strcmp(v3, *a1) == 0;
}
