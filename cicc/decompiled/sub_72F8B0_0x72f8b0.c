// Function: sub_72F8B0
// Address: 0x72f8b0
//
_BOOL8 __fastcall sub_72F8B0(__int64 **a1)
{
  _BOOL8 result; // rax
  __int64 *v2; // rcx
  __int64 v3; // rdx

  result = 0;
  if ( *a1 )
  {
    if ( (*((_BYTE *)a1 + 89) & 4) == 0 )
    {
      v2 = a1[5];
      if ( !v2 || *((_BYTE *)v2 + 28) != 3 )
      {
        v3 = **a1;
        result = 0;
        if ( (*(_BYTE *)(v3 + 73) & 2) != 0 )
          return strcmp(*(const char **)(v3 + 8), "main") == 0;
      }
    }
  }
  return result;
}
