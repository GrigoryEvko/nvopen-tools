// Function: sub_908780
// Address: 0x908780
//
__int64 __fastcall sub_908780(char *s, char *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  size_t v7; // r14
  size_t v8; // rcx

  result = sub_C96F30(s, a2, a3, a4, a5, a6);
  if ( result )
  {
    v7 = 0;
    if ( s )
      v7 = strlen(s);
    v8 = 0;
    if ( a2 )
      v8 = strlen(a2);
    return sub_C996C0(s, v7, a2, v8);
  }
  return result;
}
