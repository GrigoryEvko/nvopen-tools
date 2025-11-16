// Function: sub_1CFBE00
// Address: 0x1cfbe00
//
__int64 __fastcall sub_1CFBE00(char *s)
{
  __int64 result; // rax

  if ( !s )
    return 0;
  if ( *s == 115 && s[1] == 109 && s[2] == 95 )
    return strtol(s + 3, 0, 10);
  if ( !memcmp(s, "compute_", 8u) && strlen(s) > 9 )
    return strtol(s + 8, 0, 10);
  result = 0;
  if ( !memcmp(s, "lto_", 4u) )
    return strtol(s + 4, 0, 10);
  return result;
}
