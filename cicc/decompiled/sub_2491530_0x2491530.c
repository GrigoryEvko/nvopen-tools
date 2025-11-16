// Function: sub_2491530
// Address: 0x2491530
//
char **__fastcall sub_2491530(void *s2, size_t n)
{
  const char *v2; // r13
  char **i; // r12

  v2 = "llvm.sqrt.f32";
  for ( i = &off_49D3180; ; v2 = *i )
  {
    if ( v2 )
    {
      if ( strlen(v2) == n && (!n || !memcmp(v2, s2, n)) )
        return i;
    }
    else if ( !n )
    {
      return i;
    }
    i += 3;
    if ( i == (char **)&unk_49D3948 )
      break;
  }
  return 0;
}
