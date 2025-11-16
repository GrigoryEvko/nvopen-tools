// Function: sub_24F3250
// Address: 0x24f3250
//
__int64 __fastcall sub_24F3250(__int64 a1)
{
  const char *v1; // r14
  char **i; // rbx
  size_t v3; // rdx

  v1 = "llvm.coro.align";
  for ( i = off_49D39E8; ; ++i )
  {
    v3 = 0;
    if ( v1 )
      v3 = strlen(v1);
    if ( sub_BA8B30(a1, (__int64)v1, v3) )
      return 1;
    if ( i == &off_49D3AE0 )
      break;
    v1 = *i;
  }
  return 0;
}
