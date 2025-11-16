// Function: sub_7AFC20
// Address: 0x7afc20
//
__int64 __fastcall sub_7AFC20(char **a1)
{
  char *v1; // rax
  unsigned __int8 v2; // bl
  char *v3; // r12
  unsigned int v4; // r13d

  v1 = sub_722280(*a1);
  v2 = *v1;
  if ( !*v1 )
    return 0;
  v3 = v1;
  v4 = 0;
  do
  {
    if ( isupper(v2) )
      v2 = tolower((char)v2);
    ++v3;
    v4 = (char)v2 + 32 * v4;
    v2 = *v3;
  }
  while ( *v3 );
  return v4;
}
