// Function: sub_826000
// Address: 0x826000
//
__int64 __fastcall sub_826000(const char *a1)
{
  unsigned int v1; // r12d
  const char *v2; // rbx
  size_t v3; // rdx

  v1 = 0;
  if ( !a1 )
    return v1;
  v2 = a1;
  v3 = (size_t)&a1[strlen(a1) - 1];
  if ( (unsigned __int64)a1 > v3 )
    return v1;
  do
  {
    if ( *v2 == 92 && (v2[1] & 0xDF) == 0x55 )
      return 1;
    ++v2;
  }
  while ( v3 >= (unsigned __int64)v2 );
  return 0;
}
