// Function: sub_60F910
// Address: 0x60f910
//
__int64 __fastcall sub_60F910(unsigned __int8 *a1)
{
  unsigned __int8 v1; // al
  unsigned __int8 *v2; // rcx
  __int64 v3; // r8
  int v4; // eax
  __int64 v5; // r8

  v1 = *a1;
  if ( !*a1 )
    return 0;
  v2 = a1;
  v3 = 0;
  do
  {
    if ( (unsigned int)v1 - 48 > 9
      || (v4 = (char)v1 - 48, v3 > 0xCCCCCCCCCCCCCCCLL)
      || (v5 = 10 * v3, 0x7FFFFFFFFFFFFFFFLL - v4 < v5) )
    {
      sub_684920(574);
    }
    ++v2;
    v3 = v4 + v5;
    v1 = *v2;
  }
  while ( *v2 );
  return v3;
}
