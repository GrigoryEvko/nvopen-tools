// Function: sub_828610
// Address: 0x828610
//
__int64 __fastcall sub_828610(unsigned __int8 **a1)
{
  __int64 v1; // rcx
  int v2; // r8d
  int v3; // edx

  v1 = (__int64)*a1;
  v2 = 0;
  v3 = **a1;
  if ( (unsigned int)(v3 - 48) > 9 )
    return 0;
  while ( 1 )
  {
    if ( v2 <= 99 )
      v2 = (char)v3 + 10 * v2 - 48;
    v3 = *(unsigned __int8 *)(v1 + 1);
    if ( (unsigned int)(v3 - 48) > 9 )
      break;
    ++v1;
  }
  if ( (_BYTE)v3 != 36 )
    return 0;
  if ( v2 > 99 )
  {
    v2 = -1;
  }
  else if ( !v2 )
  {
    v2 = -2;
  }
  *a1 = (unsigned __int8 *)(v1 + 2);
  return (unsigned int)v2;
}
