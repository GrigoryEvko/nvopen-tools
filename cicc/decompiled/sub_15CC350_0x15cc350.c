// Function: sub_15CC350
// Address: 0x15cc350
//
__int64 __fastcall sub_15CC350(__int64 *a1)
{
  unsigned __int64 v1; // r12
  int v2; // eax
  int v3; // r14d
  unsigned int v4; // ebx
  int i; // r15d

  v1 = sub_157EBA0(*a1);
  v2 = sub_15F4D60(v1);
  if ( !v2 )
    return 1;
  v3 = v2;
  v4 = 0;
  for ( i = 0; ; i = 1 )
  {
    while ( a1[1] != sub_15F4DF0(v1, v4) )
    {
      if ( v3 == ++v4 )
        return 1;
    }
    if ( i == 1 )
      break;
    if ( v3 == ++v4 )
      return 1;
  }
  return 0;
}
