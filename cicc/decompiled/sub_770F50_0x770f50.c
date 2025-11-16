// Function: sub_770F50
// Address: 0x770f50
//
__int64 __fastcall sub_770F50(__int64 a1, int a2, unsigned int a3)
{
  unsigned int i; // eax
  unsigned int v5; // eax
  int v6; // edx

  for ( i = a3; ; i = v5 + 1 )
  {
    v5 = a2 & i;
    v6 = *(_DWORD *)(a1 + 4LL * v5);
    if ( a3 == v6 )
      return 1;
    if ( !v6 )
      break;
  }
  return 0;
}
