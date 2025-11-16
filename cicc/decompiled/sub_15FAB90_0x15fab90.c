// Function: sub_15FAB90
// Address: 0x15fab90
//
__int64 __fastcall sub_15FAB90(int *a1, int a2)
{
  unsigned int v2; // r8d
  __int64 i; // rax
  int v4; // edx

  v2 = sub_15FAB40(a1, a2);
  if ( !(_BYTE)v2 || a2 <= 0 )
    return v2;
  for ( i = 0; ; ++i )
  {
    v4 = a1[i];
    if ( v4 != -1 && v4 != (_DWORD)i && v4 != a2 + (_DWORD)i )
      break;
    if ( a2 - 1 == i )
      return v2;
  }
  return 0;
}
