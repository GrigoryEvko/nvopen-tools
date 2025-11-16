// Function: sub_15FAB40
// Address: 0x15fab40
//
__int64 __fastcall sub_15FAB40(int *a1, int a2)
{
  __int64 v3; // rax
  char v4; // dl
  char v5; // si
  __int64 v6; // r9
  int v7; // eax

  if ( a2 <= 0 )
    return 1;
  v3 = (unsigned int)(a2 - 1);
  v4 = 0;
  v5 = 0;
  v6 = (__int64)&a1[v3 + 1];
  while ( 1 )
  {
    v7 = *a1;
    if ( *a1 != -1 )
    {
      v5 |= v7 >= a2;
      v4 |= v7 < a2;
      if ( v4 )
      {
        if ( v5 )
          break;
      }
    }
    if ( (int *)v6 == ++a1 )
      return 1;
  }
  return 0;
}
