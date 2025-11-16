// Function: sub_3288660
// Address: 0x3288660
//
void __fastcall sub_3288660(int *a1, int a2)
{
  __int64 v2; // r9
  int v3; // eax
  int v4; // edx
  bool v5; // cc
  int v6; // eax

  if ( a2 )
  {
    v2 = (__int64)&a1[a2 - 1 + 1];
    do
    {
      v3 = *a1;
      if ( *a1 >= 0 )
      {
        v4 = a2 + v3;
        v5 = v3 < a2;
        v6 = v3 - a2;
        if ( v5 )
          v6 = v4;
        *a1 = v6;
      }
      ++a1;
    }
    while ( a1 != (int *)v2 );
  }
}
