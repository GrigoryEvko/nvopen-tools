// Function: sub_B4EF10
// Address: 0xb4ef10
//
__int64 __fastcall sub_B4EF10(_DWORD *a1, __int64 a2, int a3)
{
  unsigned int v3; // r8d
  int v4; // eax
  __int64 v5; // rdx
  int v6; // eax

  v3 = 0;
  if ( a3 == a2 && a3 > 1 && (a3 & (a3 - 1)) == 0 && *a1 <= 1u )
  {
    v4 = a1[1] - *a1;
    if ( v4 == a3 )
    {
      if ( v4 == 2 )
      {
        return 1;
      }
      else
      {
        v5 = (__int64)&a1[v4 - 3 + 1];
        while ( 1 )
        {
          v6 = a1[2];
          if ( v6 == -1 || v6 - *a1 != 2 )
            break;
          if ( ++a1 == (_DWORD *)v5 )
            return 1;
        }
        return 0;
      }
    }
  }
  return v3;
}
