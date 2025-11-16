// Function: sub_15FAD30
// Address: 0x15fad30
//
__int64 __fastcall sub_15FAD30(_DWORD *a1, int a2)
{
  __int64 result; // rax
  int v3; // edx
  __int64 v4; // rdx
  int v5; // eax

  result = 0;
  if ( a2 > 1 && (a2 & (a2 - 1)) == 0 && *a1 <= 1u )
  {
    v3 = a1[1] - *a1;
    if ( v3 == a2 )
    {
      if ( v3 <= 2 )
      {
        return 1;
      }
      else
      {
        v4 = (__int64)&a1[v3 - 3 + 1];
        while ( 1 )
        {
          v5 = a1[2];
          if ( v5 == -1 || v5 - *a1 != 2 )
            break;
          if ( ++a1 == (_DWORD *)v4 )
            return 1;
        }
        return 0;
      }
    }
  }
  return result;
}
