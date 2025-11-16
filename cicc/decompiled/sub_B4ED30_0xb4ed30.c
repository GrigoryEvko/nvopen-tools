// Function: sub_B4ED30
// Address: 0xb4ed30
//
__int64 __fastcall sub_B4ED30(int *a1, __int64 a2, int a3)
{
  int *v3; // r9
  int v5; // esi
  unsigned int v6; // edx
  int v7; // eax
  bool v8; // cl

  v3 = &a1[a2];
  if ( v3 != a1 )
  {
    v5 = 0;
    v6 = 0;
    while ( 1 )
    {
      v7 = *a1;
      if ( *a1 != -1 )
      {
        v8 = a3 > v7;
        LOBYTE(v7) = a3 <= v7;
        v5 |= v7;
        LOBYTE(v6) = v8 | v6;
        if ( (_BYTE)v6 )
        {
          if ( (_BYTE)v5 )
            break;
        }
      }
      if ( v3 == ++a1 )
        return v5 | v6;
    }
  }
  return 0;
}
