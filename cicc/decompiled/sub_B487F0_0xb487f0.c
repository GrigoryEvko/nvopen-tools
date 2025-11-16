// Function: sub_B487F0
// Address: 0xb487f0
//
__int64 __fastcall sub_B487F0(int *a1, __int64 a2, int a3)
{
  int v4; // r11d
  int *v6; // r10
  int *v7; // rax
  unsigned int v8; // r8d
  char v9; // cl
  int v10; // edx
  bool v11; // si
  __int64 i; // rax
  int v13; // edx

  v4 = a2;
  v6 = &a1[a2];
  if ( a1 != v6 )
  {
    v7 = a1;
    v8 = 0;
    v9 = 0;
    while ( 1 )
    {
      v10 = *v7;
      if ( *v7 != -1 )
      {
        v11 = a3 > v10;
        LOBYTE(v10) = a3 <= v10;
        v8 |= v10;
        v9 |= v11;
        if ( v9 )
        {
          if ( (_BYTE)v8 )
            break;
        }
      }
      if ( v6 == ++v7 )
      {
        LOBYTE(v8) = v9 | v8;
        if ( !(_BYTE)v8 || v4 <= 0 )
          return v8;
        for ( i = 0; ; ++i )
        {
          v13 = a1[i];
          if ( v13 != (_DWORD)i && v13 != -1 && v13 != a3 + (_DWORD)i )
            break;
          if ( v4 - 1 == i )
            return v8;
        }
        return 0;
      }
    }
  }
  return 0;
}
