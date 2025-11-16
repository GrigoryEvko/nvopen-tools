// Function: sub_B4EFF0
// Address: 0xb4eff0
//
__int64 __fastcall sub_B4EFF0(int *a1, __int64 a2, int a3, int *a4)
{
  int v4; // r10d
  int *v5; // rbx
  int *v7; // r9
  int *v9; // rax
  unsigned int v10; // r8d
  char v11; // cl
  int v12; // edx
  bool v13; // di
  int v14; // ecx
  int v15; // edi

  v4 = a2;
  v5 = &a1[a2];
  if ( a1 != v5 )
  {
    v7 = a1;
    v9 = a1;
    v10 = 0;
    v11 = 0;
    while ( 1 )
    {
      v12 = *v9;
      if ( *v9 != -1 )
      {
        v13 = a3 > v12;
        LOBYTE(v12) = a3 <= v12;
        v10 |= v12;
        v11 |= v13;
        if ( v11 )
        {
          if ( (_BYTE)v10 )
            break;
        }
      }
      if ( v5 == ++v9 )
      {
        LOBYTE(v10) = v11 | v10;
        if ( !(_BYTE)v10 )
          return v10;
        if ( v4 < a3 && v4 )
        {
          v14 = 0;
          v15 = -1;
          do
          {
            if ( *v7 >= 0 )
            {
              if ( v15 >= 0 && *v7 % a3 - v14 != v15 )
                return 0;
              v15 = *v7 % a3 - v14;
            }
            ++v14;
            ++v7;
          }
          while ( v4 != v14 );
          if ( v15 >= 0 && v15 + v4 <= a3 )
          {
            *a4 = v15;
            return v10;
          }
        }
        return 0;
      }
    }
  }
  return 0;
}
