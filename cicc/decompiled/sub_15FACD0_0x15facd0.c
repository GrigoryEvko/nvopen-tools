// Function: sub_15FACD0
// Address: 0x15facd0
//
__int64 __fastcall sub_15FACD0(int *a1, int a2)
{
  unsigned int v2; // eax
  unsigned int v3; // r8d
  __int64 i; // rax
  int v5; // edx

  v2 = sub_15FAB40(a1, a2);
  v3 = 0;
  if ( !(_BYTE)v2 )
  {
    if ( a2 <= 0 )
    {
      return 1;
    }
    else
    {
      v3 = v2;
      for ( i = 0; ; ++i )
      {
        v5 = a1[i];
        if ( v5 != -1 && v5 != (_DWORD)i && v5 != a2 + (_DWORD)i )
          break;
        if ( a2 - 1 == i )
          return 1;
      }
    }
  }
  return v3;
}
