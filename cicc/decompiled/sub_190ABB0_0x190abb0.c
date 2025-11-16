// Function: sub_190ABB0
// Address: 0x190abb0
//
__int64 __fastcall sub_190ABB0(__int64 a1, __int64 a2)
{
  int v2; // eax
  unsigned int v3; // r8d
  int v4; // edx
  __int64 v5; // rdi
  unsigned int v6; // eax
  __int64 v7; // rcx
  int i; // r8d

  v2 = *(_DWORD *)(a1 + 24);
  v3 = 0;
  if ( v2 )
  {
    v4 = v2 - 1;
    v5 = *(_QWORD *)(a1 + 8);
    v3 = 1;
    v6 = (v2 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v7 = *(_QWORD *)(v5 + 16LL * v6);
    if ( v7 != a2 )
    {
      for ( i = 1; ; ++i )
      {
        if ( v7 == -8 )
          return 0;
        v6 = v4 & (i + v6);
        v7 = *(_QWORD *)(v5 + 16LL * v6);
        if ( a2 == v7 )
          break;
      }
      return 1;
    }
  }
  return v3;
}
