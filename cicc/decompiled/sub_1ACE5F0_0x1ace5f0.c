// Function: sub_1ACE5F0
// Address: 0x1ace5f0
//
__int64 __fastcall sub_1ACE5F0(__int64 a1, __int64 a2)
{
  int v2; // eax
  unsigned int v3; // r8d
  int v4; // edx
  __int64 v5; // rsi
  unsigned int v6; // eax
  __int64 v7; // rcx
  int i; // r8d
  __int64 v11; // rsi

  v11 = *(_QWORD *)(a1 + 16);
  if ( !v11 )
    return 0;
  v2 = *(_DWORD *)(v11 + 24);
  v3 = 0;
  if ( v2 )
  {
    v4 = v2 - 1;
    v5 = *(_QWORD *)(v11 + 8);
    v3 = 1;
    v6 = (v2 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v7 = *(_QWORD *)(v5 + 8LL * v6);
    if ( a2 != v7 )
    {
      for ( i = 1; ; ++i )
      {
        if ( v7 == -8 )
          return 0;
        v6 = v4 & (i + v6);
        v7 = *(_QWORD *)(v5 + 8LL * v6);
        if ( a2 == v7 )
          break;
      }
      return 1;
    }
  }
  return v3;
}
