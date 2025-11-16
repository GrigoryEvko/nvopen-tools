// Function: sub_29AACC0
// Address: 0x29aacc0
//
__int64 __fastcall sub_29AACC0(__int64 a1, __int64 a2)
{
  unsigned int v2; // r8d
  int v3; // eax
  __int64 v4; // rcx
  __int64 v5; // rsi
  int v6; // edx
  unsigned int v7; // eax
  __int64 v8; // rdi
  int i; // r8d

  v2 = 0;
  if ( *(_BYTE *)a2 > 0x1Cu )
  {
    v3 = *(_DWORD *)(a1 + 24);
    v4 = *(_QWORD *)(a2 + 40);
    v5 = *(_QWORD *)(a1 + 8);
    if ( v3 )
    {
      v6 = v3 - 1;
      v2 = 1;
      v7 = (v3 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
      v8 = *(_QWORD *)(v5 + 8LL * v7);
      if ( v4 != v8 )
      {
        for ( i = 1; ; ++i )
        {
          if ( v8 == -4096 )
            return 0;
          v7 = v6 & (i + v7);
          v8 = *(_QWORD *)(v5 + 8LL * v7);
          if ( v4 == v8 )
            break;
        }
        return 1;
      }
    }
  }
  return v2;
}
