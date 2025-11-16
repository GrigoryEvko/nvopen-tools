// Function: sub_1ABB580
// Address: 0x1abb580
//
__int64 __fastcall sub_1ABB580(__int64 a1, __int64 a2)
{
  unsigned int v2; // r8d
  int v3; // eax
  __int64 v4; // rcx
  int v5; // eax
  __int64 v6; // rdi
  unsigned int v7; // edx
  __int64 v8; // rsi
  int i; // r8d

  v2 = 0;
  if ( *(_BYTE *)(a2 + 16) > 0x17u )
  {
    v3 = *(_DWORD *)(a1 + 24);
    if ( v3 )
    {
      v4 = *(_QWORD *)(a2 + 40);
      v5 = v3 - 1;
      v6 = *(_QWORD *)(a1 + 8);
      v2 = 1;
      v7 = v5 & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
      v8 = *(_QWORD *)(v6 + 8LL * v7);
      if ( v4 != v8 )
      {
        for ( i = 1; ; ++i )
        {
          if ( v8 == -8 )
            return 0;
          v7 = v5 & (i + v7);
          v8 = *(_QWORD *)(v6 + 8LL * v7);
          if ( v4 == v8 )
            break;
        }
        return 1;
      }
    }
  }
  return v2;
}
