// Function: sub_3945DA0
// Address: 0x3945da0
//
__int64 __fastcall sub_3945DA0(__int64 *a1, int a2)
{
  __int64 v2; // rdi
  unsigned int v3; // eax
  __int64 v4; // rdx
  int v5; // ecx
  int i; // eax
  __int64 v7; // rdx
  int v9; // eax
  int v10; // eax

  if ( a2 )
  {
    v2 = *a1;
    v3 = a2 - 1;
    if ( a2 != 1 )
    {
      v4 = v2 + 16LL * v3;
      do
      {
        v5 = *(_DWORD *)(v4 + 12);
        if ( v5 )
        {
          i = v3 + 1;
          v7 = *(_QWORD *)(*(_QWORD *)v4 + 8LL * (unsigned int)(v5 - 1));
          if ( a2 == i )
            return v7;
          goto LABEL_7;
        }
        v4 -= 16;
        --v3;
      }
      while ( v3 );
      v9 = *(_DWORD *)(v2 + 12);
      if ( !v9 )
        return 0;
      v7 = *(_QWORD *)(*(_QWORD *)v2 + 8LL * (unsigned int)(v9 - 1));
      for ( i = 1; i != a2; ++i )
LABEL_7:
        v7 = *(_QWORD *)((v7 & 0xFFFFFFFFFFFFFFC0LL) + 8 * (v7 & 0x3F));
      return v7;
    }
    v10 = *(_DWORD *)(v2 + 12);
    if ( v10 )
      return *(_QWORD *)(*(_QWORD *)v2 + 8LL * (unsigned int)(v10 - 1));
  }
  return 0;
}
