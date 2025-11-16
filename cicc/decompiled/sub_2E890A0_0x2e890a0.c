// Function: sub_2E890A0
// Address: 0x2e890a0
//
__int64 __fastcall sub_2E890A0(__int64 a1, unsigned int a2, _DWORD *a3)
{
  unsigned int v5; // edi
  __int64 v6; // r8
  unsigned int v7; // edx
  int i; // ecx
  unsigned int v9; // eax
  __int64 v10; // rax

  if ( a2 > 1 )
  {
    v5 = *(_DWORD *)(a1 + 40) & 0xFFFFFF;
    if ( v5 > 2 )
    {
      v6 = *(_QWORD *)(a1 + 32);
      v7 = 2;
      for ( i = 0; ; ++i )
      {
        v10 = v6 + 40LL * v7;
        if ( *(_BYTE *)v10 != 1 )
          break;
        v9 = v7 + (((unsigned int)*(_QWORD *)(v10 + 24) >> 3) & 0x1FFF) + 1;
        if ( v9 > a2 )
        {
          if ( a3 )
            *a3 = i;
          return v7;
        }
        if ( v9 >= v5 )
          return 0xFFFFFFFFLL;
        v7 = v9;
      }
    }
  }
  return 0xFFFFFFFFLL;
}
