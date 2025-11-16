// Function: sub_104D2E0
// Address: 0x104d2e0
//
__int64 __fastcall sub_104D2E0(__int64 a1, __int64 a2)
{
  int v2; // eax
  __int64 v3; // rsi
  __int64 v4; // r9
  int v5; // ecx
  int v6; // edi
  unsigned int v7; // eax
  __int64 v8; // rdx

  v2 = *(_DWORD *)(a1 + 600);
  v3 = *(_QWORD *)(a2 + 40);
  v4 = *(_QWORD *)(a1 + 584);
  if ( v2 )
  {
    v5 = v2 - 1;
    v6 = 1;
    v7 = (v2 - 1) & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
    v8 = *(_QWORD *)(v4 + 16LL * v7);
    if ( v8 == v3 )
      return 1;
    while ( v8 != -4096 )
    {
      v7 = v5 & (v6 + v7);
      v8 = *(_QWORD *)(v4 + 16LL * v7);
      if ( v3 == v8 )
        return 1;
      ++v6;
    }
  }
  return 0;
}
