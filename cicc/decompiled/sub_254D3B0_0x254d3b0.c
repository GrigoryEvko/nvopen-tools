// Function: sub_254D3B0
// Address: 0x254d3b0
//
__int64 __fastcall sub_254D3B0(__int64 a1, _QWORD *a2)
{
  __int64 v2; // rax
  __int64 v3; // rdx
  __int64 v4; // rsi
  int v5; // edx
  int v6; // ecx
  unsigned int v7; // edx
  __int64 v8; // rdi
  int v10; // r8d

  v2 = sub_25096F0(a2);
  v3 = *(_QWORD *)(a1 + 200);
  if ( !*(_DWORD *)(v3 + 40) )
    return 1;
  v4 = *(_QWORD *)(v3 + 8);
  v5 = *(_DWORD *)(v3 + 24);
  if ( v5 )
  {
    v6 = v5 - 1;
    v7 = (v5 - 1) & (((unsigned int)v2 >> 9) ^ ((unsigned int)v2 >> 4));
    v8 = *(_QWORD *)(v4 + 8LL * v7);
    if ( v2 == v8 )
      return 1;
    v10 = 1;
    while ( v8 != -4096 )
    {
      v7 = v6 & (v10 + v7);
      v8 = *(_QWORD *)(v4 + 8LL * v7);
      if ( v2 == v8 )
        return 1;
      ++v10;
    }
  }
  return 0;
}
