// Function: sub_278A6A0
// Address: 0x278a6a0
//
__int64 __fastcall sub_278A6A0(__int64 a1, __int64 a2)
{
  int v2; // eax
  __int64 v3; // r9
  int v4; // ecx
  int v5; // edi
  unsigned int v6; // eax
  __int64 v7; // rdx

  v2 = *(_DWORD *)(a1 + 24);
  v3 = *(_QWORD *)(a1 + 8);
  if ( v2 )
  {
    v4 = v2 - 1;
    v5 = 1;
    v6 = (v2 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v7 = *(_QWORD *)(v3 + 16LL * v6);
    if ( v7 == a2 )
      return 1;
    while ( v7 != -4096 )
    {
      v6 = v4 & (v5 + v6);
      v7 = *(_QWORD *)(v3 + 16LL * v6);
      if ( a2 == v7 )
        return 1;
      ++v5;
    }
  }
  return 0;
}
