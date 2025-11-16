// Function: sub_2506F10
// Address: 0x2506f10
//
__int64 __fastcall sub_2506F10(__int64 a1, __int64 a2)
{
  int v2; // eax
  __int64 v3; // rcx
  int v4; // edx
  unsigned int v5; // eax
  __int64 v6; // rdi
  int v8; // r8d

  if ( !*(_DWORD *)(a1 + 40) )
    return 1;
  v2 = *(_DWORD *)(a1 + 24);
  v3 = *(_QWORD *)(a1 + 8);
  if ( v2 )
  {
    v4 = v2 - 1;
    v5 = (v2 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v6 = *(_QWORD *)(v3 + 8LL * v5);
    if ( v6 == a2 )
      return 1;
    v8 = 1;
    while ( v6 != -4096 )
    {
      v5 = v4 & (v8 + v5);
      v6 = *(_QWORD *)(v3 + 8LL * v5);
      if ( v6 == a2 )
        return 1;
      ++v8;
    }
  }
  return 0;
}
