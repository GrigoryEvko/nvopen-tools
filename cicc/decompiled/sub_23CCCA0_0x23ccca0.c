// Function: sub_23CCCA0
// Address: 0x23ccca0
//
__int64 __fastcall sub_23CCCA0(__int64 a1, __int64 a2)
{
  int v2; // eax
  __int64 v3; // r9
  int v4; // ecx
  unsigned int v5; // eax
  __int64 v6; // rdx
  unsigned int v8; // r8d
  int v9; // r8d

  if ( a2 )
  {
    v2 = *(_DWORD *)(a1 + 344);
    v3 = *(_QWORD *)(a1 + 328);
    if ( v2 )
    {
      v4 = v2 - 1;
      v5 = (v2 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v6 = *(_QWORD *)(v3 + 16LL * v5);
      if ( v6 == a2 )
        return 0;
      v9 = 1;
      while ( v6 != -4096 )
      {
        v5 = v4 & (v9 + v5);
        v6 = *(_QWORD *)(v3 + 16LL * v5);
        if ( a2 == v6 )
          return 0;
        ++v9;
      }
    }
    return (unsigned int)sub_23CC880((_QWORD *)(a1 + 96), a2) ^ 1;
  }
  else
  {
    v8 = 0;
    if ( *(_DWORD *)(a1 + 312) )
      return 0;
    LOBYTE(v8) = *(_QWORD *)(a1 + 144) == *(_QWORD *)(a1 + 112);
    return v8;
  }
}
