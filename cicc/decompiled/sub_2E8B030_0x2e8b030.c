// Function: sub_2E8B030
// Address: 0x2e8b030
//
__int64 __fastcall sub_2E8B030(__int64 a1)
{
  __int64 v1; // rsi
  unsigned int v2; // ecx
  unsigned int v3; // edx
  _DWORD *v4; // rax
  __int64 v5; // rcx

  if ( *(_WORD *)(a1 + 68) != 68 && *(_WORD *)(a1 + 68) )
    return 0;
  v1 = *(_QWORD *)(a1 + 32);
  v2 = *(_DWORD *)(a1 + 40) & 0xFFFFFF;
  v3 = *(_DWORD *)(v1 + 48);
  if ( v2 > 3 )
  {
    v4 = (_DWORD *)(v1 + 128);
    v5 = v1 + 80LL * ((v2 - 4) >> 1) + 208;
    while ( v3 == *v4 )
    {
      v4 += 20;
      if ( v4 == (_DWORD *)v5 )
        return v3;
    }
    return 0;
  }
  return v3;
}
