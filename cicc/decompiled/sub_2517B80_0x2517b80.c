// Function: sub_2517B80
// Address: 0x2517b80
//
__int64 *__fastcall sub_2517B80(__int64 a1, __int64 *a2)
{
  int v2; // eax
  __int64 v3; // r9
  __int64 v4; // rcx
  int v5; // edx
  int v6; // edi
  unsigned int v7; // eax
  __int64 *v8; // r8
  __int64 v9; // rsi

  v2 = *(_DWORD *)(a1 + 24);
  v3 = *(_QWORD *)(a1 + 8);
  if ( v2 )
  {
    v4 = *a2;
    v5 = v2 - 1;
    v6 = 1;
    v7 = (v2 - 1) & (((unsigned int)*a2 >> 4) ^ ((unsigned int)*a2 >> 9));
    v8 = (__int64 *)(v3 + 8LL * v7);
    v9 = *v8;
    if ( v4 == *v8 )
      return v8;
    while ( v9 != -4096 )
    {
      v7 = v5 & (v6 + v7);
      v8 = (__int64 *)(v3 + 8LL * v7);
      v9 = *v8;
      if ( *v8 == v4 )
        return v8;
      ++v6;
    }
  }
  return 0;
}
