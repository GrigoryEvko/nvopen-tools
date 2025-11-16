// Function: sub_E92070
// Address: 0xe92070
//
__int64 __fastcall sub_E92070(__int64 a1, unsigned int a2, unsigned int a3)
{
  __int64 v3; // rcx
  __int64 v5; // rax
  __int64 v6; // rsi
  __int16 *v7; // rdi
  unsigned int v8; // edx
  unsigned int v9; // eax
  __int16 *v10; // rsi

  v3 = *(_QWORD *)(a1 + 8);
  v5 = 3LL * a2;
  v6 = *(_QWORD *)(a1 + 56);
  LODWORD(v5) = *(_DWORD *)(v3 + 8 * v5 + 16);
  v7 = (__int16 *)(v6 + 2LL * ((unsigned int)v5 >> 12));
  v8 = v5 & 0xFFF;
  LODWORD(v3) = *(_DWORD *)(v3 + 24LL * a3 + 16);
  v9 = v3 & 0xFFF;
  v10 = (__int16 *)(v6 + 2LL * ((unsigned int)v3 >> 12));
  if ( v8 == v9 )
    return 1;
  while ( 1 )
  {
    while ( v8 < v9 )
    {
      if ( !*v7 )
        return 0;
      v8 += *v7++;
      if ( v8 == v9 )
        return 1;
    }
    if ( !*v10 )
      break;
    v9 += *v10++;
    if ( v8 == v9 )
      return 1;
  }
  return 0;
}
