// Function: sub_2F50F60
// Address: 0x2f50f60
//
__int64 __fastcall sub_2F50F60(__int64 a1, unsigned int a2)
{
  __int64 v2; // rax
  __int64 v3; // rsi
  unsigned int v4; // r9d
  __int64 v5; // rcx
  __int16 v6; // dx
  __int16 *v7; // rax
  unsigned int v8; // edx
  __int64 v10; // rdi
  int v11; // esi

  v2 = a2;
  v3 = *(_QWORD *)(a1 + 64);
  v4 = v2;
  v5 = *(_QWORD *)(v3 + 24);
  LODWORD(v2) = *(_DWORD *)(*(_QWORD *)(v5 + 8) + 24 * v2 + 16);
  v6 = v2;
  v7 = (__int16 *)(*(_QWORD *)(v5 + 56) + 2LL * ((unsigned int)v2 >> 12));
  v8 = v6 & 0xFFF;
  if ( !v7 )
    return 0;
  v10 = *(_QWORD *)(v3 + 88);
  while ( !*(_WORD *)(v10 + 2LL * v8) )
  {
    v11 = *v7++;
    v8 += v11;
    if ( !(_WORD)v11 )
      return 0;
  }
  return (unsigned int)sub_2E211B0(*(_QWORD **)(a1 + 24), v4) ^ 1;
}
