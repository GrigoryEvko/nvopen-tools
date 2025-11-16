// Function: sub_1F5BE30
// Address: 0x1f5be30
//
bool __fastcall sub_1F5BE30(__int64 a1, int a2)
{
  __int64 v2; // rsi
  __int64 v3; // rax
  int v4; // edx
  __int64 v5; // rax

  v2 = a2 & 0x7FFFFFFF;
  v3 = *(_QWORD *)(*(_QWORD *)(a1 + 232) + 208LL) + 40 * v2;
  if ( !*(_DWORD *)(v3 + 16) )
    return 0;
  v4 = **(_DWORD **)(v3 + 8);
  if ( *(_DWORD *)v3 || !v4 )
    return 0;
  v5 = *(_QWORD *)(a1 + 264);
  if ( v4 < 0 )
    v4 = *(_DWORD *)(v5 + 4LL * (v4 & 0x7FFFFFFF));
  return *(_DWORD *)(v5 + 4 * v2) == v4;
}
