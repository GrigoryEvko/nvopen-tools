// Function: sub_2E21220
// Address: 0x2e21220
//
bool __fastcall sub_2E21220(__int64 a1, __int64 a2, unsigned int a3)
{
  int v5; // eax
  int v6; // edx
  __int64 v7; // rdi
  bool result; // al

  v5 = *(_DWORD *)(a2 + 112);
  v6 = *(_DWORD *)(a1 + 24);
  if ( v5 != *(_DWORD *)(a1 + 68) || *(_DWORD *)(a1 + 64) != v6 )
  {
    *(_DWORD *)(a1 + 64) = v6;
    v7 = *(_QWORD *)(a1 + 8);
    *(_DWORD *)(a1 + 68) = v5;
    *(_DWORD *)(a1 + 136) = 0;
    *(_DWORD *)(a1 + 80) = 0;
    sub_2E13970(v7, a2, a1 + 72);
  }
  result = 0;
  if ( *(_DWORD *)(a1 + 136) )
  {
    result = 1;
    if ( a3 )
      return (*(_QWORD *)(*(_QWORD *)(a1 + 72) + 8LL * (a3 >> 6)) & (1LL << a3)) == 0;
  }
  return result;
}
