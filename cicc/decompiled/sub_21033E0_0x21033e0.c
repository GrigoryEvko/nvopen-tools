// Function: sub_21033E0
// Address: 0x21033e0
//
bool __fastcall sub_21033E0(__int64 a1, __int64 a2, unsigned int a3)
{
  int v5; // eax
  int v6; // edx
  __int64 v7; // rdi
  bool result; // al

  v5 = *(_DWORD *)(a2 + 112);
  v6 = *(_DWORD *)(a1 + 256);
  if ( *(_DWORD *)(a1 + 404) != v5 || *(_DWORD *)(a1 + 400) != v6 )
  {
    *(_DWORD *)(a1 + 400) = v6;
    v7 = *(_QWORD *)(a1 + 240);
    *(_DWORD *)(a1 + 404) = v5;
    *(_DWORD *)(a1 + 424) = 0;
    sub_1DBCD10(v7, a2, a1 + 408);
  }
  result = 0;
  if ( *(_DWORD *)(a1 + 424) )
  {
    result = 1;
    if ( a3 )
      return (*(_QWORD *)(*(_QWORD *)(a1 + 408) + 8LL * (a3 >> 6)) & (1LL << a3)) == 0;
  }
  return result;
}
