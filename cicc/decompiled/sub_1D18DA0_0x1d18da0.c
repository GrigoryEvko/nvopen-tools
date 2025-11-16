// Function: sub_1D18DA0
// Address: 0x1d18da0
//
__int64 __fastcall sub_1D18DA0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rdx

  v2 = *(_QWORD *)(a2 + 32);
  v3 = v2 + 40LL * *(unsigned int *)(a2 + 56);
  if ( v2 == v3 )
    return 0;
  while ( *(_QWORD *)a1 != *(_QWORD *)v2 || *(_DWORD *)(a1 + 8) != *(_DWORD *)(v2 + 8) )
  {
    v2 += 40;
    if ( v3 == v2 )
      return 0;
  }
  return 1;
}
