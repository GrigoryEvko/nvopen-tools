// Function: sub_15CC590
// Address: 0x15cc590
//
__int64 __fastcall sub_15CC590(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r12
  __int64 v5; // rax
  __int64 v6; // rdx

  v3 = *(_QWORD *)(*(_QWORD *)(a2 + 56) + 80LL);
  if ( v3 )
    v3 -= 24;
  if ( a2 == v3 || a3 == v3 )
    return v3;
  v3 = sub_15CC510(a1, a2);
  v5 = sub_15CC510(a1, a3);
  if ( !v5 || !v3 )
    return 0;
  while ( v3 != v5 )
  {
    if ( *(_DWORD *)(v3 + 16) < *(_DWORD *)(v5 + 16) )
    {
      v6 = v3;
      v3 = v5;
      v5 = v6;
    }
    v3 = *(_QWORD *)(v3 + 8);
    if ( !v3 )
      return v3;
  }
  return *(_QWORD *)v3;
}
