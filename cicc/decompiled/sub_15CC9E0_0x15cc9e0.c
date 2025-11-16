// Function: sub_15CC9E0
// Address: 0x15cc9e0
//
__int64 __fastcall sub_15CC9E0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r12
  __int64 v5; // rax
  __int64 v6; // rdx

  v4 = sub_15CC960(a1, a2);
  v5 = sub_15CC960(a1, a3);
  if ( !v5 || !v4 )
    return 0;
  while ( v4 != v5 )
  {
    if ( *(_DWORD *)(v4 + 16) < *(_DWORD *)(v5 + 16) )
    {
      v6 = v4;
      v4 = v5;
      v5 = v6;
    }
    v4 = *(_QWORD *)(v4 + 8);
    if ( !v4 )
      return 0;
  }
  return *(_QWORD *)v4;
}
