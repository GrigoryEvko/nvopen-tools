// Function: sub_C1BA30
// Address: 0xc1ba30
//
__int64 __fastcall sub_C1BA30(_QWORD *a1, __int64 a2)
{
  unsigned __int64 v3; // rdi
  _DWORD *v4; // r8
  _DWORD *v5; // rax
  unsigned __int64 v6; // rcx
  __int64 result; // rax

  v3 = a1[1];
  v4 = *(_DWORD **)(*a1 + 8 * (*(_QWORD *)a2 % v3));
  if ( !v4 )
    return 0;
  v5 = *(_DWORD **)v4;
  v6 = *(_QWORD *)(*(_QWORD *)v4 + 24LL);
  while ( *(_QWORD *)a2 != v6 || *(_DWORD *)a2 != v5[2] || *(_DWORD *)(a2 + 4) != v5[3] )
  {
    if ( !*(_QWORD *)v5 )
      return 0;
    v6 = *(_QWORD *)(*(_QWORD *)v5 + 24LL);
    v4 = v5;
    if ( *(_QWORD *)a2 % v3 != v6 % v3 )
      return 0;
    v5 = *(_DWORD **)v5;
  }
  result = *(_QWORD *)v4;
  if ( !*(_QWORD *)v4 )
    return 0;
  return result;
}
