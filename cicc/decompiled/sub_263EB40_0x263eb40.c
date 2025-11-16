// Function: sub_263EB40
// Address: 0x263eb40
//
__int64 __fastcall sub_263EB40(__int64 a1, unsigned __int64 *a2)
{
  __int64 v2; // rax
  __int64 v3; // r8
  unsigned __int64 v4; // rdx
  __int64 v5; // rcx

  v2 = *(_QWORD *)(a1 + 16);
  v3 = a1 + 8;
  if ( !v2 )
    return a1 + 8;
  v4 = *a2;
  v5 = a1 + 8;
  do
  {
    while ( *(_QWORD *)(v2 + 32) >= v4 && (*(_QWORD *)(v2 + 32) != v4 || *(_DWORD *)(v2 + 40) >= *((_DWORD *)a2 + 2)) )
    {
      v5 = v2;
      v2 = *(_QWORD *)(v2 + 16);
      if ( !v2 )
        goto LABEL_8;
    }
    v2 = *(_QWORD *)(v2 + 24);
  }
  while ( v2 );
LABEL_8:
  if ( v3 == v5 || *(_QWORD *)(v5 + 32) > v4 )
    return a1 + 8;
  if ( *(_QWORD *)(v5 + 32) != v4 )
    return v5;
  if ( *((_DWORD *)a2 + 2) >= *(_DWORD *)(v5 + 40) )
    return v5;
  return v3;
}
