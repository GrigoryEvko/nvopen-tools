// Function: sub_18A8380
// Address: 0x18a8380
//
__int64 __fastcall sub_18A8380(__int64 a1, unsigned int *a2)
{
  __int64 v2; // rax
  __int64 v3; // r8
  unsigned int v4; // edx
  __int64 v5; // rcx

  v2 = *(_QWORD *)(a1 + 16);
  v3 = a1 + 8;
  if ( !v2 )
    return v3;
  v4 = *a2;
  v5 = a1 + 8;
  do
  {
    while ( 1 )
    {
      if ( *(_DWORD *)(v2 + 32) < v4 )
      {
        v2 = *(_QWORD *)(v2 + 24);
        goto LABEL_7;
      }
      if ( *(_DWORD *)(v2 + 32) == v4 && *(_DWORD *)(v2 + 36) < a2[1] )
        break;
      v5 = v2;
      v2 = *(_QWORD *)(v2 + 16);
      if ( !v2 )
        goto LABEL_8;
    }
    v2 = *(_QWORD *)(v2 + 24);
LABEL_7:
    ;
  }
  while ( v2 );
LABEL_8:
  if ( v3 == v5 || *(_DWORD *)(v5 + 32) > v4 )
    return v3;
  if ( *(_DWORD *)(v5 + 32) != v4 )
    return v5;
  if ( a2[1] >= *(_DWORD *)(v5 + 36) )
    return v5;
  return v3;
}
