// Function: sub_3286E00
// Address: 0x3286e00
//
__int64 __fastcall sub_3286E00(_DWORD *a1)
{
  int v1; // edx
  __int64 v2; // rax
  int v3; // ecx
  __int64 v4; // rax

  v1 = a1[2];
  v2 = *(_QWORD *)(*(_QWORD *)a1 + 56LL);
  if ( !v2 )
    return 0;
  v3 = 1;
  while ( 1 )
  {
    while ( v1 != *(_DWORD *)(v2 + 8) )
    {
      v2 = *(_QWORD *)(v2 + 32);
      if ( !v2 )
        return v3 ^ 1u;
    }
    if ( !v3 )
      return 0;
    v4 = *(_QWORD *)(v2 + 32);
    if ( !v4 )
      break;
    if ( v1 == *(_DWORD *)(v4 + 8) )
      return 0;
    v2 = *(_QWORD *)(v4 + 32);
    v3 = 0;
    if ( !v2 )
      return v3 ^ 1u;
  }
  return 1;
}
