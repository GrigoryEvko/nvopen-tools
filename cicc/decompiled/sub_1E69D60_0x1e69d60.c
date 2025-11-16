// Function: sub_1E69D60
// Address: 0x1e69d60
//
__int64 __fastcall sub_1E69D60(__int64 a1, int a2)
{
  __int64 v2; // r8
  __int64 v3; // rax

  if ( a2 < 0 )
    v2 = *(_QWORD *)(*(_QWORD *)(a1 + 24) + 16LL * (a2 & 0x7FFFFFFF) + 8);
  else
    v2 = *(_QWORD *)(*(_QWORD *)(a1 + 272) + 8LL * (unsigned int)a2);
  if ( !v2 )
    return v2;
  v3 = *(_QWORD *)(v2 + 32);
  if ( (*(_BYTE *)(v2 + 3) & 0x10) == 0 )
  {
    v2 = 0;
    if ( !v3 || (*(_BYTE *)(v3 + 3) & 0x10) == 0 )
      return v2;
    v2 = v3;
    v3 = *(_QWORD *)(v3 + 32);
  }
  v2 = *(_QWORD *)(v2 + 16);
  while ( v3 && (*(_BYTE *)(v3 + 3) & 0x10) != 0 )
  {
    if ( v2 != *(_QWORD *)(v3 + 16) )
      return 0;
    v3 = *(_QWORD *)(v3 + 32);
  }
  return v2;
}
