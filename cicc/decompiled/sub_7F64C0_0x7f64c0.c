// Function: sub_7F64C0
// Address: 0x7f64c0
//
__int64 __fastcall sub_7F64C0(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 v3; // rax
  __int64 v4; // rcx
  char v5; // dl

  v2 = a1;
  v3 = *(_QWORD *)(a1 + 72);
  if ( !v3 )
    return 0;
  while ( 1 )
  {
    v4 = 0;
    while ( *(_QWORD *)(v3 + 16) )
    {
      v4 = v3;
      v3 = *(_QWORD *)(v3 + 16);
    }
    v5 = *(_BYTE *)(v3 + 40);
    if ( v5 != 11 )
      break;
    v2 = v3;
    v3 = *(_QWORD *)(v3 + 72);
    if ( !v3 )
      return 0;
  }
  if ( v5 != 8 )
    return 0;
  if ( v4 )
    *(_QWORD *)(v4 + 16) = 0;
  else
    *(_QWORD *)(v2 + 72) = 0;
  *(_QWORD *)(v3 + 16) = *(_QWORD *)(a2 + 16);
  *(_QWORD *)(v3 + 24) = *(_QWORD *)(a2 + 24);
  *(_QWORD *)(a2 + 16) = v3;
  for ( *(_BYTE *)(*(_QWORD *)(a1 + 80) + 24LL) |= 1u; v2 != a1; *(_BYTE *)(*(_QWORD *)(a1 + 80) + 24LL) |= 1u )
    a1 = sub_7E2C20(a1);
  return 1;
}
