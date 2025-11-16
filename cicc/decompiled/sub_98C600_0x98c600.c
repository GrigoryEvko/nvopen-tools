// Function: sub_98C600
// Address: 0x98c600
//
__int64 __fastcall sub_98C600(__int64 a1)
{
  __int64 v1; // rbx
  __int64 v2; // r12
  __int64 v4; // rax
  __int64 v5; // rdi

  v5 = *(_QWORD *)(a1 + 16);
  if ( !v5 )
    return 1;
  v1 = v5;
  do
  {
    v2 = *(_QWORD *)(v1 + 24);
    if ( *(_BYTE *)v2 != 85 )
      return 0;
    v4 = *(_QWORD *)(v2 - 32);
    if ( !v4
      || *(_BYTE *)v4
      || *(_QWORD *)(v4 + 24) != *(_QWORD *)(v2 + 80)
      || (*(_BYTE *)(v4 + 33) & 0x20) == 0
      || !(unsigned __int8)sub_B46A10(*(_QWORD *)(v1 + 24), 1) && !(unsigned __int8)sub_BD2BE0(v2) )
    {
      return 0;
    }
    v1 = *(_QWORD *)(v1 + 8);
  }
  while ( v1 );
  return 1;
}
