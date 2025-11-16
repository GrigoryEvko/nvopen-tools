// Function: sub_7175E0
// Address: 0x7175e0
//
_BOOL8 __fastcall sub_7175E0(__int64 a1, _QWORD *a2)
{
  char v2; // al
  char v4; // al
  __int64 *v5; // rbx
  __int64 v6; // r13
  __int64 v7; // rax
  __int64 v8; // rdx

  if ( a2 )
    *a2 = 0;
  v2 = *(_BYTE *)(a1 + 24);
  if ( v2 == 2 )
    return (unsigned int)sub_717530(*(_QWORD *)(a1 + 56), a2) != 0;
  if ( v2 != 1 )
    return 0;
  v4 = *(_BYTE *)(a1 + 56);
  v5 = 0;
  if ( v4 == 5 )
  {
    v8 = *(_QWORD *)(a1 + 72);
    if ( *(_BYTE *)(v8 + 24) != 1 )
      return 0;
    v5 = (__int64 *)a1;
    v4 = *(_BYTE *)(v8 + 56);
    a1 = *(_QWORD *)(a1 + 72);
  }
  if ( v4 != 21 )
    return 0;
  v6 = *(_QWORD *)(a1 + 72);
  if ( (*(_BYTE *)(v6 + 25) & 1) == 0 )
    return 0;
  if ( *(_BYTE *)(v6 + 24) != 2 )
    return 0;
  if ( *(_BYTE *)(*(_QWORD *)(v6 + 56) + 173LL) != 2 )
    return 0;
  if ( v5 )
  {
    v7 = sub_8D67C0(*(_QWORD *)v6);
    if ( !sub_70D540(v7, *v5) )
      return 0;
  }
  if ( a2 )
    *a2 = *(_QWORD *)(v6 + 56);
  return 1;
}
