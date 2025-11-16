// Function: sub_214ACB0
// Address: 0x214acb0
//
__int64 __fastcall sub_214ACB0(__int64 a1, _QWORD *a2)
{
  unsigned __int8 v2; // al
  const char *v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v8; // rbx
  _QWORD *v9; // rax

  v2 = *(_BYTE *)(a1 + 16);
  if ( v2 == 3 )
  {
    v3 = sub_1649960(a1);
    if ( v4 == 9 && *(_QWORD *)v3 == 0x6573752E6D766C6CLL && v3[8] == 100 )
      return 1;
    v2 = *(_BYTE *)(a1 + 16);
  }
  if ( v2 <= 0x17u )
  {
    v8 = *(_QWORD *)(a1 + 8);
    if ( !v8 )
      return 1;
    while ( 1 )
    {
      v9 = sub_1648700(v8);
      if ( !(unsigned __int8)sub_214ACB0(v9, a2) )
        break;
      v8 = *(_QWORD *)(v8 + 8);
      if ( !v8 )
        return 1;
    }
    return 0;
  }
  v5 = *(_QWORD *)(a1 + 40);
  if ( !v5 )
    return 0;
  v6 = *(_QWORD *)(v5 + 56);
  if ( !v6 || *a2 && *a2 != v6 )
    return 0;
  *a2 = v6;
  return 1;
}
