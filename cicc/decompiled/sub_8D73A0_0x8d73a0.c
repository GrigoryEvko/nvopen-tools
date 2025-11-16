// Function: sub_8D73A0
// Address: 0x8d73a0
//
_BOOL8 __fastcall sub_8D73A0(__int64 a1, __int64 a2)
{
  __int64 *v3; // rbx
  __int64 *v4; // r12
  __int128 v5; // rdi
  __int64 *v6; // rax
  __int64 v7; // rax
  __int64 v8; // rdx

  if ( ((*(_BYTE *)(*(_QWORD *)(a2 + 168) + 20LL) ^ *(_BYTE *)(*(_QWORD *)(a1 + 168) + 20LL)) & 2) != 0 )
    return 0;
  v3 = sub_736C60(19, *(__int64 **)(a1 + 104));
  v4 = sub_736C60(19, *(__int64 **)(a2 + 104));
  if ( !v4 || !v3 )
    return v3 == v4;
  while ( 1 )
  {
    v7 = v3[4];
    v8 = v4[4];
    if ( !v7 )
      break;
    if ( !v8 )
      break;
    if ( *(_BYTE *)(v7 + 10) != 5 )
      break;
    if ( *(_BYTE *)(v8 + 10) != 5 )
      break;
    *((_QWORD *)&v5 + 1) = *(_QWORD *)(v8 + 40);
    *(_QWORD *)&v5 = *(_QWORD *)(v7 + 40);
    if ( !(unsigned int)sub_7386E0(v5, 4u) )
      break;
    v3 = sub_736C60(19, (__int64 *)*v3);
    v6 = sub_736C60(19, (__int64 *)*v4);
    v4 = v6;
    if ( !v3 || !v6 )
      return v3 == v4;
  }
  return 0;
}
